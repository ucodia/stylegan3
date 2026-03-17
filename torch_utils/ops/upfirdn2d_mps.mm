// Objective-C++ bridge for the Metal upfirdn2d compute shader.
//
// Uses PyTorch's internal MPS APIs for zero-copy dispatch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fstream>
#include <sstream>
#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSStream.h>

// -----------------------------------------------------------------------
// Must match the struct in upfirdn2d.metal exactly.

struct upfirdn2d_params {
    int   upx, upy;
    int   downx, downy;
    int   padx0, pady0;
    int   flip;
    float gain;

    int   inW, inH, inC, inN;
    int   inStrideW, inStrideH, inStrideC, inStrideN;

    int   filterW, filterH;
    int   filterStrideW, filterStrideH;

    int   outW, outH, outC, outN;
    int   outStrideW, outStrideH, outStrideC, outStrideN;
};

// -----------------------------------------------------------------------
// Shader library singleton.

static id<MTLLibrary> _shaderLibrary = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* _pipelineCache = nil;

static void ensureShaderLibrary(id<MTLDevice> device, const std::string& metalSource) {
    if (_shaderLibrary != nil) return;

    @autoreleasepool {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:metalSource.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion2_4;

        _shaderLibrary = [device newLibraryWithSource:src options:opts error:&error];
        TORCH_CHECK(_shaderLibrary != nil,
            "Failed to compile upfirdn2d Metal shader: ",
            error ? [[error localizedDescription] UTF8String] : "unknown error");

        _pipelineCache = [NSMutableDictionary new];
    }
}

static id<MTLComputePipelineState> getPipelineState(
    id<MTLDevice> device, const std::string& kernelName) {

    @autoreleasepool {
        NSString* name = [NSString stringWithUTF8String:kernelName.c_str()];
        id<MTLComputePipelineState> pso = _pipelineCache[name];
        if (pso != nil) return pso;

        NSError* error = nil;
        id<MTLFunction> func = [_shaderLibrary newFunctionWithName:name];
        TORCH_CHECK(func != nil,
            "Metal function '", kernelName, "' not found in upfirdn2d shader library");

        pso = [device newComputePipelineStateWithFunction:func error:&error];
        TORCH_CHECK(pso != nil,
            "Failed to create pipeline state for '", kernelName, "': ",
            error ? [[error localizedDescription] UTF8String] : "unknown error");

        _pipelineCache[name] = pso;
        return pso;
    }
}

static std::string readMetalSource(const std::string& path) {
    std::ifstream f(path);
    TORCH_CHECK(f.is_open(), "Cannot open Metal shader file: ", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// -----------------------------------------------------------------------
// Main entry point: upfirdn2d_mps
//
// Signature mirrors the CUDA plugin.

static torch::Tensor upfirdn2d_mps(
    torch::Tensor x,
    torch::Tensor f,
    int64_t upx, int64_t upy,
    int64_t downx, int64_t downy,
    int64_t padx0, int64_t padx1,
    int64_t pady0, int64_t pady1,
    bool flip,
    double gain,
    const std::string& metalSourcePath)
{
    // --- Validate inputs ---------------------------------------------------
    TORCH_CHECK(x.is_mps(), "x must be on MPS device");
    TORCH_CHECK(f.device() == x.device(), "f must be on the same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");

    // --- Create output tensor ----------------------------------------------
    int outW = ((int)x.size(3) * (int)upx + (int)padx0 + (int)padx1 - (int)f.size(1) + (int)downx) / (int)downx;
    int outH = ((int)x.size(2) * (int)upy + (int)pady0 + (int)pady1 - (int)f.size(0) + (int)downy) / (int)downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");

    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW}, x.options(), x.suggest_memory_format());

    // --- Fill params -------------------------------------------------------
    upfirdn2d_params p;
    p.upx   = (int)upx;   p.upy   = (int)upy;
    p.downx = (int)downx;  p.downy = (int)downy;
    p.padx0 = (int)padx0;  p.pady0 = (int)pady0;
    p.flip  = flip ? 1 : 0;
    p.gain  = (float)gain;

    p.inW = (int)x.size(3);  p.inH = (int)x.size(2);
    p.inC = (int)x.size(1);  p.inN = (int)x.size(0);
    p.inStrideW = (int)x.stride(3);  p.inStrideH = (int)x.stride(2);
    p.inStrideC = (int)x.stride(1);  p.inStrideN = (int)x.stride(0);

    p.filterW = (int)f.size(1);       p.filterH = (int)f.size(0);
    p.filterStrideW = (int)f.stride(1); p.filterStrideH = (int)f.stride(0);

    p.outW = outW;                     p.outH = outH;
    p.outC = (int)y.size(1);          p.outN = (int)y.size(0);
    p.outStrideW = (int)y.stride(3);  p.outStrideH = (int)y.stride(2);
    p.outStrideC = (int)y.stride(1);  p.outStrideN = (int)y.stride(0);

    // --- Choose kernel -----------------------------------------------------
    std::string kernelName;
    if (x.scalar_type() == at::ScalarType::Float) {
        kernelName = "upfirdn2d_large_float";
    } else if (x.scalar_type() == at::ScalarType::Half) {
        kernelName = "upfirdn2d_large_half";
    } else {
        TORCH_CHECK(false, "upfirdn2d_mps: unsupported dtype (need float32 or float16)");
    }

    int totalPixels = outW * outH;
    int totalNC     = p.inN * p.inC;

    if (totalPixels == 0 || totalNC == 0) return y;

    // --- Compile shader & dispatch -----------------------------------------
    @autoreleasepool {
        using namespace at::mps;
        using at::native::mps::getMTLBufferStorage;

        id<MTLDevice> device = MPSDevice::getInstance()->device();
        ensureShaderLibrary(device, readMetalSource(metalSourcePath));
        id<MTLComputePipelineState> pso = getPipelineState(device, kernelName);

        MPSStream* stream = getCurrentMPSStream();
        dispatch_sync(stream->queue(), ^() {
            @autoreleasepool {
                id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
                [enc setComputePipelineState:pso];

                // Buffer 0: x (input)
                id<MTLBuffer> xBuf = getMTLBufferStorage(x);
                [enc setBuffer:xBuf offset:x.storage_offset() * x.element_size() atIndex:0];

                // Buffer 1: f (filter, always float32)
                id<MTLBuffer> fBuf = getMTLBufferStorage(f);
                [enc setBuffer:fBuf offset:f.storage_offset() * f.element_size() atIndex:1];

                // Buffer 2: y (output)
                id<MTLBuffer> yBuf = getMTLBufferStorage(y);
                [enc setBuffer:yBuf offset:y.storage_offset() * y.element_size() atIndex:2];

                // Buffer 3: params
                [enc setBytes:&p length:sizeof(p) atIndex:3];

                // --- Dispatch 2D grid: (outW*outH, N*C) -------------------
                NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
                if (threadGroupSize > 256) threadGroupSize = 256;

                // 2D dispatch: x-dim = output pixels, y-dim = batch*channel
                NSUInteger groupsX = ((NSUInteger)totalPixels + threadGroupSize - 1) / threadGroupSize;
                NSUInteger groupsY = (NSUInteger)totalNC;

                [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
            }
        });

        stream->synchronize(SyncType::COMMIT_AND_CONTINUE);
    }

    return y;
}

// -----------------------------------------------------------------------
// pybind11 module.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upfirdn2d", &upfirdn2d_mps,
          "Fused upsample+FIR filter+downsample via Metal compute shader",
          py::arg("x"),
          py::arg("f"),
          py::arg("upx"),
          py::arg("upy"),
          py::arg("downx"),
          py::arg("downy"),
          py::arg("padx0"),
          py::arg("padx1"),
          py::arg("pady0"),
          py::arg("pady1"),
          py::arg("flip"),
          py::arg("gain"),
          py::arg("metal_source_path"));
}
