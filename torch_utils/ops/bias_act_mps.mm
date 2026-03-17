// Objective-C++ bridge for the Metal bias_act compute shader.
//
// Uses PyTorch's internal MPS APIs for zero-copy dispatch:
//   - MetalShaderLibrary for shader compilation
//   - getMTLBufferStorage() for direct buffer access
//   - getCurrentMPSStream() for command encoding
//
// Build with torch.utils.cpp_extension (see bias_act.py).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <fstream>
#include <sstream>
#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSStream.h>

// -----------------------------------------------------------------------
// Must match the struct in bias_act.metal exactly.

struct bias_act_params {
    int   grad;
    int   act;
    float alpha;
    float gain;
    float clamp_val;
    int   sizeX;
    int   sizeB;
    int   stepB;
    int   has_xref;
    int   has_yref;
    int   has_dy;
};

// -----------------------------------------------------------------------
// Shader library singleton.  Compiled once from the .metal source file
// embedded as a string, then pipeline states are cached per kernel name.

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
            "Failed to compile bias_act Metal shader: ",
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
            "Metal function '", kernelName, "' not found in bias_act shader library");

        pso = [device newComputePipelineStateWithFunction:func error:&error];
        TORCH_CHECK(pso != nil,
            "Failed to create pipeline state for '", kernelName, "': ",
            error ? [[error localizedDescription] UTF8String] : "unknown error");

        _pipelineCache[name] = pso;
        return pso;
    }
}

// -----------------------------------------------------------------------
// Read the .metal source file and return its contents.

static std::string readMetalSource(const std::string& path) {
    std::ifstream f(path);
    TORCH_CHECK(f.is_open(), "Cannot open Metal shader file: ", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// -----------------------------------------------------------------------
// Main entry point called from Python.
//
// Signature mirrors the CUDA plugin:
//   bias_act_mps(x, b, xref, yref, dy, grad, dim, act, alpha, gain, clamp)
//
// Empty tensors (numel==0) signal "not provided".

static torch::Tensor bias_act_mps(
    torch::Tensor x,
    torch::Tensor b,
    torch::Tensor xref,
    torch::Tensor yref,
    torch::Tensor dy,
    int64_t grad,
    int64_t dim,
    int64_t act,
    double alpha,
    double gain,
    double clamp,
    const std::string& metalSourcePath)
{
    // --- Validate inputs ---------------------------------------------------
    TORCH_CHECK(x.is_mps(), "x must be on MPS device");
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");

    bool has_b    = b.numel() > 0;
    bool has_xref = xref.numel() > 0;
    bool has_yref = yref.numel() > 0;
    bool has_dy   = dy.numel() > 0;

    if (has_b) {
        TORCH_CHECK(b.dtype() == x.dtype() && b.is_mps(), "b must match x dtype/device");
        TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    }

    // --- Create output tensor (same shape/dtype/device as x) ---------------
    torch::Tensor y = torch::empty_like(x);

    int sizeX = (int)x.numel();
    if (sizeX == 0) return y;

    // --- Fill params -------------------------------------------------------
    bias_act_params p;
    p.grad      = (int)grad;
    p.act       = (int)act;
    p.alpha     = (float)alpha;
    p.gain      = (float)gain;
    p.clamp_val = (float)clamp;
    p.sizeX     = sizeX;
    p.sizeB     = has_b ? (int)b.numel() : 0;
    p.stepB     = has_b ? (int)x.stride(dim) : 1;
    p.has_xref  = has_xref ? 1 : 0;
    p.has_yref  = has_yref ? 1 : 0;
    p.has_dy    = has_dy ? 1 : 0;

    // --- Choose kernel name ------------------------------------------------
    std::string kernelName;
    if (x.scalar_type() == at::ScalarType::Float) {
        kernelName = "bias_act_float";
    } else if (x.scalar_type() == at::ScalarType::Half) {
        kernelName = "bias_act_half";
    } else {
        TORCH_CHECK(false, "bias_act_mps: unsupported dtype (need float32 or float16)");
    }

    // --- Compile shader & get pipeline state -------------------------------
    @autoreleasepool {
        using namespace at::mps;
        using at::native::mps::getMTLBufferStorage;

        id<MTLDevice> device = MPSDevice::getInstance()->device();
        ensureShaderLibrary(device, readMetalSource(metalSourcePath));
        id<MTLComputePipelineState> pso = getPipelineState(device, kernelName);

        // --- Encode compute command on the MPS stream ----------------------
        // Use the stream's shared encoder (standard PyTorch MPS pattern).
        MPSStream* stream = getCurrentMPSStream();
        dispatch_sync(stream->queue(), ^() {
            @autoreleasepool {
                id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

                [enc setComputePipelineState:pso];

                // Buffer 0: x
                id<MTLBuffer> xBuf = getMTLBufferStorage(x);
                [enc setBuffer:xBuf offset:x.storage_offset() * x.element_size() atIndex:0];

                // Buffer 1: b  (point to x if absent -- shader checks sizeB)
                if (has_b) {
                    id<MTLBuffer> bBuf = getMTLBufferStorage(b);
                    [enc setBuffer:bBuf offset:b.storage_offset() * b.element_size() atIndex:1];
                } else {
                    [enc setBuffer:xBuf offset:0 atIndex:1];
                }

                // Buffer 2: xref
                if (has_xref) {
                    id<MTLBuffer> xrefBuf = getMTLBufferStorage(xref);
                    [enc setBuffer:xrefBuf offset:xref.storage_offset() * xref.element_size() atIndex:2];
                } else {
                    [enc setBuffer:xBuf offset:0 atIndex:2];
                }

                // Buffer 3: yref
                if (has_yref) {
                    id<MTLBuffer> yrefBuf = getMTLBufferStorage(yref);
                    [enc setBuffer:yrefBuf offset:yref.storage_offset() * yref.element_size() atIndex:3];
                } else {
                    [enc setBuffer:xBuf offset:0 atIndex:3];
                }

                // Buffer 4: dy
                if (has_dy) {
                    id<MTLBuffer> dyBuf = getMTLBufferStorage(dy);
                    [enc setBuffer:dyBuf offset:dy.storage_offset() * dy.element_size() atIndex:4];
                } else {
                    [enc setBuffer:xBuf offset:0 atIndex:4];
                }

                // Buffer 5: y (output)
                id<MTLBuffer> yBuf = getMTLBufferStorage(y);
                [enc setBuffer:yBuf offset:y.storage_offset() * y.element_size() atIndex:5];

                // Buffer 6: params
                [enc setBytes:&p length:sizeof(p) atIndex:6];

                // --- Dispatch threadgroups ---------------------------------
                NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
                if (threadGroupSize > 256) threadGroupSize = 256;
                NSUInteger numGroups = ((NSUInteger)sizeX + threadGroupSize - 1) / threadGroupSize;

                [enc dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];

                // Don't end encoding -- the stream manages the shared encoder lifecycle.
            }
        });

        // Let PyTorch's MPS stream handle commit timing.
        stream->synchronize(SyncType::COMMIT_AND_CONTINUE);
    }

    return y;
}

// -----------------------------------------------------------------------
// pybind11 module.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bias_act", &bias_act_mps,
          "Fused bias+activation via Metal compute shader",
          py::arg("x"),
          py::arg("b"),
          py::arg("xref"),
          py::arg("yref"),
          py::arg("dy"),
          py::arg("grad"),
          py::arg("dim"),
          py::arg("act"),
          py::arg("alpha"),
          py::arg("gain"),
          py::arg("clamp"),
          py::arg("metal_source_path"));
}
