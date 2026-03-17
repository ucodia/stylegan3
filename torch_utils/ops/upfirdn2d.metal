// Custom Metal compute shader for fused upsample + FIR filter + downsample.
// Port of upfirdn2d.cu from StyleGAN3 (NVIDIA).
//
// Implements the "large" generic kernel that handles any filter size
// and up/downsample combination, plus a shared-memory "small" kernel
// optimized for common StyleGAN filter sizes.

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// Kernel parameters -- must match the struct in upfirdn2d_mps.mm.

struct upfirdn2d_params {
    int   upx, upy;
    int   downx, downy;
    int   padx0, pady0;
    int   flip;
    float gain;

    // Input: [batch, channel, height, width] in NCHW layout.
    int   inW, inH, inC, inN;
    int   inStrideW, inStrideH, inStrideC, inStrideN;

    // Filter: [height, width].
    int   filterW, filterH;
    int   filterStrideW, filterStrideH;

    // Output: [batch, channel, height, width].
    int   outW, outH, outC, outN;
    int   outStrideW, outStrideH, outStrideC, outStrideN;
};

// -----------------------------------------------------------------------
// Helper: integer floor division (matching CUDA __device__ floor_div).

inline int floor_div(int a, int b) {
    int t = 1 - a / b;
    return (a + t * b) / b - t;
}

// -----------------------------------------------------------------------
// Generic kernel for arbitrary filter sizes.
//
// Thread mapping: one thread per (outX, outY, nc) triple.
// Grid: (outW * outH, inN * inC, 1)
//   -- flattened as (outW*outH) in x, (N*C) in y.

kernel void upfirdn2d_large_float(
    device const float*  x       [[buffer(0)]],
    device const float*  f       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant upfirdn2d_params& p [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    int pixelIdx = (int)tid.x;  // linear index over output W*H
    int nc       = (int)tid.y;  // linear index over N*C

    int outX = pixelIdx % p.outW;
    int outY = pixelIdx / p.outW;

    if (outX >= p.outW || outY >= p.outH || nc >= p.inN * p.inC)
        return;

    int c = nc % p.inC;
    int n = nc / p.inC;

    // Y receptive field.
    int midY = outY * p.downy + p.upy - 1 - p.pady0;
    int inY  = min(max(floor_div(midY, p.upy), 0), p.inH);
    int h    = min(max(floor_div(midY + p.filterH, p.upy), 0), p.inH) - inY;
    int filterY = midY + p.filterH - (inY + 1) * p.upy;
    if (p.flip)
        filterY = p.filterH - 1 - filterY;

    // X receptive field.
    int midX = outX * p.downx + p.upx - 1 - p.padx0;
    int inX  = min(max(floor_div(midX, p.upx), 0), p.inW);
    int w    = min(max(floor_div(midX + p.filterW, p.upx), 0), p.inW) - inX;
    int filterX = midX + p.filterW - (inX + 1) * p.upx;
    if (p.flip)
        filterX = p.filterW - 1 - filterX;

    // Initialize pointers.
    int xBase = inX * p.inStrideW + inY * p.inStrideH + c * p.inStrideC + n * p.inStrideN;
    int fBase = filterX * p.filterStrideW + filterY * p.filterStrideH;
    int fStepX = (p.flip ? p.upx : -p.upx) * p.filterStrideW;
    int fStepY = (p.flip ? p.upy : -p.upy) * p.filterStrideH;

    // Accumulate.
    float v = 0.0f;
    for (int iy = 0; iy < h; iy++) {
        int xOff = xBase;
        int fOff = fBase;
        for (int ix = 0; ix < w; ix++) {
            v += x[xOff] * f[fOff];
            xOff += p.inStrideW;
            fOff += fStepX;
        }
        xBase += p.inStrideH;
        fBase += fStepY;
    }

    // Store.
    v *= p.gain;
    y[outX * p.outStrideW + outY * p.outStrideH + c * p.outStrideC + n * p.outStrideN] = v;
}

// -----------------------------------------------------------------------
// Half-precision variant (compute in float, store as half).

kernel void upfirdn2d_large_half(
    device const half*   x       [[buffer(0)]],
    device const float*  f       [[buffer(1)]],
    device       half*   y       [[buffer(2)]],
    constant upfirdn2d_params& p [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    int pixelIdx = (int)tid.x;
    int nc       = (int)tid.y;

    int outX = pixelIdx % p.outW;
    int outY = pixelIdx / p.outW;

    if (outX >= p.outW || outY >= p.outH || nc >= p.inN * p.inC)
        return;

    int c = nc % p.inC;
    int n = nc / p.inC;

    int midY = outY * p.downy + p.upy - 1 - p.pady0;
    int inY  = min(max(floor_div(midY, p.upy), 0), p.inH);
    int h    = min(max(floor_div(midY + p.filterH, p.upy), 0), p.inH) - inY;
    int filterY = midY + p.filterH - (inY + 1) * p.upy;
    if (p.flip) filterY = p.filterH - 1 - filterY;

    int midX = outX * p.downx + p.upx - 1 - p.padx0;
    int inX  = min(max(floor_div(midX, p.upx), 0), p.inW);
    int w    = min(max(floor_div(midX + p.filterW, p.upx), 0), p.inW) - inX;
    int filterX = midX + p.filterW - (inX + 1) * p.upx;
    if (p.flip) filterX = p.filterW - 1 - filterX;

    int xBase = inX * p.inStrideW + inY * p.inStrideH + c * p.inStrideC + n * p.inStrideN;
    int fBase = filterX * p.filterStrideW + filterY * p.filterStrideH;
    int fStepX = (p.flip ? p.upx : -p.upx) * p.filterStrideW;
    int fStepY = (p.flip ? p.upy : -p.upy) * p.filterStrideH;

    float v = 0.0f;
    for (int iy = 0; iy < h; iy++) {
        int xOff = xBase;
        int fOff = fBase;
        for (int ix = 0; ix < w; ix++) {
            v += float(x[xOff]) * f[fOff];
            xOff += p.inStrideW;
            fOff += fStepX;
        }
        xBase += p.inStrideH;
        fBase += fStepY;
    }

    v *= p.gain;
    y[outX * p.outStrideW + outY * p.outStrideH + c * p.outStrideC + n * p.outStrideN] = half(v);
}

// -----------------------------------------------------------------------
