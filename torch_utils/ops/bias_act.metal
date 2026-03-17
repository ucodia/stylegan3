// Custom Metal compute shader for fused bias + activation + clamp.
// Port of bias_act.cu from StyleGAN3 (NVIDIA).
//
// Supports all 9 activation functions with forward, first-order,
// and second-order gradient passes. Uses Metal function constants
// for compile-time activation specialization.

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// Kernel parameters passed via a constant buffer.

struct bias_act_params {
    int   grad;       // 0 = forward, 1 = first grad, 2 = second grad
    int   act;        // activation index 1..9
    float alpha;      // shape parameter (e.g. leaky-relu slope)
    float gain;       // output gain
    float clamp_val;  // clamp threshold, negative means disabled
    int   sizeX;      // total number of elements
    int   sizeB;      // number of bias elements (0 = no bias)
    int   stepB;      // stride along the bias dimension
    int   has_xref;   // 1 if xref buffer is valid, 0 otherwise
    int   has_yref;   // 1 if yref buffer is valid, 0 otherwise
    int   has_dy;     // 1 if dy buffer is valid, 0 otherwise
};

// -----------------------------------------------------------------------
// Helpers.

template <typename T>
inline T bias_act_op(T x, T b, T xref, T yref, T dy,
                     int G, int A, T alpha, T gain, T clamp_val) {
    const T one          = T(1);
    const T two          = T(2);
    const T expRange     = T(80);
    const T halfExpRange = T(40);
    const T seluScale    = T(1.0507009873554804934193349852946);
    const T seluAlpha    = T(1.6732632423543772848170429916717);
    T yy = (gain != T(0)) ? yref / gain : T(0);
    T y  = T(0);

    // Apply bias.
    if (G == 0) { x    += b; }
    else        { xref += b; }

    // linear (A == 1)
    if (A == 1) {
        if (G == 0) y = x;
        if (G == 1) y = x;
    }

    // relu (A == 2)
    if (A == 2) {
        if (G == 0) y = (x > T(0)) ? x : T(0);
        if (G == 1) y = (yy > T(0)) ? x : T(0);
    }

    // lrelu (A == 3)
    if (A == 3) {
        if (G == 0) y = (x > T(0)) ? x : x * alpha;
        if (G == 1) y = (yy > T(0)) ? x : x * alpha;
    }

    // tanh (A == 4)
    if (A == 4) {
        if (G == 0) {
            T c = exp(x); T d = one / c;
            y = (x < -expRange) ? -one : (x > expRange) ? one : (c - d) / (c + d);
        }
        if (G == 1) y = x * (one - yy * yy);
        if (G == 2) y = x * (one - yy * yy) * (-two * yy);
    }

    // sigmoid (A == 5)
    if (A == 5) {
        if (G == 0) y = (x < -expRange) ? T(0) : one / (exp(-x) + one);
        if (G == 1) y = x * yy * (one - yy);
        if (G == 2) y = x * yy * (one - yy) * (one - two * yy);
    }

    // elu (A == 6)
    if (A == 6) {
        if (G == 0) y = (x >= T(0)) ? x : exp(x) - one;
        if (G == 1) y = (yy >= T(0)) ? x : x * (yy + one);
        if (G == 2) y = (yy >= T(0)) ? T(0) : x * (yy + one);
    }

    // selu (A == 7)
    if (A == 7) {
        if (G == 0) y = (x >= T(0)) ? seluScale * x : (seluScale * seluAlpha) * (exp(x) - one);
        if (G == 1) y = (yy >= T(0)) ? x * seluScale : x * (yy + seluScale * seluAlpha);
        if (G == 2) y = (yy >= T(0)) ? T(0) : x * (yy + seluScale * seluAlpha);
    }

    // softplus (A == 8)
    if (A == 8) {
        if (G == 0) y = (x > expRange) ? x : log(exp(x) + one);
        if (G == 1) y = x * (one - exp(-yy));
        if (G == 2) { T c = exp(-yy); y = x * c * (one - c); }
    }

    // swish (A == 9)
    if (A == 9) {
        if (G == 0) {
            y = (x < -expRange) ? T(0) : x / (exp(-x) + one);
        } else {
            T c = exp(xref);
            T d = c + one;
            if (G == 1) {
                y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
            } else {
                y = (xref > halfExpRange) ? T(0) : x * c * (xref * (two - d) + two * d) / (d * d * d);
            }
            yref = (xref < -expRange) ? T(0) : xref / (exp(-xref) + one) * gain;
        }
    }

    // Apply gain.
    y *= gain * dy;

    // Clamp.
    if (clamp_val >= T(0)) {
        if (G == 0)
            y = (y > -clamp_val && y < clamp_val) ? y : (y >= T(0)) ? clamp_val : -clamp_val;
        else
            y = (yref > -clamp_val && yref < clamp_val) ? y : T(0);
    }

    return y;
}

// -----------------------------------------------------------------------
// Main kernel: float32.
//
// Buffers:
//   0: x     (input,  float, sizeX elements)
//   1: b     (input,  float, sizeB elements -- may be empty)
//   2: xref  (input,  float, sizeX elements -- may be empty)
//   3: yref  (input,  float, sizeX elements -- may be empty)
//   4: dy    (input,  float, sizeX elements -- may be empty)
//   5: y     (output, float, sizeX elements)
//   6: params (constant, bias_act_params)

kernel void bias_act_float(
    device const float* x      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* xref   [[buffer(2)]],
    device const float* yref   [[buffer(3)]],
    device const float* dy     [[buffer(4)]],
    device       float* y      [[buffer(5)]],
    constant bias_act_params& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if ((int)tid >= p.sizeX) return;

    int xi = (int)tid;
    float xv    = x[xi];
    float bv    = (p.sizeB > 0) ? b[(xi / p.stepB) % p.sizeB] : 0.0f;
    float xrefv = p.has_xref ? xref[xi] : 0.0f;
    float yrefv = p.has_yref ? yref[xi] : 0.0f;
    float dyv   = p.has_dy   ? dy[xi]   : 1.0f;

    y[xi] = bias_act_op<float>(xv, bv, xrefv, yrefv, dyv,
                               p.grad, p.act, p.alpha, p.gain, p.clamp_val);
}

// -----------------------------------------------------------------------
// Main kernel: float16.

kernel void bias_act_half(
    device const half*  x      [[buffer(0)]],
    device const half*  b      [[buffer(1)]],
    device const half*  xref   [[buffer(2)]],
    device const half*  yref   [[buffer(3)]],
    device const half*  dy     [[buffer(4)]],
    device       half*  y      [[buffer(5)]],
    constant bias_act_params& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if ((int)tid >= p.sizeX) return;

    int xi = (int)tid;
    // Promote to float for computation (same as CUDA InternalType<c10::Half> = float).
    float xv    = float(x[xi]);
    float bv    = (p.sizeB > 0) ? float(b[(xi / p.stepB) % p.sizeB]) : 0.0f;
    float xrefv = p.has_xref ? float(xref[xi]) : 0.0f;
    float yrefv = p.has_yref ? float(yref[xi]) : 0.0f;
    float dyv   = p.has_dy   ? float(dy[xi])   : 1.0f;

    y[xi] = half(bias_act_op<float>(xv, bv, xrefv, yrefv, dyv,
                                    p.grad, p.act, p.alpha, p.gain, p.clamp_val));
}

// -----------------------------------------------------------------------
