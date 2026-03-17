#!/usr/bin/env python3
"""Test script to verify all Metal compute shaders against reference implementations.

Run on macOS with Apple Silicon:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python test_metal_kernels.py
"""

import sys
import os
import platform

if platform.system() != 'Darwin':
    print("This test must be run on macOS with Apple Silicon.")
    sys.exit(1)

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from torch_utils.ops.bias_act import bias_act, _bias_act_ref, activation_funcs
from torch_utils.ops.bias_act import _init_mps as _init_bias_act_mps
from torch_utils.ops.upfirdn2d import upfirdn2d, _upfirdn2d_ref, setup_filter
from torch_utils.ops.upfirdn2d import _init_mps as _init_upfirdn2d_mps

# ---------------------------------------------------------------------------
# bias_act tests
# ---------------------------------------------------------------------------

def test_bias_act_forward(act_name, shape=(4, 64, 16, 16), use_bias=True, clamp_val=None):
    spec = activation_funcs[act_name]
    x_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape[1], dtype=torch.float32) if use_bias else None

    x_ref = x_cpu.clone()
    b_ref = b_cpu.clone() if b_cpu is not None else None
    y_ref = _bias_act_ref(x=x_ref, b=b_ref, dim=1, act=act_name,
                          alpha=spec.def_alpha, gain=spec.def_gain, clamp=clamp_val)

    x_mps = x_cpu.clone().to('mps')
    b_mps = b_cpu.clone().to('mps') if b_cpu is not None else None
    y_mps = bias_act(x_mps, b=b_mps, dim=1, act=act_name,
                     alpha=spec.def_alpha, gain=spec.def_gain, clamp=clamp_val, impl='cuda')

    max_diff = (y_ref - y_mps.cpu()).abs().max().item()
    return max_diff


def test_bias_act_backward(act_name, shape=(4, 64, 8, 8)):
    spec = activation_funcs[act_name]
    x_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape[1], dtype=torch.float32)

    x_ref = x_cpu.clone().requires_grad_(True)
    b_ref = b_cpu.clone().requires_grad_(True)
    y_ref = _bias_act_ref(x=x_ref, b=b_ref, dim=1, act=act_name,
                          alpha=spec.def_alpha, gain=spec.def_gain)
    y_ref.sum().backward()

    x_mps = x_cpu.clone().to('mps').requires_grad_(True)
    b_mps = b_cpu.clone().to('mps').requires_grad_(True)
    y_mps = bias_act(x_mps, b=b_mps, dim=1, act=act_name,
                     alpha=spec.def_alpha, gain=spec.def_gain, impl='cuda')
    y_mps.sum().backward()

    dx_max = (x_ref.grad - x_mps.grad.cpu()).abs().max().item()
    db_max = (b_ref.grad - b_mps.grad.cpu()).abs().max().item()
    return dx_max, db_max

# ---------------------------------------------------------------------------
# upfirdn2d tests
# ---------------------------------------------------------------------------

def test_upfirdn2d_forward(up=1, down=1, filter_size=4, shape=(2, 32, 16, 16), flip=False):
    f_cpu = setup_filter([1] * filter_size, normalize=True)
    if f_cpu.ndim == 1:
        f_cpu = f_cpu.ger(f_cpu)

    x_cpu = torch.randn(shape, dtype=torch.float32)

    y_ref = _upfirdn2d_ref(x_cpu, f_cpu, up=up, down=down, flip_filter=flip)

    x_mps = x_cpu.to('mps')
    f_mps = f_cpu.to('mps')
    y_mps = upfirdn2d(x_mps, f_mps, up=up, down=down, flip_filter=flip, impl='cuda')

    max_diff = (y_ref - y_mps.cpu()).abs().max().item()
    return max_diff


def test_upfirdn2d_backward(up=1, down=1, filter_size=4, shape=(2, 16, 8, 8)):
    f_cpu = setup_filter([1] * filter_size, normalize=True)
    if f_cpu.ndim == 1:
        f_cpu = f_cpu.ger(f_cpu)

    x_cpu = torch.randn(shape, dtype=torch.float32)

    x_ref = x_cpu.clone().requires_grad_(True)
    y_ref = _upfirdn2d_ref(x_ref, f_cpu, up=up, down=down)
    y_ref.sum().backward()

    x_mps = x_cpu.clone().to('mps').requires_grad_(True)
    f_mps = f_cpu.to('mps')
    y_mps = upfirdn2d(x_mps, f_mps, up=up, down=down, impl='cuda')
    y_mps.sum().backward()

    dx_max = (x_ref.grad - x_mps.grad.cpu()).abs().max().item()
    return dx_max

# ---------------------------------------------------------------------------

def main():
    if not torch.backends.mps.is_available():
        print("MPS not available.")
        sys.exit(1)

    all_pass = True

    # ===== bias_act =====
    print("=" * 60)
    print("Testing Metal bias_act kernel")
    print("=" * 60)

    print("\nCompiling bias_act MPS plugin...")
    if not _init_bias_act_mps():
        print("ERROR: Failed to compile. Skipping bias_act tests.")
    else:
        print("OK\n")

        acts = ['linear', 'relu', 'lrelu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus', 'swish']

        print("--- Forward ---")
        for act in acts:
            d = test_bias_act_forward(act)
            ok = d < 1e-5
            if not ok: all_pass = False
            print(f"  {act:10s}  max_diff={d:.2e}  [{'PASS' if ok else 'FAIL'}]")

        print("\n--- Forward (clamp=1.0) ---")
        for act in ['lrelu', 'swish']:
            d = test_bias_act_forward(act, clamp_val=1.0)
            ok = d < 1e-5
            if not ok: all_pass = False
            print(f"  {act:10s}  max_diff={d:.2e}  [{'PASS' if ok else 'FAIL'}]")

        print("\n--- Backward ---")
        for act in acts:
            try:
                dx, db = test_bias_act_backward(act)
                ok = dx < 1e-4 and db < 1e-4
                if not ok: all_pass = False
                print(f"  {act:10s}  dx={dx:.2e}  db={db:.2e}  [{'PASS' if ok else 'FAIL'}]")
            except Exception as e:
                all_pass = False
                print(f"  {act:10s}  ERROR: {e}")

    # ===== upfirdn2d =====
    print("\n" + "=" * 60)
    print("Testing Metal upfirdn2d kernel")
    print("=" * 60)

    print("\nCompiling upfirdn2d MPS plugin...")
    if not _init_upfirdn2d_mps():
        print("ERROR: Failed to compile. Skipping upfirdn2d tests.")
    else:
        print("OK\n")

        configs = [
            {"label": "identity (1x1 filter)",  "up": 1, "down": 1, "filter_size": 1},
            {"label": "filter only (4-tap)",     "up": 1, "down": 1, "filter_size": 4},
            {"label": "2x upsample + 4-tap",    "up": 2, "down": 1, "filter_size": 4},
            {"label": "2x downsample + 4-tap",   "up": 1, "down": 2, "filter_size": 4},
            {"label": "2x up + 2x down + 4-tap", "up": 2, "down": 2, "filter_size": 4},
            {"label": "filter only (6-tap)",     "up": 1, "down": 1, "filter_size": 6},
        ]

        print("--- Forward ---")
        for cfg in configs:
            d = test_upfirdn2d_forward(up=cfg["up"], down=cfg["down"], filter_size=cfg["filter_size"])
            ok = d < 1e-4
            if not ok: all_pass = False
            print(f"  {cfg['label']:30s}  max_diff={d:.2e}  [{'PASS' if ok else 'FAIL'}]")

        print("\n--- Forward (flip_filter=True) ---")
        for cfg in configs[:3]:
            d = test_upfirdn2d_forward(up=cfg["up"], down=cfg["down"], filter_size=cfg["filter_size"], flip=True)
            ok = d < 1e-4
            if not ok: all_pass = False
            print(f"  {cfg['label']:30s}  max_diff={d:.2e}  [{'PASS' if ok else 'FAIL'}]")

        print("\n--- Backward ---")
        for cfg in configs[:4]:
            try:
                d = test_upfirdn2d_backward(up=cfg["up"], down=cfg["down"], filter_size=cfg["filter_size"])
                ok = d < 1e-4
                if not ok: all_pass = False
                print(f"  {cfg['label']:30s}  dx_max={d:.2e}  [{'PASS' if ok else 'FAIL'}]")
            except Exception as e:
                all_pass = False
                print(f"  {cfg['label']:30s}  ERROR: {e}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED -- check output above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
