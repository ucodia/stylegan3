#!/usr/bin/env python3
"""Test script to verify the Metal bias_act kernel against the reference implementation.

Run on macOS with Apple Silicon:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python test_bias_act_mps.py
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

# Ensure project root is on path.
sys.path.insert(0, os.path.dirname(__file__))

from torch_utils.ops.bias_act import bias_act, _bias_act_ref, activation_funcs, _init_mps

# ---------------------------------------------------------------------------

def test_forward(act_name, shape=(4, 64, 16, 16), use_bias=True, clamp_val=None):
    """Compare Metal forward pass against reference implementation."""
    spec = activation_funcs[act_name]
    alpha = spec.def_alpha
    gain = spec.def_gain

    x_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape[1], dtype=torch.float32) if use_bias else None

    # Reference (CPU).
    x_ref = x_cpu.clone().requires_grad_(False)
    b_ref = b_cpu.clone() if b_cpu is not None else None
    y_ref = _bias_act_ref(x=x_ref, b=b_ref, dim=1, act=act_name, alpha=alpha, gain=gain, clamp=clamp_val)

    # MPS Metal kernel.
    x_mps = x_cpu.clone().to('mps')
    b_mps = b_cpu.clone().to('mps') if b_cpu is not None else None
    y_mps = bias_act(x_mps, b=b_mps, dim=1, act=act_name, alpha=alpha, gain=gain, clamp=clamp_val, impl='cuda')
    y_mps_cpu = y_mps.cpu()

    # Compare.
    max_diff = (y_ref - y_mps_cpu).abs().max().item()
    mean_diff = (y_ref - y_mps_cpu).abs().mean().item()
    return max_diff, mean_diff


def test_backward(act_name, shape=(4, 64, 8, 8), clamp_val=None):
    """Compare Metal backward pass (first-order gradient) against reference."""
    spec = activation_funcs[act_name]
    alpha = spec.def_alpha
    gain = spec.def_gain

    x_cpu = torch.randn(shape, dtype=torch.float32)
    b_cpu = torch.randn(shape[1], dtype=torch.float32)

    # Reference gradient.
    x_ref = x_cpu.clone().requires_grad_(True)
    b_ref = b_cpu.clone().requires_grad_(True)
    y_ref = _bias_act_ref(x=x_ref, b=b_ref, dim=1, act=act_name, alpha=alpha, gain=gain, clamp=clamp_val)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    dx_ref = x_ref.grad.clone()
    db_ref = b_ref.grad.clone()

    # MPS gradient.
    x_mps = x_cpu.clone().to('mps').requires_grad_(True)
    b_mps = b_cpu.clone().to('mps').requires_grad_(True)
    y_mps = bias_act(x_mps, b=b_mps, dim=1, act=act_name, alpha=alpha, gain=gain, clamp=clamp_val, impl='cuda')
    loss_mps = y_mps.sum()
    loss_mps.backward()
    dx_mps = x_mps.grad.cpu()
    db_mps = b_mps.grad.cpu()

    dx_max = (dx_ref - dx_mps).abs().max().item()
    db_max = (db_ref - db_mps).abs().max().item()
    return dx_max, db_max


# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Testing Metal bias_act kernel")
    print("=" * 60)

    # Check MPS availability.
    if not torch.backends.mps.is_available():
        print("MPS not available. Exiting.")
        sys.exit(1)

    # Try to compile the MPS plugin.
    print("\nCompiling MPS plugin...")
    if not _init_mps():
        print("ERROR: Failed to compile MPS bias_act plugin.")
        sys.exit(1)
    print("MPS plugin compiled successfully.\n")

    # Test forward pass for each activation.
    acts_to_test = ['linear', 'relu', 'lrelu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus', 'swish']

    print("--- Forward pass tests ---")
    all_pass = True
    for act_name in acts_to_test:
        max_diff, mean_diff = test_forward(act_name)
        status = "PASS" if max_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {act_name:10s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

    # Test with clamping.
    print("\n--- Forward pass with clamp=1.0 ---")
    for act_name in ['lrelu', 'swish']:
        max_diff, mean_diff = test_forward(act_name, clamp_val=1.0)
        status = "PASS" if max_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {act_name:10s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

    # Test without bias.
    print("\n--- Forward pass without bias ---")
    for act_name in ['lrelu', 'relu']:
        max_diff, mean_diff = test_forward(act_name, use_bias=False)
        status = "PASS" if max_diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {act_name:10s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

    # Test backward pass.
    print("\n--- Backward pass tests (first-order gradient) ---")
    for act_name in acts_to_test:
        try:
            dx_max, db_max = test_backward(act_name)
            status = "PASS" if dx_max < 1e-4 and db_max < 1e-4 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {act_name:10s}  dx_max={dx_max:.2e}  db_max={db_max:.2e}  [{status}]")
        except Exception as e:
            all_pass = False
            print(f"  {act_name:10s}  ERROR: {e}")

    print("\n" + "=" * 60)
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED -- check output above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
