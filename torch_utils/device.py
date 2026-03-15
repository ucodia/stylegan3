# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Device abstraction utilities for cross-platform training (CUDA, MPS, CPU)."""

import time
import torch

#----------------------------------------------------------------------------
# Device selection.

def get_device(rank=0):
    """Select the best available device for training/inference.

    Priority: CUDA > MPS > CPU.

    Args:
        rank: GPU rank for multi-GPU CUDA setups. Ignored for MPS/CPU.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda', rank)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def is_cuda(device):
    """Check if a device is a CUDA device."""
    return device.type == 'cuda'

def is_mps(device):
    """Check if a device is an MPS (Apple Silicon) device."""
    return device.type == 'mps'

#----------------------------------------------------------------------------
# Backend configuration.

def configure_backends(device, cudnn_benchmark=True):
    """Configure torch backends based on device type.

    Args:
        device:           The target device.
        cudnn_benchmark:  Whether to enable cuDNN benchmarking (CUDA only).
    """
    if is_cuda(device):
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

#----------------------------------------------------------------------------
# Timing utilities that work across devices.

class DeviceTimer:
    """Cross-platform GPU timing that works on CUDA, MPS, and CPU.

    On CUDA, uses torch.cuda.Event for accurate GPU timing.
    On MPS/CPU, falls back to wall-clock time with synchronization.
    """
    def __init__(self, device):
        self.device = device
        self._use_cuda_events = is_cuda(device)
        self.start_event = None
        self.end_event = None
        self._start_time = None
        self._end_time = None

        if self._use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def record_start(self):
        """Record the start timestamp."""
        if self._use_cuda_events:
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self._synchronize()
            self._start_time = time.perf_counter()

    def record_end(self):
        """Record the end timestamp."""
        if self._use_cuda_events:
            self.end_event.record(torch.cuda.current_stream(self.device))
        else:
            self._synchronize()
            self._end_time = time.perf_counter()

    def elapsed_ms(self):
        """Return elapsed time in milliseconds between start and end.

        Must be called after record_end(). Synchronizes as needed.
        """
        if self._use_cuda_events:
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        if self._start_time is not None and self._end_time is not None:
            return (self._end_time - self._start_time) * 1000.0
        return 0.0

    def _synchronize(self):
        """Synchronize the device to get accurate wall-clock timing."""
        if is_cuda(self.device):
            torch.cuda.synchronize(self.device)
        elif is_mps(self.device):
            torch.mps.synchronize()

#----------------------------------------------------------------------------
# Memory reporting utilities.

def peak_memory_allocated_gb(device):
    """Return peak memory allocated in GB, or 0 if not trackable."""
    if is_cuda(device):
        return torch.cuda.max_memory_allocated(device) / 2**30
    if is_mps(device):
        try:
            return torch.mps.current_allocated_memory() / 2**30
        except AttributeError:
            return 0.0
    return 0.0

def peak_memory_reserved_gb(device):
    """Return peak memory reserved in GB, or 0 if not trackable."""
    if is_cuda(device):
        return torch.cuda.max_memory_reserved(device) / 2**30
    if is_mps(device):
        try:
            return torch.mps.driver_allocated_memory() / 2**30
        except AttributeError:
            return 0.0
    return 0.0

def reset_peak_memory_stats(device):
    """Reset peak memory tracking stats."""
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)

#----------------------------------------------------------------------------
# Pin memory helper.

def maybe_pin_memory(tensor, device):
    """Pin memory only if the target device supports it (CUDA).

    MPS uses unified memory so pinning is unnecessary and unsupported.
    """
    if is_cuda(device):
        return tensor.pin_memory()
    return tensor

#----------------------------------------------------------------------------
# Distributed backend selection.

def get_distributed_backend():
    """Select the appropriate distributed backend for the current platform.

    Returns 'nccl' for CUDA (Linux), 'gloo' for everything else.
    """
    import os
    if os.name == 'nt':
        return 'gloo'
    if torch.cuda.is_available():
        return 'nccl'
    return 'gloo'

#----------------------------------------------------------------------------
