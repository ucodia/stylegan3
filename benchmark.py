"""Benchmark raw generator inference speed for StyleGAN2/3 pickles.

Runs a warmup phase then a timed loop against a user-supplied .pkl, and
writes a JSON report capturing system info, torch/CUDA state, model
metadata, and per-iteration timing stats. Intended to compare performance
across different OSes and GPUs — collect the JSON files from each machine
and diff them.

Must be invoked from the repo root so that `import dnnlib` and
`import legacy` resolve.

Examples:

\b
# Quick run against a local file, defaults (30s timed, 5s warmup, batch=1)
python benchmark.py --network=model.pkl

\b
# Longer run with larger batch, float16 autocast, custom output path
python benchmark.py --network=model.pkl --duration=60 --batch-size=8 \\
    --dtype=float16 --output=results/rtx4090.json

\b
# Force CPU (much slower), shorter run
python benchmark.py --network=model.pkl --device=cpu --duration=10 --warmup=2
"""

import json
import platform
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import click
import numpy as np
import psutil
import torch

import dnnlib
import legacy


SCHEMA_VERSION = 1


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise click.ClickException('--device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    if device_arg == 'mps':
        if not torch.backends.mps.is_available():
            raise click.ClickException('--device=mps requested but MPS is not available')
        return torch.device('mps')
    if device_arg == 'cpu':
        return torch.device('cpu')
    # auto — same cascade as gen_images.py:106-110
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _get_sync_fn(device: torch.device) -> Callable[[], None]:
    if device.type == 'cuda':
        return torch.cuda.synchronize
    if device.type == 'mps':
        return torch.mps.synchronize
    return lambda: None


def _get_cpu_name() -> Optional[str]:
    try:
        if sys.platform.startswith('linux'):
            cpuinfo = Path('/proc/cpuinfo').read_text()
            for line in cpuinfo.splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
        elif sys.platform == 'darwin':
            out = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif sys.platform.startswith('win'):
            proc = platform.processor()
            if proc:
                return proc
    except Exception:
        pass
    # Fallback
    return platform.processor() or platform.machine() or None


def _get_driver_version() -> Optional[str]:
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ''
            return first or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    except Exception:
        return None
    return None


def _get_git_info() -> dict:
    info = {'git_commit': None, 'git_dirty': None}
    try:
        repo_root = Path(__file__).resolve().parent
        commit = subprocess.run(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if commit.returncode == 0:
            info['git_commit'] = commit.stdout.strip() or None
        status = subprocess.run(
            ['git', '-C', str(repo_root), 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if status.returncode == 0:
            info['git_dirty'] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def _gpu_slug(device: torch.device) -> str:
    if device.type != 'cuda':
        return device.type  # 'cpu' or 'mps'
    name = torch.cuda.get_device_name(device)
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    return slug[:40] or 'cuda'


def _dtype_from_str(s: str) -> torch.dtype:
    return {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[s]


def _percentile(values, q: float) -> float:
    return float(np.percentile(values, q, method='linear'))


def _compute_stats(per_iter_ms, total_wall: float, batch_size: int,
                   img_resolution: int) -> dict:
    arr = np.asarray(per_iter_ms, dtype=np.float64)
    count = int(arr.size)
    total_images = count * batch_size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if count > 1 else 0.0
    stats = {
        'total_iterations': count,
        'total_wall_seconds': float(total_wall),
        'per_iteration_ms': {
            'mean': mean,
            'median': float(np.median(arr)),
            'std': std,
            'min': float(arr.min()),
            'max': float(arr.max()),
            'p50': _percentile(arr, 50),
            'p90': _percentile(arr, 90),
            'p95': _percentile(arr, 95),
            'p99': _percentile(arr, 99),
        },
        'throughput': {
            'images_per_sec': total_images / total_wall if total_wall > 0 else 0.0,
            'megapixels_per_sec': (total_images * img_resolution * img_resolution)
                                  / 1e6 / total_wall if total_wall > 0 else 0.0,
        },
        'stability_cov': (std / mean) if mean > 0 else 0.0,
    }
    return stats


def _build_report(*, config: dict, device: torch.device, G, per_iter_ms,
                  total_wall: float, warmup_iters: int, interrupted: bool,
                  cudnn_benchmark: bool, allow_tf32: bool) -> dict:
    now = datetime.now(timezone.utc)

    gpu_block = None
    gpu_memory = None
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        cap = torch.cuda.get_device_capability(device)
        gpu_block = {
            'name': torch.cuda.get_device_name(device),
            'capability': f'{cap[0]}.{cap[1]}',
            'total_memory_mb': props.total_memory / 1024**2,
            'multi_processor_count': props.multi_processor_count,
            'driver_version': _get_driver_version(),
        }
        gpu_memory = {
            'peak_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
            'peak_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2,
        }

    results = _compute_stats(per_iter_ms, total_wall, config['batch_size'],
                             G.img_resolution) if per_iter_ms else {
        'total_iterations': 0,
        'total_wall_seconds': float(total_wall),
        'per_iteration_ms': None,
        'throughput': None,
        'stability_cov': None,
    }
    results['warmup_iterations'] = warmup_iters
    results['gpu_memory'] = gpu_memory

    report = {
        'schema_version': SCHEMA_VERSION,
        'timestamp_utc': now.isoformat().replace('+00:00', 'Z'),
        'interrupted': interrupted,
        'benchmark_tool': {'script': 'benchmark.py', **_get_git_info()},
        'system': {
            'os': {
                'name': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
            },
            'python_version': platform.python_version(),
            'cpu': {
                'name': _get_cpu_name(),
                'count_logical': psutil.cpu_count(logical=True),
                'count_physical': psutil.cpu_count(logical=False),
            },
            'ram_total_gb': psutil.virtual_memory().total / 1024**3,
            'hostname': platform.node(),
        },
        'torch': {
            'version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'cudnn_benchmark': cudnn_benchmark,
            'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
        },
        'gpu': gpu_block,
        'config': config,
        'model': {
            'img_resolution': int(G.img_resolution),
            'img_channels': int(G.img_channels),
            'z_dim': int(G.z_dim),
            'c_dim': int(G.c_dim),
            'num_parameters': int(sum(p.numel() for p in G.parameters())),
            'class_name': type(G).__name__,
            'has_synthesis_input': hasattr(G.synthesis, 'input'),
            'is_conditional': int(G.c_dim) > 0,
        },
        'results': results,
    }
    return report


def _print_summary(report: dict) -> None:
    r = report['results']
    cfg = report['config']
    gpu_name = report['gpu']['name'] if report['gpu'] else cfg['device_resolved']
    peak_mb = (r['gpu_memory']['peak_allocated_mb']
               if r.get('gpu_memory') else None)
    peak_str = f'{peak_mb:.0f} MB' if peak_mb is not None else 'n/a'

    print()
    print('=' * 60)
    print(f'Device        : {gpu_name}')
    print(f'Dtype         : {cfg["dtype"]}   Batch: {cfg["batch_size"]}   '
          f'Resolution: {report["model"]["img_resolution"]}')
    if r['per_iteration_ms']:
        p = r['per_iteration_ms']
        print(f'Iterations    : {r["total_iterations"]} in {r["total_wall_seconds"]:.2f}s '
              f'(warmup {r["warmup_iterations"]})')
        print(f'Per-iter (ms) : mean {p["mean"]:.2f} ± {p["std"]:.2f}   '
              f'p50 {p["p50"]:.2f}   p95 {p["p95"]:.2f}   p99 {p["p99"]:.2f}')
        print(f'Throughput    : {r["throughput"]["images_per_sec"]:.2f} images/sec   '
              f'({r["throughput"]["megapixels_per_sec"]:.2f} MP/sec)')
        print(f'Peak GPU mem  : {peak_str}')
        print(f'Stability CoV : {r["stability_cov"]:.3f}')
    else:
        print('No timed iterations completed.')
    if report['interrupted']:
        print('NOTE: run was interrupted; results are partial.')
    print('=' * 60)


@click.command()
@click.option('--network', 'network_pkl', required=True,
              help='Path or URL to .pkl generator network')
@click.option('--duration', type=float, default=30.0, show_default=True,
              help='Minimum wall-time of the timed loop (seconds)')
@click.option('--warmup', type=float, default=5.0, show_default=True,
              help='Warmup duration (seconds); always runs at least 1 iteration')
@click.option('--batch-size', type=int, default=1, show_default=True,
              help='Batch size for each forward pass')
@click.option('--truncation-psi', type=float, default=1.0, show_default=True,
              help='Truncation psi forwarded to the generator')
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']),
              default='const', show_default=True)
@click.option('--device', 'device_arg', type=click.Choice(['auto', 'cuda', 'cpu', 'mps']),
              default='auto', show_default=True)
@click.option('--output', 'output_path', type=str, default=None,
              help='Output JSON report path. Default: benchmark_{gpu_slug}_{utc_ts}.json in cwd')
@click.option('--seed', type=int, default=0, show_default=True,
              help='Seed for latent sampling (does not affect throughput)')
@click.option('--cudnn-benchmark/--no-cudnn-benchmark', default=True, show_default=True,
              help='Set torch.backends.cudnn.benchmark (safe: input shape is constant)')
@click.option('--allow-tf32/--no-allow-tf32', default=True, show_default=True,
              help='Enable TF32 for matmul and cuDNN (reflects realistic inference perf)')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'bfloat16']),
              default='float32', show_default=True,
              help='Non-fp32 wraps the forward pass in torch.autocast')
def run_benchmark(
    network_pkl: str,
    duration: float,
    warmup: float,
    batch_size: int,
    truncation_psi: float,
    noise_mode: str,
    device_arg: str,
    output_path: Optional[str],
    seed: int,
    cudnn_benchmark: bool,
    allow_tf32: bool,
    dtype: str,
):
    """Benchmark raw generator inference speed for a StyleGAN .pkl."""

    device = _resolve_device(device_arg)

    # Apply global torch flags BEFORE loading the model so any cudnn/tf32
    # autotuning during the first forward respects them.
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.manual_seed(seed)

    print(f'Loading network from "{network_pkl}" ...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    print(f'Network loaded: {type(G).__name__}  '
          f'{G.img_resolution}x{G.img_resolution}  z_dim={G.z_dim}  c_dim={G.c_dim}')
    print(f'Device resolved: {device}')

    # Pre-allocate reused inputs (NOT part of the timed region)
    rng = np.random.RandomState(seed)
    z_np = rng.randn(batch_size, G.z_dim)
    if device.type == 'mps':
        z_np = z_np.astype(np.float32)  # gen_images.py:130-131
    z = torch.from_numpy(z_np).to(device)
    c = torch.zeros([batch_size, G.c_dim], device=device)
    if G.c_dim > 0:
        c[:, 0] = 1
        print(f'Conditional network detected (c_dim={G.c_dim}); using class 0 for timing.')

    sync = _get_sync_fn(device)
    if dtype == 'float32':
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device.type,
                                      dtype=_dtype_from_str(dtype))

    # Local resolution of the network file size (for local paths only)
    network_size = None
    try:
        p = Path(network_pkl)
        if p.is_file():
            network_size = p.stat().st_size
    except Exception:
        network_size = None

    config = {
        'network_pkl': network_pkl,
        'network_pkl_size_bytes': network_size,
        'duration_seconds': float(duration),
        'warmup_seconds': float(warmup),
        'batch_size': int(batch_size),
        'truncation_psi': float(truncation_psi),
        'noise_mode': noise_mode,
        'device_requested': device_arg,
        'device_resolved': str(device),
        'dtype': dtype,
        'seed': int(seed),
        'cudnn_benchmark': bool(cudnn_benchmark),
        'allow_tf32': bool(allow_tf32),
    }

    def forward():
        return G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)

    per_iter_ms: list[float] = []
    warmup_iters = 0
    total_wall = 0.0
    interrupted = False

    with torch.inference_mode(), autocast_ctx:
        # Warmup
        print(f'Warmup: running for {warmup:.1f}s (at least 1 iter) ...')
        sync()
        warmup_start = time.perf_counter()
        while True:
            forward()
            sync()
            warmup_iters += 1
            if (time.perf_counter() - warmup_start) >= warmup:
                break
        warmup_wall = time.perf_counter() - warmup_start
        print(f'Warmup complete: {warmup_iters} iter(s) in {warmup_wall:.2f}s')

        # Reset peak memory to capture steady-state peak only
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # Timed loop
        print(f'Benchmark: running for {duration:.1f}s ...')
        try:
            loop_start = time.perf_counter()
            while True:
                sync()
                t0 = time.perf_counter()
                forward()
                sync()
                t1 = time.perf_counter()
                per_iter_ms.append((t1 - t0) * 1000.0)
                if (t1 - loop_start) >= duration:
                    break
            total_wall = time.perf_counter() - loop_start
        except KeyboardInterrupt:
            total_wall = time.perf_counter() - loop_start
            interrupted = True
            print('\nInterrupted — writing partial results.')

    report = _build_report(
        config=config, device=device, G=G, per_iter_ms=per_iter_ms,
        total_wall=total_wall, warmup_iters=warmup_iters,
        interrupted=interrupted,
        cudnn_benchmark=cudnn_benchmark, allow_tf32=allow_tf32,
    )

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output_path = f'benchmark_{_gpu_slug(device)}_{ts}.json'
    out = Path(output_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    _print_summary(report)
    print(f'Report written to: {out}')

    if report['results'].get('total_iterations', 0) < 10 and not interrupted:
        print(f'WARNING: only {report["results"]["total_iterations"]} timed '
              f'iteration(s) completed — results may be noisy. '
              f'Consider increasing --duration.')


if __name__ == '__main__':
    run_benchmark()  # pylint: disable=no-value-for-parameter
