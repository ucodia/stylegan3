# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import hashlib
import importlib
import importlib.util
import os
import shutil
import uuid

import torch
import torch.utils.cpp_extension

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_cuda_compute_version():
    if not torch.cuda.is_available():
        return None
    return ".".join(map(str, torch.cuda.get_device_capability(torch.cuda.current_device())))

def _get_cache_identifier():
    arch_list_env = os.environ.get('TORCH_CUDA_ARCH_LIST', '').strip()
    
    if arch_list_env:
        arch_versions = [arch.strip() for arch in arch_list_env.split(';') if arch.strip()]
        if arch_versions:
            sorted_archs = sorted(arch_versions, key=lambda x: tuple(map(int, x.split('.'))))
            return '_'.join(sorted_archs)
    
    return _get_cuda_compute_version()

def _find_compatible_cache_dir(build_top_dir, source_digest):
    current_compute = _get_cuda_compute_version()
    
    if current_compute is None:
        return None
    
    cache_pattern = os.path.join(build_top_dir, f'{source_digest}-*')
    existing_caches = glob.glob(cache_pattern)
    
    if not existing_caches:
        return None
    
    for cache_dir in existing_caches:
        cache_suffix = os.path.basename(cache_dir).split('-', 1)[1]      
        arch_list = cache_suffix.split('_')
        if current_compute in arch_list:
            return cache_dir
    
    return None  # No compatible cache found

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    # Compile and load.
    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Some containers set TORCH_CUDA_ARCH_LIST to a list that can either
        # break the build or unnecessarily restrict what's available to nvcc.
        # Unset it to let nvcc decide based on what's available on the
        # machine.
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        #
        # EDIT: We now do it regardless of TORCH_EXTENSIOS_DIR, in order to work
        # around the *.cu dependency bug in ninja config.
        #
        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1: # and ('TORCH_EXTENSIONS_DIR' in os.environ):

            # Compute combined hash digest for all source files.
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            # Select cached build directory name.
            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            
            # First, try to find a compatible existing cache
            cached_build_dir = _find_compatible_cache_dir(build_top_dir, source_digest)
            
            if cached_build_dir is None:
                # No compatible cache found, create new one with current identifier
                cache_identifier = _get_cache_identifier()
                cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{cache_identifier}')

            # Check if we already have a compiled module in the cache
            if os.path.isdir(cached_build_dir):
                for ext in ['.so', '.pyd']:
                    compiled_file = os.path.join(cached_build_dir, f'{module_name}{ext}')
                    if os.path.isfile(compiled_file):
                        try:
                            spec = importlib.util.spec_from_file_location(module_name, compiled_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            print(f'Loaded cached PyTorch plugin "{module_name}" from {compiled_file}.')
                            _cached_plugins[module_name] = module
                            return module
                        except (ImportError, Exception) as e:
                            print(f'Cached module "{module_name}" failed to load ({e}), rebuilding...')
                            break
            else:
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir) # atomic
                except OSError:
                    # source directory already exists, delete tmpdir and its contents.
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir): raise

            # Compile.
            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs)
        else:
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)

        # Load.
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache dict.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
