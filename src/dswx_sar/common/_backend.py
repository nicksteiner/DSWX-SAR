"""Array backend abstraction for CPU/GPU transparent execution.

GPU acceleration is **auto-detected**: if CuPy is installed and a CUDA
GPU is available, it is used automatically. No configuration required.

Override with environment variables:
    DSWX_USE_GPU=0   Force CPU even when GPU is available
    DSWX_USE_GPU=1   Force GPU (fail loudly if unavailable)

When neither is set, auto-detection decides.

Usage in existing modules:
    from dswx_sar.common._backend import (
        xp, xndimage, xsignal,
        x_threshold_otsu, x_threshold_multiotsu,
        to_device, to_host)
"""
import os
import logging

logger = logging.getLogger('dswx_sar')

_GPU_AVAILABLE = False
_ENV_OVERRIDE = os.environ.get('DSWX_USE_GPU', None)

# Auto-detect: try importing CuPy and probing for a CUDA device
try:
    import cupy as _cp
    _cp.cuda.runtime.getDeviceCount()  # raises if no GPU
    import cupyx.scipy.ndimage as _cp_ndimage
    import cupyx.scipy.signal as _cp_signal
    from cucim.skimage.filters import threshold_otsu as _gpu_otsu
    from cucim.skimage.filters import threshold_multiotsu as _gpu_multiotsu
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

# Resolve final decision
if _ENV_OVERRIDE == '0':
    _USE_GPU = False
    if _GPU_AVAILABLE:
        logger.info('GPU available but disabled via DSWX_USE_GPU=0')
elif _ENV_OVERRIDE == '1':
    _USE_GPU = True
    if not _GPU_AVAILABLE:
        raise RuntimeError(
            'DSWX_USE_GPU=1 but CuPy/cucim not installed or no CUDA GPU '
            'found. Install with: pip install cupy-cuda12x cucim-cu12')
else:
    # Auto-detect
    _USE_GPU = _GPU_AVAILABLE
    if _USE_GPU:
        logger.info('GPU auto-detected — using CuPy/cucim backend')

import numpy as _np
from scipy import ndimage as _cpu_ndimage
from scipy import signal as _cpu_signal
from skimage.filters import threshold_otsu as _cpu_otsu
from skimage.filters import threshold_multiotsu as _cpu_multiotsu


def use_gpu():
    """Return True if GPU execution is active."""
    return _USE_GPU


# --- Array namespace (drop-in for numpy) ---
xp = _cp if _USE_GPU else _np

# --- scipy.ndimage namespace ---
xndimage = _cp_ndimage if _USE_GPU else _cpu_ndimage

# --- scipy.signal namespace ---
xsignal = _cp_signal if _USE_GPU else _cpu_signal

# --- skimage thresholding ---
x_threshold_otsu = _gpu_otsu if _USE_GPU else _cpu_otsu
x_threshold_multiotsu = _gpu_multiotsu if _USE_GPU else _cpu_multiotsu


def to_device(arr):
    """Move numpy array to GPU if GPU mode is active.

    No-op when running on CPU.
    """
    if _USE_GPU:
        return _cp.asarray(arr)
    return arr


def to_host(arr):
    """Move array back to CPU (numpy). No-op if already numpy."""
    if _USE_GPU and hasattr(arr, 'get'):
        return arr.get()
    return arr
