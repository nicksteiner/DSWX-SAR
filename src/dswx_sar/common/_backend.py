"""Array backend abstraction for CPU/GPU transparent execution.

Set DSWX_USE_GPU=1 to enable GPU acceleration via CuPy/cucim.
Falls back to numpy/scipy/skimage when CuPy is not installed or
DSWX_USE_GPU is unset.

Usage in existing modules:
    from dswx_sar.common._backend import (
        xp, xndimage, xsignal,
        x_threshold_otsu, x_threshold_multiotsu,
        to_device, to_host)
"""
import os
import logging

logger = logging.getLogger('dswx_sar')

_USE_GPU = os.environ.get('DSWX_USE_GPU', '0') == '1'
_GPU_AVAILABLE = False

if _USE_GPU:
    try:
        import cupy as _cp
        import cupyx.scipy.ndimage as _cp_ndimage
        import cupyx.scipy.signal as _cp_signal
        from cucim.skimage.filters import threshold_otsu as _gpu_otsu
        from cucim.skimage.filters import threshold_multiotsu as _gpu_multiotsu
        _GPU_AVAILABLE = True
        logger.info('GPU backend enabled (CuPy + cucim)')
    except ImportError as e:
        logger.warning(f'DSWX_USE_GPU=1 but GPU packages unavailable: {e}. '
                       'Falling back to CPU.')

import numpy as _np
from scipy import ndimage as _cpu_ndimage
from scipy import signal as _cpu_signal
from skimage.filters import threshold_otsu as _cpu_otsu
from skimage.filters import threshold_multiotsu as _cpu_multiotsu


def use_gpu():
    """Return True if GPU execution is enabled and available."""
    return _USE_GPU and _GPU_AVAILABLE


# --- Array namespace (drop-in for numpy) ---
xp = _cp if use_gpu() else _np

# --- scipy.ndimage namespace ---
xndimage = _cp_ndimage if use_gpu() else _cpu_ndimage

# --- scipy.signal namespace ---
xsignal = _cp_signal if use_gpu() else _cpu_signal

# --- skimage thresholding ---
x_threshold_otsu = _gpu_otsu if use_gpu() else _cpu_otsu
x_threshold_multiotsu = _gpu_multiotsu if use_gpu() else _cpu_multiotsu


def to_device(arr):
    """Move numpy array to GPU if GPU mode is enabled.

    No-op when running on CPU.
    """
    if use_gpu():
        return _cp.asarray(arr)
    return arr


def to_host(arr):
    """Move array back to CPU (numpy). No-op if already numpy."""
    if use_gpu() and hasattr(arr, 'get'):
        return arr.get()
    return arr
