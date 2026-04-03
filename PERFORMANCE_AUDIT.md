# DSWX-SAR Performance Audit & Optimization Plan

**Date:** 2026-04-03
**Repo:** ~21k lines, 47 Python modules
**Pipeline:** mosaic -> filter -> threshold -> fuzzy -> region growing -> masking -> bimodality -> inundated veg -> MGRS tiles

---

## Critical Issues

### 1. Region Growing: Iterative Binary Dilation (biggest wall-clock bottleneck)
- **File:** `src/dswx_sar/common/_region_growing.py:94-116`
- **Problem:** `ndimage.binary_dilation()` called every iteration of a while-loop (can run 200+ times), each time over the full image. O(iterations x image_size).
- **Fix:** Replace iterative single-pixel dilation with distance-transform approach (`scipy.ndimage.distance_transform_edt`) or batched `iterations=N` dilation.
- **GPU alternative:** `cupyx.scipy.ndimage.binary_dilation` is a direct drop-in — 10-50x speedup per call.
- **Expected speedup:** 5-50x for this stage
- **Effort:** Medium

### 2. Pipeline is Fully Sequential
- **File:** `src/dswx_sar/dswx_s1.py`
- **Problem:** Each polarization set goes through entire pipeline serially. Stages are independent per pol set.
- **Fix:** `joblib.Parallel` or `concurrent.futures` across polarization sets.
- **Expected speedup:** ~2x for dual-pol runs
- **Effort:** Low

### 3. Mosaic Loads All Bursts Fully Into Memory
- **File:** `src/dswx_sar/common/_mosaic.py:366-480`
- **Problem:** Allocates `np.zeros((num_bands, height, width))` for full output + reads each burst entirely with `ReadAsArray()`.
- **Fix:** Process in row-strips/blocks; only load overlapping burst portions.
- **Expected speedup:** Memory reduction, avoids swapping on large mosaics
- **Effort:** High

### 4. `np.append()` in Loop — O(n^2) Memory Copies
- **File:** `src/dswx_sar/common/_initial_threshold.py:1966-1971`
- **Problem:** `np.append()` copies entire accumulated array each iteration.
- **Fix:** Accumulate in a Python list, `np.concatenate()` once at end.
- **Expected speedup:** 10-100x for that loop
- **Effort:** Low

### 5. GDAL Open/Close Per Block in Nested Loop
- **File:** `src/dswx_sar/common/_initial_threshold.py:1854-1866`
- **Problem:** Two `gdal.Open()` + close cycles per block in (n_rows x n_cols) loop.
- **Fix:** Open each file once before the loop, read windowed blocks from persistent handle.
- **Expected speedup:** 2-5x for threshold stage
- **Effort:** Low

### 6. Sliding Window Triple-Nested Loop with List Appends
- **File:** `src/dswx_sar/common/_initial_threshold.py:367-499`
- **Problem:** `while` > `for` > `for` with dynamic list `.append()` calls inside inner loop.
- **Fix:** Pre-allocate result arrays. Consider `np.lib.stride_tricks.sliding_window_view`.
- **Expected speedup:** 2-5x
- **Effort:** Medium

---

## Medium Priority Issues

### 7. Unnecessary `.copy()` Calls
- **File:** `src/dswx_sar/common/_masking_with_ancillary.py` (7+ locations)
- **Fix:** Audit each copy; remove where mutation isn't a risk.

### 8. Duplicate Code: `check_water_land_mixture` vs `check_water_land_mixture_v2`
- **File:** `src/dswx_sar/common/_masking_with_ancillary.py`
- **Fix:** Consolidate into single parameterized function.

### 9. Pixel-Level Nested Loop in Comparison Tool
- **File:** `src/dswx_sar/dswx_comparison.py:51-65`
- **Fix:** Replace with `np.allclose()` or vectorized difference.

### 10. No Per-Stage Profiling
- Only coarse `time.time()` in a few places. No memory monitoring.
- **Fix:** Add `time.perf_counter()` around each pipeline stage; log to structured output.

---

## Revision Plan A: joblib Parallelization (CPU-only, no new dependencies)

### Current State

Already parallelized with joblib:
- `_initial_threshold.py` — block processing via `Parallel(n_jobs=n_jobs)(delayed(process_block)(...))`
- `_region_growing.py` — block processing via `Parallel(n_jobs=use_cpu)(delayed(process_region_growing_block)(...))`
- `_masking_with_ancillary.py` — 3 parallel sections (check_water_land_mixture_v2, check_water_land_mixture, compute_spatial_coverage)
- `_refine_with_bimodality.py` — 2 parallel sections (process_dark_land_component, process_bright_water_component)
- `nisar/pre_processing.py` — block filtering via `Parallel(n_jobs=use_cpu)(delayed(_filter_one_block)(...))`

NOT parallelized yet (all have independent block loops, verified safe):
- `_fuzzy_value_computation.py` — block loop at lines 535-627
- `sentinel1/pre_processing.py` — block loop at lines 201-281
- `_detect_inundated_vegetation.py` — block loop at lines 151-264
- `dswx_s1.py` / `dswx_ni.py` — pol set loop at line 56/65

### A1. Add joblib to `_fuzzy_value_computation.py`
- **File:** `src/dswx_sar/common/_fuzzy_value_computation.py:535-627`
- **Current:** Sequential `for block_param in block_params:` loop computing fuzzy values per block
- **Change:** Extract block body into standalone function, wrap with `Parallel(n_jobs=-1)(delayed(process_fuzzy_block)(...))`
- **Pattern:** Parallel compute, sequential GDAL write (accumulate results in memory, write after)
- **Safety:** Blocks are independent — each reads separate regions from intensity, HAND, slope, landcover files and produces independent output regions
- **Expected speedup:** 2-4x
- **Effort:** Low

### A2. Add joblib to `sentinel1/pre_processing.py`
- **File:** `src/dswx_sar/sentinel1/pre_processing.py:201-281`
- **Current:** Sequential `for block_ind, block_param in enumerate(block_params):` with inner pol loop
- **Change:** Extract block body into standalone function, wrap with `Parallel(n_jobs=-1)(delayed(process_filter_block)(...))`
- **Note:** `nisar/pre_processing.py` already does this — mirror that pattern
- **Safety:** Each block reads/filters independent regions per polarization
- **Expected speedup:** 2-3x
- **Effort:** Low (copy pattern from nisar/pre_processing.py)

### A3. Add joblib to `_detect_inundated_vegetation.py`
- **File:** `src/dswx_sar/common/_detect_inundated_vegetation.py:151-264`
- **Current:** Sequential `for block_param in block_params:` loop
- **Change:** Extract block body into standalone function, wrap with `Parallel(n_jobs=-1)(delayed(process_inundated_veg_block)(...))`
- **Safety:** Blocks are independent — each reads SAR ratio, applies filtering, produces independent output
- **Expected speedup:** 2-4x
- **Effort:** Low

### A4. Parallelize polarization sets in top-level workflow
- **Files:** `src/dswx_sar/dswx_s1.py:56-82`, `src/dswx_sar/dswx_ni.py:65-91`
- **Current:** `for pol_set in proc_pol_set:` runs entire pipeline per pol set sequentially
- **Change:**
  ```python
  import copy
  from joblib import Parallel, delayed

  def _run_pol_set(cfg, pol_set, dswx_workflow, inundated_veg_cfg):
      cfg_copy = copy.deepcopy(cfg)
      cfg_copy.groups.processing.polarizations = pol_set
      pre_processing.run(cfg_copy)
      initial_threshold.run(cfg_copy)
      fuzzy_value_computation.run(cfg_copy)
      region_growing.run(cfg_copy)
      if dswx_workflow == 'opera_dswx_s1':
          masking_with_ancillary.run(cfg_copy)
          refine_with_bimodality.run(cfg_copy)
          if ((inundated_veg_cfg.enabled == 'auto') and len(pol_set) >= 2) \
             or inundated_veg_cfg.enabled is True:
              detect_inundated_vegetation.run(cfg_copy)

  Parallel(n_jobs=len(proc_pol_set))(
      delayed(_run_pol_set)(cfg, pol_set, dswx_workflow, inundated_veg_cfg)
      for pol_set in proc_pol_set
  )
  ```
- **Critical:** Must `deepcopy(cfg)` per worker — the current code mutates `processing_cfg.polarizations` on a shared config object
- **Safety:** Pol sets write to different output files (filenames include pol string). No cross-pol data dependencies until final MGRS tile merge.
- **Expected speedup:** ~2x for dual-pol, ~4x for quad-pol (NISAR)
- **Effort:** Low

### A5. GDAL Write Safety Pattern for All Parallel Block Loops
- **Problem:** `write_raster_block()` uses `gdal.Open(..., GA_Update)` per block — not safe for concurrent writes to the same file from multiple processes
- **Solution:** Adopt compute-then-write pattern:
  ```python
  # Parallel compute phase
  results = Parallel(n_jobs=-1)(
      delayed(compute_block)(block_param, ...) for block_param in block_params
  )
  # Sequential write phase
  for block_param, result_data in zip(block_params, results):
      write_raster_block(output_path, result_data, block_param, ...)
  ```
- **Applies to:** A1, A2, A3 above
- **Alternative:** Use `joblib` with `backend='loky'` + file locking, but compute-then-write is simpler and avoids GDAL thread-safety issues entirely

---

## Revision Plan B: CuPy/cucim GPU Acceleration

### Prerequisites
- NVIDIA GPU with CUDA toolkit installed
- `pip install cupy-cuda12x cucim-cuda12x` (match CUDA version)
- Add to `docker/requirements.txt`: `cupy-cuda12x`, `cucim-cu12`

### Backend Abstraction Layer (implement first)

Create `src/dswx_sar/common/_backend.py` to allow CPU/GPU switching without touching every file:

```python
"""Array backend abstraction — transparently use numpy or cupy."""
import os

_USE_GPU = os.environ.get('DSWX_USE_GPU', '0') == '1'

try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cp_ndimage
    import cupyx.scipy.signal as _cp_signal
    from cucim.skimage.filters import threshold_otsu as _gpu_otsu
    from cucim.skimage.filters import threshold_multiotsu as _gpu_multiotsu
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

import numpy as _np
from scipy import ndimage as _cpu_ndimage
from scipy import signal as _cpu_signal
from skimage.filters import threshold_otsu as _cpu_otsu
from skimage.filters import threshold_multiotsu as _cpu_multiotsu

def use_gpu():
    return _USE_GPU and _GPU_AVAILABLE

# Array namespace — drop-in for `import numpy as np`
xp = _cp if use_gpu() else _np

# scipy.ndimage namespace
xndimage = _cp_ndimage if use_gpu() else _cpu_ndimage

# scipy.signal namespace
xsignal = _cp_signal if use_gpu() else _cpu_signal

# skimage thresholding
x_threshold_otsu = _gpu_otsu if use_gpu() else _cpu_otsu
x_threshold_multiotsu = _gpu_multiotsu if use_gpu() else _cpu_multiotsu

def to_device(arr):
    """Move numpy array to GPU if GPU mode enabled."""
    if use_gpu():
        return _cp.asarray(arr)
    return arr

def to_host(arr):
    """Move array to CPU (numpy). No-op if already numpy."""
    if use_gpu() and hasattr(arr, 'get'):
        return arr.get()
    return arr
```

**Usage in existing code (minimal changes per file):**
```python
# Before:
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

# After:
from dswx_sar.common._backend import xp as np, xndimage as ndimage, x_threshold_otsu as threshold_otsu, to_device, to_host
```

### B1. GPU-accelerate region growing (highest impact)
- **File:** `src/dswx_sar/common/_region_growing.py:94-116`
- **Change:** Use `_backend` imports; wrap GDAL reads with `to_device()`, GDAL writes with `to_host()`
- **Key operation:** `ndimage.binary_dilation()` called 200+ times per tile → `cupyx.scipy.ndimage.binary_dilation()`
- **Transfer pattern:** `to_device()` once after GDAL read, iterate on GPU, `to_host()` once before GDAL write
- **Expected speedup:** 10-50x for this stage (morphology is massively parallel on GPU)

### B2. GPU-accelerate SAR filtering
- **File:** `src/dswx_sar/common/_filter_SAR.py`
- **Key operations:**
  - `uniform_filter()` → `cupyx.scipy.ndimage.uniform_filter()` (Lee filter windowed stats)
  - `signal.convolve()` → `cupyx.scipy.signal.convolve()` (masked convolution)
  - `denoise_tv_chambolle/bregman` → `cucim.skimage.restoration`
- **Transfer pattern:** `to_device()` per block after read, `to_host()` before write
- **Expected speedup:** 10-50x for filtering stage

### B3. GPU-accelerate masking morphology
- **File:** `src/dswx_sar/common/_masking_with_ancillary.py`
- **Key operations:** 9x `binary_dilation`, 2x `binary_erosion`, `threshold_multiotsu`
- **Change:** Use `_backend` imports
- **Expected speedup:** 5-20x for masking stage

### B4. GPU-accelerate thresholding
- **Files:** `_initial_threshold.py`, `_refine_with_bimodality.py`
- **Key operations:** `threshold_otsu` (7 calls), `threshold_multiotsu` (4 calls), `binary_dilation` (3 calls)
- **Note:** `curve_fit` and `griddata` MUST stay on CPU — add `to_host()` before these calls, `to_device()` after
- **Expected speedup:** 5-20x for threshold/bimodality stages

### B5. GPU-accelerate fuzzy value computation
- **File:** `src/dswx_sar/common/_fuzzy_value_computation.py`
- **Key operations:** `cv2.Sobel` → GPU Sobel, bulk numpy array math
- **Expected speedup:** 3-10x

### GPU Memory Management
- SAR images can be 5000x5000+ pixels. At float32, one band = ~100 MB.
- GPU VRAM is typically 8-24 GB. Block-based processing fits easily.
- The existing `block_param_generator()` already chunks data — just add `to_device()`/`to_host()` at block boundaries.
- For the mosaic stage (full-image arrays), skip GPU — it's I/O-bound anyway.

### CPU-Only Operations (no changes needed)
These stay on CPU with no backend swap:
- `scipy.optimize.curve_fit` — Gaussian mixture fitting
- `scipy.interpolate.griddata` / `Rbf` — scattered interpolation
- `scipy.stats.norm.cdf`, `skew`, `kurtosis` — statistical functions
- `scipy.spatial.ConvexHull` — computational geometry
- All GDAL I/O — disk-bound

---

## Dask Status

**Dask is NOT used in any processing code.** It only appears in `notebook/instruction.ipynb`
for demo purposes. No action needed.

---

## Revised Execution Order

### Phase 1: Quick CPU Wins (no new dependencies, low risk)
| Step | Change | Speedup | Effort | Risk |
|------|--------|---------|--------|------|
| 1 | Fix `np.append` in loop (issue #4) | 10-100x local | Low | None |
| 2 | Open GDAL files once (issue #5) | 2-5x local | Low | None |
| 3 | Add per-stage profiling (issue #10) | Diagnostic | Low | None |

### Phase 2: joblib Everywhere (Plan A)
| Step | Change | Speedup | Effort | Risk |
|------|--------|---------|--------|------|
| 4 | A1: joblib in `_fuzzy_value_computation.py` | 2-4x local | Low | Low |
| 5 | A2: joblib in `sentinel1/pre_processing.py` | 2-3x local | Low | Low |
| 6 | A3: joblib in `_detect_inundated_vegetation.py` | 2-4x local | Low | Low |
| 7 | A4: Parallelize pol sets in `dswx_s1.py`/`dswx_ni.py` | ~2x overall | Low | Low |

### Phase 3: CuPy/GPU Acceleration (Plan B)
| Step | Change | Speedup | Effort | Risk |
|------|--------|---------|--------|------|
| 8 | Create `_backend.py` abstraction layer | Foundation | Low | None |
| 9 | B1: GPU region growing (binary_dilation loop) | 10-50x local | Medium | Medium |
| 10 | B2: GPU SAR filtering (uniform_filter, convolve) | 10-50x local | Medium | Medium |
| 11 | B3: GPU masking morphology | 5-20x local | Medium | Low |
| 12 | B4: GPU thresholding (otsu, multiotsu) | 5-20x local | Medium | Low |
| 13 | B5: GPU fuzzy value computation | 3-10x local | Medium | Low |

### Phase 4: Structural Improvements
| Step | Change | Speedup | Effort | Risk |
|------|--------|---------|--------|------|
| 14 | Rewrite region growing algorithm (distance transform) | 5-50x local | Medium | Medium |
| 15 | Block-based mosaic (issue #3) | Memory savings | High | Medium |
| 16 | Remove unnecessary `.copy()` calls | Minor | Low | Low |
| 17 | Consolidate duplicate `check_water_land_mixture` functions | Maintenance | Low | Low |
