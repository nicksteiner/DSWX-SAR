# DSWX-SAR
Dynamic Surface Water Extent from Synthetic Aperture Radar

## Installation

### Download the Source Code
Download the source code and change working directory to cloned repository:

```bash
git clone https://github.com/opera-adt/DSWX-SAR.git
```

### Standard Installation
Install dependencies (installation via conda is recommended):
```bash
conda install --file docker/requirements.txt
conda install -c conda-forge --file docker/requirements_forge.txt
```

Install via setup.py:

```bash
python setup.py install
python setup.py clean
```

Note: Installation via pip is not currently recommended due to an
issue with the osgeo and gdal dependency.


OR update environment path to run DSWX-SAR:

```bash
export DSWX_SAR_HOME=$PWD
export PYTHONPATH=${PYTHONPATH}:${DSWX_SAR_HOME}/src
export PATH=${PATH}:${DSWX_SAR_HOME}/bin
```

Process data sets; use a runconfig file to specify the location
of the dataset, the output directory, parameters, etc.

```bash
dswx_s1.py <path to runconfig file>
```
Note: Only Sentinel-1 data is currently supported. 

A default runconfig file can be found: `DSWX-SAR > src > dswx_sar > defaults > dswx_s1.yaml`.
This file can be copied and modified for your needs.
Note: The runconfig must meet this schema: `DSWX-SAR > src > dswx_sar > schemas > dswx_s1.yaml`.


### Turbo Branch (Performance-Optimized)

The `turbo` branch includes performance optimizations that can significantly reduce
processing time. These changes are maintained as a separate branch for benchmarking
and to stay cleanly rebasable onto JPL's upstream `main`.

```bash
git checkout turbo
```

**What's different on `turbo`:**

- **joblib parallelization** of all block-processing stages (fuzzy value computation,
  S1 pre-processing, inundated vegetation detection) and top-level polarization set
  processing. ~2-4x speedup on multi-core systems.
- **O(n^2) → O(n) fix** in threshold computation (`np.append` in loop replaced with
  list accumulation + single `np.concatenate`).
- **Per-stage profiling** — every pipeline stage logs wall-clock time with `[profile]`
  tags so you can see exactly where time goes.
- **Vectorized comparison tool** — pixel-level nested loop replaced with numpy ops.
- **GPU acceleration via CuPy** (optional) — region growing, SAR filtering, and
  morphological operations run on GPU when available.

**To enable GPU acceleration:**

```bash
# Install CuPy and cucim (match your CUDA version)
pip install cupy-cuda12x cucim-cu12

# Enable GPU backend
export DSWX_USE_GPU=1

# Run as normal
dswx_s1.py <path to runconfig file>
```

When `DSWX_USE_GPU` is unset or `0`, the code falls back to numpy/scipy with zero
overhead — no GPU hardware or packages required.

**Keeping turbo up to date with upstream:**

```bash
git checkout main
git pull origin main
git checkout turbo
git rebase main
```

See [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md) for the full audit and optimization details.


### Alternate Installation: Docker Image

Skip the standard installation process above.

Then, from inside the cloned repository, build the Docker image:
(This will automatically run the workflow tests.)

```bash
./build_docker_image.sh
```

Load the Docker container image onto your computer:

```bash
docker load -i docker/dockerimg_dswx_s1_final_patch_1.1.tar
```

See DSWx-SAR Science Algorithm Software (SAS) User Guide for instructions on processing via Docker.


### License
**Copyright (c) 2021** California Institute of Technology (“Caltech”). U.S. Government
sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the
names of its contributors may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

