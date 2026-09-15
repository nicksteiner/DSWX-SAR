#!/usr/bin/env python
"""
D832 Benchmark: Accuracy trade for DSWx-NI maximum_tile_size

Strategy: Run mosaic + preproc once. For each tile size, run the
tile-dependent steps (initial_threshold → downstream) in-place,
saving key outputs before wiping for the next tile size.

Tile sizes tested: 200, 400, 800
"""

import copy
import json
import logging
import os
import shutil
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

DEV_SRC = str(Path(__file__).resolve().parents[1] / "src")
if DEV_SRC not in sys.path:
    sys.path.insert(0, DEV_SRC)

from dswx_sar.nisar.dswx_ni_runconfig import (
    RunConfig,
    _get_parser,
    DSWX_NI_POL_DICT,
)
from dswx_sar.common import _initial_threshold as initial_threshold
from dswx_sar.common import _fuzzy_value_computation as fuzzy_value_computation
from dswx_sar.nisar import (
    mosaic_gcov_frame,
    pre_processing,
    refine_with_bimodality,
    region_growing,
    masking_with_ancillary,
)

logger = logging.getLogger("dswx_sar")

TILE_SIZES = [200, 400, 800]

FRAMES = {
    "frame_069": {
        "gcov_h5": "/media/nsteiner/data3/nisar/oasis/"
                   "NISAR_L2_PR_GCOV_006_069_D_068_4005_DHDH_M_"
                   "20251009T235223_20251009T235259_P00410_N_F_J_001.h5",
        "dem": "/media/nsteiner/data3/nisar/oasis/out/"
               "NISAR_L2_PR_GCOV_006_069_D_068_4005_DHDH_M_"
               "20251009T235223_20251009T235259_P00410_N_F_J_001/dem.tif",
        "hand": "/media/nsteiner/data3/nisar/oasis/out/"
                "NISAR_L2_PR_GCOV_006_069_D_068_4005_DHDH_M_"
                "20251009T235223_20251009T235259_P00410_N_F_J_001/hand.tif",
        "landcover": "/media/nsteiner/data3/nisar/oasis/out/"
                     "NISAR_L2_PR_GCOV_006_069_D_068_4005_DHDH_M_"
                     "20251009T235223_20251009T235259_P00410_N_F_J_001/"
                     "landcover.tif",
        "watercover": "/media/nsteiner/data3/nisar/oasis/out/"
                      "NISAR_L2_PR_GCOV_006_069_D_068_4005_DHDH_M_"
                      "20251009T235223_20251009T235259_P00410_N_F_J_001/"
                      "watercover.tif",
    },
    "frame_081": {
        "gcov_h5": "/media/nsteiner/data3/nisar/oasis/"
                   "NISAR_L2_PR_GCOV_006_081_D_009_4005_DHDH_A_"
                   "20251003T231520_20251003T231554_S00408_P_P_J_001.h5",
        "dem": "/media/nsteiner/data3/nisar/oasis/out/"
               "NISAR_L2_PR_GCOV_006_081_D_009_4005_DHDH_A_"
               "20251003T231520_20251003T231554_S00408_P_P_J_001/dem.tif",
        "hand": "/media/nsteiner/data3/nisar/oasis/out/"
                "NISAR_L2_PR_GCOV_006_081_D_009_4005_DHDH_A_"
                "20251003T231520_20251003T231554_S00408_P_P_J_001/hand.tif",
        "landcover": "/media/nsteiner/data3/nisar/oasis/out/"
                     "NISAR_L2_PR_GCOV_006_081_D_009_4005_DHDH_A_"
                     "20251003T231520_20251003T231554_S00408_P_P_J_001/"
                     "landcover.tif",
        "watercover": "/media/nsteiner/data3/nisar/oasis/out/"
                      "NISAR_L2_PR_GCOV_006_081_D_009_4005_DHDH_A_"
                      "20251003T231520_20251003T231554_S00408_P_P_J_001/"
                      "watercover.tif",
    },
}

BENCHMARK_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_ROOT / "results"

# Files to preserve between tile-size runs (pre-processing outputs)
PREPROC_FILES = [
    "mosaic_*.tif", "filtered_image_*.tif", "no_data_area_*.tif",
    "interpolated_*.tif", "temp_ref_water.tif", "intensity_*_db.tif",
    "slope.tif", "NISAR_*.tif",
]

# Files produced by tile-dependent steps (removed between runs)
TILE_DEPENDENT_GLOBS = [
    "intensity_threshold_filled_*.tif", "mode_tau_filled_*.tif",
    "water_mask_*.tif", "fuzzy_image_*.tif", "fuzzy_*.tif",
    "region_growing_*.tif", "temp_region_growing_*.tif",
    "initial_mask_*.tif", "mask_excluded_*.tif",
    "darkland_*.tif", "initial_darkland_*.tif",
    "split_mask_*.tif", "intermediate_*.tif",
    "false_positive_*.tif", "refine_landcover_*.tif",
    "masking_ancillary_*.tif", "input_image_test.tif",
    "remove_false_water_*.tif", "false_water_label_*.tif",
    "check_remove_false_water_*.tif", "merged_removed_false_water_*.tif",
    "fill_gap_water_*.tif", "water_label_bright_water_*.tif",
    "check_fill_gap_*.tif", "merged_fill_gap_*.tif",
    "initial_water_*.tif", "int_threshold_visualization_*",
    "data_thres_*.csv", "data_thres_*.vrt",
    "land_portion_*.tif", "binary_*.tif",
]


def write_algo_yaml(path, tile_size):
    """Write algorithm parameter YAML with given tile size."""
    content = f"""runconfig:
    name: dswx_ni_workflow_algorithm

    processing:
        dswx_workflow: 'opera_dswx_ni'
        polarizations: ['auto']
        polarimetric_option:

        reference_water:
            max_value: 100
            no_data_value: 255
            permanent_water_value: 0.9
            drought_erosion_pixel: 10
            flood_dilation_pixel: 16

        partial_surface_water:
            enabled: True
            threshold: 5

        hand:
            mask_value: 200

        ocean_mask:
            mask_enabled: False
            mask_margin_km: 5
            mask_polygon_water: True

        mosaic:
            mosaic_prefix: 'mosaic'
            mosaic_cog_enable: True
            mosaic_mode: 'first'
            mosaic_posting_thresh: 40
            resamp_required: True
            resamp_method: 'nearest'
            resamp_out_res: 30
            read_row_blk_size: 100
            read_col_blk_size: 100

        filter:
            enabled: True
            method: bregman
            block_pad: 300
            lee_filter:
                window_size: 3
            guided_filter:
                radius: 1
                eps: 3
                ddepth: -1
            bregman:
                lambda_value: 20
            anisotropic_diffusion:
                weight: 1
            line_per_block: 1000
            n_cpus: 8

        initial_threshold:
            maximum_tile_size:
                x: {tile_size}
                y: {tile_size}
            minimum_tile_size:
                x: 40
                y: 40
            selection_method: ['chini', 'bimodality']
            tile_selection_twele: [0.09, 0.8, 0.97]
            tile_selection_bimodality: 0.7
            extending_method: 'gdal_grid'
            threshold_method: 'ki'
            threshold_bounds:
                co_pol: [-28, -11]
                cross_pol: [-28, -18]
            multi_threshold: True
            adjust_if_nonoverlap: True
            low_dist_percentile: 0.99
            high_dist_percentile: 0.01
            number_cpu: -1
            tile_average: True
            line_per_block: 300

        fuzzy_value:
            line_per_block: 200
            intensity:
                intensity_mid_alpha: 1
            hand:
                member_min: 0
                member_max: 15
            slope:
                member_min: 0.5
                member_max: 15
            reference_water:
                member_min: 0.8
                member_max: 0.95
            area:
                member_min: 0
                member_max: 40
            dark_area:
                cross_land: -18
                cross_water: -24
            high_frequent_water:
                water_min_value: 0.1
                water_max_value: 0.9

        region_growing:
            initial_threshold: 0.81
            relaxed_threshold: 0.51
            line_per_block: 400

        masking_ancillary:
            land_cover_darkland_list: ['Bare sparse vegetation', 'Urban', 'Moss and lichen']
            land_cover_darkland_extension_list: ['Grassland', 'Shrubs']
            land_cover_water_label: ['Permanent water bodies']
            co_pol_threshold: -14.6
            cross_pol_threshold: -22.8
            water_threshold: 0.05
            extended_darkland: True
            extended_darkland_minimum_pixel: 3
            extended_darkland_water_buffer: 10
            hand_variation_mask: True
            hand_variation_threshold: 2.5
            line_per_block: 400
            number_cpu: 1

        refine_with_bimodality:
            minimum_pixel: 4
            lines_per_block: 500
            number_cpu: 1
            thresholds:
                ashman: 1.5
                Bhattacharyya_coefficient: 0.97
                bm_coefficient: 0.7
                surface_ratio: 0.1

        inundated_vegetation:
            enabled: auto
            dual_pol_ratio_max: 12
            dual_pol_ratio_min: 7
            dual_pol_ratio_threshold: 8
            cross_pol_min: -26
            line_per_block: 300
            target_area_file_type: 'auto'
            target_worldcover_class: ['Herbaceous wetland']
            target_glad_class: ['112-124', '200-207', '19-24', '125-148']
            filter:
                enabled: True
                method: lee
                block_pad: 300
                lee_filter:
                    window_size: 3
                guided_filter:
                    radius: 1
                    eps: 3
                    ddepth: -1
                bregman:
                    lambda_value: 20
                anisotropic_diffusion:
                    weight: 1
                line_per_block: 1000

        debug_mode: False
"""
    with open(path, "w") as f:
        f.write(content)


def write_runconfig_yaml(path, frame_info, algo_yaml_path, scratch_dir, output_dir):
    """Write main runconfig YAML."""
    content = f"""runconfig:
    name: d832_benchmark

    groups:
        pge_name_group:
            pge_name: DSWX_NI_PGE

        input_file_group:
            input_file_path:
                - {frame_info['gcov_h5']}
            input_file_historical_path:
            input_mgrs_collection_id:

        dynamic_ancillary_file_group:
            dem_file: {frame_info['dem']}
            dem_file_description:
            worldcover_file: {frame_info['landcover']}
            worldcover_file_description:
            glad_classification_file:
            glad_classification_file_description:
            reference_water_file: {frame_info['watercover']}
            reference_water_file_description:
            hand_file: {frame_info['hand']}
            hand_file_description:
            shoreline_shapefile:
            shoreline_shapefile_description:
            algorithm_parameters: {algo_yaml_path}
            mean_backscattering:
            standard_deviation_backscattering:

        static_ancillary_file_group:
            static_ancillary_inputs_flag: False
            mgrs_database_file:
            mgrs_collection_database_file:

        primary_executable:
            product_type: dswx_ni

        product_path_group:
            product_path: {output_dir}
            scratch_path: {scratch_dir}
            sas_output_path: {output_dir}
            product_version: 0.4
            output_imagery_format: COG
            output_imagery_compression: DEFLATE
            output_imagery_nbits: 32
            output_spacing: 30

        browse_image_group:
            save_browse: False
            browse_image_height: 1024
            browse_image_width: 1024
            flag_collapse_wtr_classes: True
            exclude_inundated_vegetation: False
            set_not_water_to_nodata: False
            set_hand_mask_to_nodata: True
            set_layover_shadow_to_nodata: True
            set_ocean_masked_to_nodata: False
            save_tif_to_output: False

        log_file: None
"""
    with open(path, "w") as f:
        f.write(content)


def load_cfg(frame_key, frame_info, tile_size, scratch_dir, output_dir):
    """Create config files and load RunConfig."""
    algo_yaml = str(BENCHMARK_ROOT / f"algo_{frame_key}_tile{tile_size}.yaml")
    rc_yaml = str(BENCHMARK_ROOT / f"rc_{frame_key}_tile{tile_size}.yaml")

    write_algo_yaml(algo_yaml, tile_size)
    write_runconfig_yaml(rc_yaml, frame_info, algo_yaml, scratch_dir, output_dir)

    parser = _get_parser()
    args = parser.parse_args([rc_yaml])
    return RunConfig.load_from_yaml(rc_yaml, "dswx_ni", args)


def get_pol_sets(cfg):
    """Determine polarization sets from config."""
    pol_mode = cfg.groups.processing.polarization_mode
    pol_list = cfg.groups.processing.polarizations
    if pol_mode == "MIX_DUAL_POL":
        return [DSWX_NI_POL_DICT["DV_POL"], DSWX_NI_POL_DICT["DH_POL"]]
    elif pol_mode == "MIX_SINGLE_POL":
        return [DSWX_NI_POL_DICT["SV_POL"], DSWX_NI_POL_DICT["SH_POL"]]
    elif pol_mode == "MIX_DUAL_H_SINGLE_V_POL":
        return [DSWX_NI_POL_DICT["DH_POL"], DSWX_NI_POL_DICT["SV_POL"]]
    elif pol_mode == "MIX_DUAL_V_SINGLE_H_POL":
        return [DSWX_NI_POL_DICT["DV_POL"], DSWX_NI_POL_DICT["SH_POL"]]
    return [pol_list]


def ensure_glad_placeholder(scratch_dir):
    """Create a placeholder interpolated_glad.tif if missing.

    The masking_with_ancillary step unconditionally loads this file.
    A zero-filled raster means 'no GLAD classes to mask'.
    """
    glad_path = os.path.join(scratch_dir, "interpolated_glad.tif")
    if os.path.exists(glad_path):
        return
    # Copy geometry from an existing raster
    ref_path = os.path.join(scratch_dir, "interpolated_landcover.tif")
    if not os.path.exists(ref_path):
        return
    ref_ds = gdal.Open(ref_path)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        glad_path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal.GDT_Byte
    )
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.GetRasterBand(1).Fill(0)
    out_ds.FlushCache()
    out_ds = None
    ref_ds = None
    logger.info(f"  Created placeholder {glad_path}")


def remove_tile_dependent_files(scratch_dir):
    """Remove files produced by tile-dependent steps."""
    for pattern in TILE_DEPENDENT_GLOBS:
        for fpath in glob(os.path.join(scratch_dir, pattern)):
            os.remove(fpath)


def save_key_outputs(scratch_dir, dest_dir, pol_str, pols):
    """Save key comparison files from scratch_dir to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    files_to_save = [
        f"intensity_threshold_filled_{pol}.tif" for pol in pols
    ] + [
        f"mode_tau_filled_{pol}.tif" for pol in pols
    ] + [
        f"initial_water_{pol}.tif" for pol in pols
    ] + [
        f"merged_fill_gap_{pol_str}.tif",
        f"merged_removed_false_water_{pol_str}.tif",
        f"refine_landcover_binary_{pol_str}.tif",
        f"region_growing_output_binary_{pol_str}.tif",
        f"fuzzy_image_{pol_str}.tif",
    ]
    saved = []
    for fname in files_to_save:
        src = os.path.join(scratch_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))
            saved.append(fname)
    return saved


def read_raster(path):
    """Read a GeoTIFF as numpy array."""
    if not os.path.exists(path):
        return None
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    ds = None
    return arr


def compute_metrics(water_a, water_b):
    """Compute IoU and F1 between two binary water maps."""
    a = (water_a > 0).astype(bool)
    b = (water_b > 0).astype(bool)

    intersection = float(np.sum(a & b))
    union = float(np.sum(a | b))
    tp = intersection
    fp = float(np.sum(a & ~b))
    fn = float(np.sum(~a & b))

    iou = tp / union if union > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "iou": float(iou), "f1": float(f1),
        "precision": float(precision), "recall": float(recall),
        "water_pixels_a": int(np.sum(a)),
        "water_pixels_b": int(np.sum(b)),
        "intersection": int(intersection), "union": int(union),
        "changed_pixels": int(np.sum(a != b)),
        "total_pixels": int(a.size),
    }


def find_water_map(directory, pol_str):
    """Find the most refined water map."""
    candidates = [
        f"merged_fill_gap_{pol_str}.tif",
        f"merged_removed_false_water_{pol_str}.tif",
        f"refine_landcover_binary_{pol_str}.tif",
        f"region_growing_output_binary_{pol_str}.tif",
    ]
    for pol in pol_str.split("_"):
        candidates.append(f"initial_water_{pol}.tif")
    for fname in candidates:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            return path, fname
    return None, None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Only run frames that don't have all saved results yet
    frames_to_run = {k: v for k, v in FRAMES.items()
                     if not all(
                         os.path.exists(str(RESULTS_DIR / f"saved_{k}_tile{ts}"))
                         for ts in TILE_SIZES)}
    if not frames_to_run:
        frames_to_run = FRAMES  # run all if all cached
    for frame_key, frame_info in frames_to_run.items():
        logger.info(f"\n{'#'*70}")
        logger.info(f"  FRAME: {frame_key}")
        logger.info(f"{'#'*70}")

        missing = [k for k, v in frame_info.items() if not os.path.exists(v)]
        if missing:
            logger.error(f"Missing files for {frame_key}: {missing}")
            continue

        scratch_dir = str(RESULTS_DIR / f"scratch_{frame_key}")
        output_dir = str(RESULTS_DIR / f"output_{frame_key}")
        os.makedirs(scratch_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Load config with default tile size to get pol info
        cfg_base = load_cfg(frame_key, frame_info, 400, scratch_dir, output_dir)
        pol_sets = get_pol_sets(cfg_base)
        pol_list = cfg_base.groups.processing.polarizations
        pol_str = "_".join(pol_list)

        # ── Phase 1: Shared mosaic + pre_processing ──────────────
        base_filtered = os.path.join(scratch_dir, f"filtered_image_{pol_str}.tif")
        preproc_timings = {}

        if not os.path.exists(base_filtered):
            logger.info("Phase 1: Running mosaic + pre_processing")

            t0 = time.time()
            mosaic_gcov_frame.run(cfg_base)
            preproc_timings["mosaic"] = time.time() - t0
            logger.info(f"  mosaic: {preproc_timings['mosaic']:.1f}s")

            for pol_set in pol_sets:
                cfg_copy = copy.deepcopy(cfg_base)
                cfg_copy.groups.processing.polarizations = pol_set
                t0 = time.time()
                pre_processing.run(cfg_copy)
                ptag = "_".join(pol_set)
                preproc_timings[f"preproc_{ptag}"] = time.time() - t0
                logger.info(f"  preproc ({ptag}): {preproc_timings[f'preproc_{ptag}']:.1f}s")
        else:
            logger.info("Phase 1: Re-using existing pre-processed data")

        # Ensure GLAD placeholder exists (masking step requires it)
        ensure_glad_placeholder(scratch_dir)

        # ── Phase 2: Per-tile-size runs (in-place) ───────────────
        frame_results = {"preproc_timings": preproc_timings}
        water_maps = {}  # tile_size -> numpy array

        for tile_size in TILE_SIZES:
            run_key = f"{frame_key}_tile{tile_size}"
            saved_dir = str(RESULTS_DIR / f"saved_{run_key}")
            logger.info(f"\n  Phase 2: {run_key}")

            # Check if already completed (saved outputs exist)
            saved_water_path, _ = find_water_map(saved_dir, pol_str)
            if saved_water_path is not None:
                logger.info(f"    Already completed — loading from {saved_dir}")
                wm = read_raster(saved_water_path)
                if wm is not None:
                    water_maps[tile_size] = wm
                frame_results[tile_size] = {
                    "status": "cached",
                    "water_pixels": int(np.sum(wm > 0)) if wm is not None else None,
                }
                continue

            # Clean tile-dependent files from previous run
            remove_tile_dependent_files(scratch_dir)

            # Load config with this tile size
            cfg = load_cfg(frame_key, frame_info, tile_size, scratch_dir, output_dir)

            timings = {}
            try:
                for pol_set in pol_sets:
                    cfg_copy = copy.deepcopy(cfg)
                    cfg_copy.groups.processing.polarizations = pol_set
                    ptag = "_".join(pol_set)

                    t0 = time.time()
                    initial_threshold.run(cfg_copy)
                    timings[f"initial_threshold_{ptag}"] = time.time() - t0
                    logger.info(f"    initial_threshold ({ptag}): "
                                f"{timings[f'initial_threshold_{ptag}']:.1f}s")

                    t0 = time.time()
                    fuzzy_value_computation.run(cfg_copy)
                    timings[f"fuzzy_value_{ptag}"] = time.time() - t0

                    t0 = time.time()
                    region_growing.run(cfg_copy)
                    timings[f"region_growing_{ptag}"] = time.time() - t0

                    t0 = time.time()
                    masking_with_ancillary.run(cfg_copy)
                    timings[f"masking_ancillary_{ptag}"] = time.time() - t0

                    t0 = time.time()
                    refine_with_bimodality.run(cfg_copy)
                    timings[f"refine_bimodality_{ptag}"] = time.time() - t0

                timings["total_phase2"] = sum(timings.values())

            except Exception as e:
                logger.error(f"    Pipeline error: {e}")
                import traceback
                traceback.print_exc()
                frame_results[tile_size] = {"error": str(e), "timings": timings}
                continue

            # Save key outputs
            saved = save_key_outputs(scratch_dir, saved_dir, pol_str, pol_list)
            logger.info(f"    Saved {len(saved)} files to {saved_dir}")

            # Read water map
            water_path, water_src = find_water_map(saved_dir, pol_str)
            water_map = read_raster(water_path) if water_path else None
            if water_map is not None:
                water_maps[tile_size] = water_map

            frame_results[tile_size] = {
                "timings": timings,
                "water_map_source": water_src,
                "water_pixels": int(np.sum(water_map > 0)) if water_map is not None else None,
                "total_pixels": int(water_map.size) if water_map is not None else None,
                "saved_files": saved,
            }

        # ── Phase 3: Cross-compare ───────────────────────────────
        comparisons = {}
        ref = 400
        for ts in TILE_SIZES:
            if ts != ref and ts in water_maps and ref in water_maps:
                metrics = compute_metrics(water_maps[ts], water_maps[ref])
                comparisons[f"tile{ts}_vs_tile{ref}"] = metrics
                logger.info(
                    f"  {frame_key} tile{ts} vs tile{ref}: "
                    f"IoU={metrics['iou']:.6f} F1={metrics['f1']:.6f} "
                    f"changed={metrics['changed_pixels']}"
                )
        if 200 in water_maps and 800 in water_maps:
            m = compute_metrics(water_maps[200], water_maps[800])
            comparisons["tile200_vs_tile800"] = m

        # Threshold surface comparison
        threshold_stats = {}
        for ts in TILE_SIZES:
            sd = str(RESULTS_DIR / f"saved_{frame_key}_tile{ts}")
            for pol in pol_list:
                path = os.path.join(sd, f"intensity_threshold_filled_{pol}.tif")
                arr = read_raster(path)
                if arr is not None:
                    valid = arr[arr != -50]
                    if len(valid) > 0:
                        threshold_stats[f"tile{ts}_{pol}"] = {
                            "mean": float(np.mean(valid)),
                            "std": float(np.std(valid)),
                            "min": float(np.min(valid)),
                            "max": float(np.max(valid)),
                        }

        all_results[frame_key] = {
            "tile_results": {str(k): v for k, v in frame_results.items()},
            "comparisons": comparisons,
            "threshold_stats": threshold_stats,
        }

    # ── Save and report ──────────────────────────────────────────
    results_path = RESULTS_DIR / "d832_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("D832 BENCHMARK SUMMARY")
    print("=" * 70)
    for fk, fd in all_results.items():
        print(f"\n--- {fk} ---")
        for ts_key, data in fd["tile_results"].items():
            if ts_key == "preproc_timings":
                continue
            if "error" in data:
                print(f"  tile={ts_key}: ERROR - {data['error'][:80]}")
            elif "timings" in data:
                t = data["timings"]
                it = sum(v for k, v in t.items() if "initial_threshold" in k)
                tot = t.get("total_phase2", 0)
                print(
                    f"  tile={ts_key}: phase2={tot:.1f}s "
                    f"init_thresh={it:.1f}s "
                    f"water_px={data.get('water_pixels', '?')}"
                )
            elif data.get("status") == "cached":
                print(f"  tile={ts_key}: (cached) water_px={data.get('water_pixels', '?')}")

        for ck, m in fd.get("comparisons", {}).items():
            pct = m["changed_pixels"] / m["total_pixels"] * 100
            print(
                f"  {ck}: IoU={m['iou']:.6f} F1={m['f1']:.6f} "
                f"changed={m['changed_pixels']} ({pct:.4f}%)"
            )

    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
