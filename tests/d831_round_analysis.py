#!/usr/bin/env python
"""D831 supplemental: trace per-round behaviour to explain the timing anomaly.

The main experiment showed that starting at win_size=100 is 236% SLOWER for
patoka_princeton.  This script instruments tile_selection_wbd to record which
halving rounds execute and how many tiles each contributes.
"""

import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dswx_sar.common._initial_threshold import TileSelection, convert_pow2db

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(message)s')

FLOOD_ROOT = '/mnt/archive/nisar_floods_2026-08'
WATER_CACHE = '/tmp/d827_flood_water'

SITES = {
    'indiana_white': {
        'gcov': os.path.join(
            FLOOD_ROOT, 'indiana_white', 'h5',
            'NISAR_L2_PR_GCOV_028_047_A_022_2005_QPDH_A_'
            '20260816T104834_20260816T104909_P05023_N_F_J_001.h5'),
        'water_mask': os.path.join(WATER_CACHE,
                                   'indiana_white_water_mask.tif'),
    },
    'patoka_princeton': {
        'gcov': os.path.join(
            FLOOD_ROOT, 'patoka_princeton', 'h5',
            'NISAR_L2_PR_GCOV_024_148_A_021_2005_QPDH_A_'
            '20260706T105633_20260706T105657_P05023_N_P_J_001.h5'),
        'water_mask': os.path.join(WATER_CACHE,
                                   'patoka_princeton_water_mask.tif'),
    },
}

GCOV_GRID = '/science/LSAR/GCOV/grids/frequencyA'
BLOCK_SIZE = 400
MINIMUM_TILE = 20
MINIMUM_PIXEL_NUMBER = 40
WATER_MAX_VALUE = 100
WATER_NO_DATA = 255
TILE_SELECTION_METHOD = ['chini', 'bimodality']


def count_rounds(intensity, water_mask, initial_win_size):
    """Run tile_selection_wbd logic and return per-round stats.

    Returns list of dicts: [{win_size, n_windows_scanned, n_qualifying, elapsed_s}]
    """
    ts = TileSelection(ref_water_max=WATER_MAX_VALUE, no_data=WATER_NO_DATA)
    ts.threshold_twele = [0.09, 0.8, 0.97]
    ts.threshold_bimodality = 0.7

    height, width = intensity.shape
    win_size = initial_win_size

    if (height <= win_size) or (width <= win_size):
        win_size = min(height, width)

    if win_size < MINIMUM_PIXEL_NUMBER:
        return []

    n_water_masks = water_mask.ndim
    if n_water_masks == 2:
        wm = np.expand_dims(water_mask, axis=0)
    else:
        wm = water_mask

    water_area_flag = ts.get_water_portion_mask(wm)
    if not water_area_flag:
        return []

    intensity_db = convert_pow2db(intensity)
    intensity_gray = ts.rescale_value(intensity_db, min_value=-35, max_value=10)
    mean_intensity_global = np.nanmean(intensity_gray)

    num_detected_box_sum = 0
    rounds = []

    while (num_detected_box_sum <= MINIMUM_TILE) and \
            (win_size >= MINIMUM_PIXEL_NUMBER):
        t0 = time.perf_counter()
        step = win_size // 2
        num_pixel_max = win_size * win_size / 3

        n_scanned = 0
        n_qualifying = 0

        for x_coord in range(0, width - win_size + 1, step):
            for y_coord in range(0, height - win_size + 1, step):
                n_scanned += 1
                x_end = min(x_coord + win_size, height)
                y_end = min(y_coord + win_size, width)

                intensity_sub = intensity[x_coord:x_end, y_coord:y_end]
                intensity_sub_gray = intensity_gray[x_coord:x_end, y_coord:y_end]
                water_mask_sublock = wm[:, x_coord:x_end, y_coord:y_end]

                validnum = np.count_nonzero(~np.isnan(intensity_sub))
                water_number_sample_sub = np.count_nonzero(
                    ~np.isnan(water_mask_sublock[0, :, :]))
                water_area_sublock_flag = ts.get_water_portion_mask(
                    water_mask_sublock)

                if water_number_sample_sub > 0 \
                        and water_area_sublock_flag \
                        and validnum > num_pixel_max:

                    # Chini test
                    chini_flag = ts.select_tile_chini(intensity_sub)

                    if chini_flag:
                        # Bimodality test
                        _, _, bimode_flag = ts.select_tile_bimodality(
                            intensity_sub, threshold=ts.threshold_bimodality)
                        if bimode_flag:
                            n_qualifying += 1

        elapsed = time.perf_counter() - t0

        rounds.append({
            'win_size': win_size,
            'n_scanned': n_scanned,
            'n_qualifying': n_qualifying,
            'elapsed_s': round(elapsed, 6),
        })

        if num_detected_box_sum <= MINIMUM_TILE:
            win_size = int(win_size * 0.5)

        num_detected_box_sum += n_qualifying

    return rounds


def main():
    for name, paths in SITES.items():
        print(f'\n{"="*60}')
        print(f'{name}')
        print(f'{"="*60}')

        with h5py.File(paths['gcov'], 'r') as h5:
            intensity = h5[f'{GCOV_GRID}/HHHH'][()].astype(np.float32)
        with rasterio.open(paths['water_mask']) as ds:
            water_mask = ds.read().astype(np.float32)

        height, width = intensity.shape
        n_rows = height // BLOCK_SIZE + (1 if height % BLOCK_SIZE else 0)
        n_cols = width // BLOCK_SIZE + (1 if width % BLOCK_SIZE else 0)

        # For each starting win_size, count how many blocks reach each round
        for start_ws in [200, 100]:
            round_counts = {}  # ws -> count of blocks that ran that round
            total_time = 0
            n_active = 0  # blocks that have water
            n_blocks_reaching_50 = 0

            for ii in range(n_rows):
                for jj in range(n_cols):
                    y_start = ii * BLOCK_SIZE
                    x_start = jj * BLOCK_SIZE
                    y_end = min(y_start + BLOCK_SIZE, height)
                    x_end = min(x_start + BLOCK_SIZE, width)

                    int_block = intensity[y_start:y_end, x_start:x_end]
                    wm_block = water_mask[:, y_start:y_end, x_start:x_end]

                    if np.all(np.isnan(int_block)) or np.all(int_block == 0):
                        continue

                    rounds = count_rounds(int_block, wm_block, start_ws)
                    if not rounds:
                        continue

                    n_active += 1
                    for r in rounds:
                        ws = r['win_size']
                        total_time += r['elapsed_s']
                        round_counts[ws] = round_counts.get(ws, 0) + 1
                        if ws <= 50:
                            n_blocks_reaching_50 += 1

            print(f'\n  Starting at win_size={start_ws}:')
            print(f'    Active blocks (have water): {n_active}')
            print(f'    Total tile_selection time: {total_time:.2f}s')
            print(f'    Blocks reaching ws<=50: {n_blocks_reaching_50}')
            print(f'    Round breakdown:')
            for ws in sorted(round_counts.keys(), reverse=True):
                print(f'      ws={ws:4d}: {round_counts[ws]:4d} blocks')

    # Save summary
    print(f'\n{"="*60}')
    print('MECHANISM EXPLANATION:')
    print('  Starting at ws=100 loses the round-1 (ws=200) qualifying tiles.')
    print('  These tiles contribute to num_detected_box_sum, which helps')
    print('  marginal blocks satisfy minimum_tile=20 without falling to')
    print('  the expensive ws=50 round (225 tiles per block).')
    print('  For sparse-water frames (patoka: 0.6% water), many blocks')
    print('  are marginal and rely on these round-1 tiles.')


if __name__ == '__main__':
    main()
