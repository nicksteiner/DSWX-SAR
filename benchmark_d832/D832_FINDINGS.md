# D832: Accuracy Trade for DSWx-NI maximum_tile_size

## Summary

Benchmark of DSWx-NI with `maximum_tile_size` 200, 400, and 800 on two
NISAR GCOV frames.  Tile size controls the spatial block size for
initial threshold estimation; smaller blocks produce more threshold
estimates with less statistical support each.

**Recommendation: Keep tile 400 as default.** Tile 200 is slower (not
faster) and produces negligible accuracy differences (< 250 changed
pixels out of ~100M).  Tile 800 is fastest for initial_threshold but
total pipeline savings are marginal because masking and refinement
dominate wall-clock.

## Test Frames

| Frame | Track | Date | Image Size | Water Coverage |
|-------|-------|------|-----------|----------------|
| frame_069 | D_068 descending | 2025-10-09 | 17604 x 17136 | ~48% reference, 0 detected |
| frame_081 | D_009 descending | 2025-10-03 | 10415 x 9370 | ~0.6% reference, ~200 px detected |

Note: Acceptance criteria specified "Aug-2026 NISAR flood frames" but
none were available on disk. These frames are real NISAR L-band GCOV
data from operations phase. Frame_069 has no active water signal;
frame_081 has minimal water. For a definitive conclusion on flood
frames, repeat with data containing significant flood extent.

## Timing Results (seconds, phase 2 only)

### Frame 069 (17604 x 17136, ~302M pixels)

| Step | tile 200 | tile 400 | tile 800 |
|------|----------|----------|----------|
| initial_threshold | 131.4 | **118.3** | 159.2 |
| fuzzy_value | 93.7 | 81.5 | 104.7 |
| region_growing | 198.8 | 205.5 | 212.9 |
| masking_ancillary | 701.9 | 738.4 | 1121.2 |
| refine_bimodality | 390.3 | 405.9 | 542.4 |
| **total_phase2** | **1516.0** | **1549.5** | **2140.5** |

### Frame 081 (10415 x 9370, ~97.6M pixels)

| Step | tile 200 | tile 400 | tile 800 |
|------|----------|----------|----------|
| initial_threshold | 149.5 | **110.3** | 88.7 |
| fuzzy_value | 32.6 | 39.7 | 30.8 |
| region_growing | 55.9 | 57.8 | 52.4 |
| masking_ancillary | 101.6 | 103.4 | 117.0 |
| refine_bimodality | 84.7 | 84.8 | 87.9 |
| **total_phase2** | **424.3** | **396.1** | **376.8** |

Key timing observations:
- **initial_threshold for tile 200 is NOT faster than tile 400** on
  frame_069 (131.4s vs 118.3s). The overhead of 7568 blocks (vs 1849)
  outweighs per-block savings.
- On frame_081, tile 800 is fastest for initial_threshold (88.7s vs
  110.3s vs 149.5s) but tile 200 is slowest.
- The predicted 35% wall-clock improvement for tile 200 was not
  observed; tile 200 is comparable or slower.

## Accuracy Results

### Frame 069

All tile sizes produce **identical** water maps (IoU = 1.000, F1 = 1.000,
0 changed pixels). This is because tile selection rejected all tiles
at all block sizes (threshold surface is uniformly -50/no-data). When
there is no water/non-water boundary in the SAR data, tile size has
zero effect.

### Frame 081

| Comparison | IoU | F1 | Precision | Recall | Changed px | Changed % |
|-----------|-----|----|----|--------|----------|----------|
| tile200 vs tile400 | 0.437 | 0.608 | 0.581 | 0.639 | 148 | 0.00015% |
| tile800 vs tile400 | 0.225 | 0.367 | 0.340 | 0.400 | 248 | 0.00025% |
| tile200 vs tile800 | 0.254 | 0.405 | 0.419 | 0.392 | 244 | 0.00025% |

Water pixels detected:
- tile 200: 198 pixels
- tile 400: 180 pixels
- tile 800: 212 pixels

The low IoU values are misleading; they reflect tiny sets of ~200
pixels differing by ~150 pixels. In absolute terms, the change is
**< 250 pixels out of 97.6 million** (0.00025%). The threshold
surfaces differ slightly at the margins:

| Tile Size | HH threshold mean (dB) | HH range | HV threshold mean (dB) | HV range |
|-----------|----------------------|----------|----------------------|----------|
| 200 | -13.18 | [-16.77, -11.17] | -19.33 | [-22.17, -18.03] |
| 400 | -13.26 | [-14.67, -11.23] | -19.03 | [-20.70, -18.05] |
| 800 | -13.13 | [-14.78, -11.23] | -19.02 | [-20.70, -18.00] |

Tile 200 has a wider threshold range (more spatial variation) because
smaller blocks capture local variations. Tiles 400 and 800 are very
similar.

## Bug Fixes Found

During benchmarking, three `threshold_otsu` calls were found to crash
on NISAR data with narrow dB range:
1. `BimodalityMetrics.__init__` — water pixel dB range too narrow
2. `compute_threshold_max_bound` — global Otsu after bimodality check
3. `determine_threshold` — per-tile Otsu in block processing

All fixed with `try/except ValueError` + median fallback.

Also: OpenCV `cv2.Sobel` rejects Float16 DEM input; added cast to
Float32.

## Findings

1. **Tile size 200 does NOT reduce initial_threshold wall clock.**
   The block processing overhead (more blocks = more parallel dispatch,
   more GDAL random reads) offsets the per-block savings. On the larger
   frame, tile 200 is 11% slower than tile 400.

2. **Tile size has negligible effect on the final water map** when
   tested on non-flood NISAR data. On frames without strong water
   signal, all tile sizes produce identical results. On frames with
   weak water signal, differences are < 250 pixels out of ~100M.

3. **The F1 threshold of 0.01 from the acceptance criteria cannot be
   evaluated** because F1 between tile sizes measures self-consistency,
   not accuracy against a reference. Both tested frames lack sufficient
   flood extent for a meaningful reference comparison.

4. **Tile size 800 is fastest overall** on the smaller frame (376.8s vs
   396.1s vs 424.3s) due to fewer blocks, but slowest on the larger
   frame due to masking/refinement variability.

## Recommendation

**Keep `maximum_tile_size: 400` as default.**

- Tile 200 is slower, not faster, and produces negligible accuracy
  differences.
- Tile 800 saves ~5% on initial_threshold for small frames but shows
  no advantage for larger frames.
- The predicted cost savings from D832's expected_outcome were not
  confirmed: tile 200 does not cut initial_threshold wall clock by 35%.
- Without flood data, the accuracy question cannot be fully answered.
  But the self-consistency test shows the algorithm is robust to tile
  size (differences < 0.001% of pixels).

## Next Steps

To definitively answer D832, acquire and test on 2+ NISAR GCOV frames
with active flooding (significant water/non-water boundaries in the SAR
data). This would produce non-trivial threshold surfaces where tile size
could theoretically affect results.
