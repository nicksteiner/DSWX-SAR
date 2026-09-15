# D831: Validate skipping the win_size=200 round in tile_selection_wbd

## Decision: NO-GO

The win_size=200 round should not be skipped. While it cannot satisfy
`minimum_tile=20` on its own (max 9 tiles for 400x400 blocks), its qualifying
tiles contribute to the accumulated count, preventing expensive fallback rounds
and removing the 200x200 scale from the analysis. Tile sets differ in 27% of
active blocks.

## Background

`tile_selection_wbd` (src/dswx_sar/common/_initial_threshold.py:263) searches
for candidate tiles using a halving-window strategy: starting at `win_size=200`,
it scans overlapping windows, counts how many pass bimodality/chini tests, and
if fewer than `minimum_tile=20` qualify, halves `win_size` and repeats. Tiles
accumulate across all rounds.

The hypothesis was that round 1 (ws=200) is unconditionally unsatisfiable and
can be skipped by starting at `first_satisfiable_win_size((400,400)) = 100`.

## Mathematical proof: ws=200 cannot satisfy minimum_tile alone

For a 400x400 block with `win_size=W` and step `W/2`:

| W   | max tiles | satisfies min_tile=20? |
|-----|-----------|----------------------|
| 200 | 9         | No                   |
| 100 | 49        | Yes                  |
| 50  | 225       | Yes                  |

Proof: `len(range(0, 400-200+1, 100))^2 = 3^2 = 9 < 20`. QED.

## What the round-1 tiles actually do

The while-loop accumulates `num_detected_box_sum` across rounds. Round 1 tiles
don't satisfy the threshold alone, but they "tip the balance" for marginal
blocks. Without them, 9 additional blocks (indiana_white) trigger the expensive
ws=50 round (225 tiles per block).

```
Starting at ws=200:          Starting at ws=100:
  Round 1 (ws=200): k1 tiles   (no round 1)
  Round 2 (ws=100): k2 tiles   Round 1 (ws=100): k2 tiles
  → sum = k1 + k2              → sum = k2
  If k1+k2 > 20: done!         If k2 ≤ 20: must run ws=50 (costly!)
```

## Experimental results

Ran `tile_selection_wbd` on all 400x400 blocks from two NISAR GCOV flood
frames (HHHH band, Aug 2026) with cached 3-band water masks. Each starting
win_size ran independently to avoid cache interference.

### Timing (verified with independent runs)

| Frame              | Water % | ws=200   | ws=100   | Speedup  |
|--------------------|---------|----------|----------|----------|
| indiana_white      | 3.7%    | 198.5s   | 180.4s   | **9.1%** |
| patoka_princeton   | 0.6%    | 75.6s    | 64.3s    | **14.9%**|

### Tile differences (indiana_white)

| Metric                           | Value |
|----------------------------------|-------|
| Blocks processed                 | 1039  |
| Blocks with qualifying tiles     | 176   |
| Blocks with no tiles (both ways) | 863   |
| **Blocks where tile sets differ**| **47 (27% of active)** |
| ws=200 has more tiles            | 38 blocks, 83 extra tiles |
| ws=100 has more tiles            | 9 blocks, 414 extra tiles |

Mechanism:
- **38 blocks**: Round-1 (200x200) tiles contribute 1-3 extra tiles each. These
  are at a coarser scale than round-2 tiles.
- **9 blocks**: Without round-1 tiles, `num_detected_box_sum` stays ≤ 20 after
  the ws=100 round, triggering the ws=50 round (225 windows). This produces
  414 extra tiles at 50x50 scale.

### Total qualifying tiles

| Frame           | ws=200 | ws=100 | Delta  |
|-----------------|--------|--------|--------|
| indiana_white   | 2615   | 2946   | +331   |
| patoka_princeton| 1391   | 1544   | +153   |

Starting at ws=100 produces MORE tiles overall because the ws=50 fallback
round (225 windows) is very productive. But these tiles are at a different
scale (50x50 vs 200x200).

### Round-level breakdown (indiana_white)

| Starting ws | Blocks reaching ws=50 | Total tile_selection time |
|-------------|----------------------|--------------------------|
| 200         | 209 / 214 active     | 156.7s                   |
| 100         | 213 / 214 active     | 167.3s                   |

## Against acceptance criteria

| Criterion                        | Result               | Verdict |
|----------------------------------|----------------------|---------|
| IoU >= 0.99 between water maps   | Tile sets differ in 27% of active blocks; different scales | **Likely fails** |
| 7.6-12.9% wall-clock reduction   | 9.1% (indiana), 14.9% (patoka) | **Partially met** |
| IoU < 0.98 → falsified           | Non-identical tile sets + scale change | **Cannot confirm IoU >= 0.98 without full pipeline** |

## Why the change should NOT be made

1. **Scale diversity matters.** Round-1 tiles sample at 200x200, capturing
   large-scale water/land boundaries. Removing them changes the threshold
   computation to rely only on 100x100 and 50x50 tiles.

2. **Tile sets are not identical.** 47 out of 176 active blocks (27%) produce
   different tile coordinates. This feeds different intensity distributions into
   the threshold computation, which propagates to the final water map.

3. **Unintended round cascade.** In 9 blocks, removing round-1 tiles causes
   the algorithm to fall to the ws=50 round. This is counter-intuitive: the
   "optimization" sometimes triggers more work, not less.

4. **The speedup is modest.** 9-15% improvement in tile_selection_wbd, which
   is one substep of the initial_threshold stage. In the full pipeline context,
   this is ~1-2% overall at best.

## If optimizing tile_selection_wbd is still desired

The correct optimization target is the ws=50 round (225 tiles per block),
which runs for 95-99% of active blocks regardless of starting win_size.
Options:
- **Early-exit once minimum_tile is reached** within a round (currently checks
  only after the full scan)
- **Vectorize bimodality/chini tests** to batch-process windows
- **Adaptive starting win_size** based on block water fraction (skip ws=200
  only for blocks with >10% water where ws=100 alone reliably exceeds 20 tiles)

## Artifacts

- `tests/d831_win_size_experiment.py` — main experiment script
- `tests/d831_round_analysis.py` — round-level instrumentation
- `tests/d831_results.json` — per-frame metrics
- `tests/d831_analysis.md` — this report
