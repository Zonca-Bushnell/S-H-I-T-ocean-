# Origin Eddy Detection Performance Validation

Date: 2026-09-09

Environment:

```text
Python: D:\Util\lever\02_miniforge\envs\eddy_detection\python.exe
Data root: F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current
Filter root: F:\Global Ocean Ensemble Physics Reanalysis Kuroshio Current\FILTER
Result root: E:\DATA\01_Eddy_correspond\03_Original_detection
```

## Implemented Acceleration Path

The extraction keeps the original Hua detection-to-shape scientific chain, but
uses local execution optimizations:

- year-sharded detection workers, matching the yearly NetCDF file layout;
- default `10 deg x 10 deg` tile-top15 surface seed selection;
- source-depth default via `max_depth_m <= 0`, which uses all NetCDF depth
  levels;
- default 12-way detection-worker concurrency for multi-year runs;
- 512 MB NetCDF chunk cache per detection worker;
- default skip of nonessential detection/tracking QA plots;
- fast parquet engine auto-selection;
- MATLAB/GPU precompute option for surface SSH extrema candidates;
- explicit small 6x6 solver for the subgrid quadratic speed-center fit, avoiding
  local Windows BLAS/LAPACK crashes while preserving the same fitted model;
- empty shape products when no tracks pass shape eligibility, instead of
  failing short strict-window smoke runs.

## Smoke Benchmark: 1993-01-01 To 1993-01-03

Command output root:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\benchmarks_19930101_19930103_v5
```

Settings:

```text
max_depth_m = 1000
max_candidates_per_day = 20
lifetime_min_days = 1
radius_min_m = 10000
detect_parallel = 1
detect_shard_mode = year
```

Results:

| Run | Return Code | Elapsed |
| --- | ---: | ---: |
| MATLAB/GPU candidate precompute | 0 | 20.88 s |
| Python year-shard, no candidate cache | 0 | 6.21 s |
| Python year-shard, MATLAB candidate cache | 0 | 6.04 s |

Output counts:

| Run | Centers | Pass Layers | Vertical Objects | Tracks | Shape Tracks |
| --- | ---: | ---: | ---: | ---: | ---: |
| No candidate cache | 150 | 90 | 3 | 3 | 0 |
| MATLAB candidate cache | 252 | 192 | 7 | 6 | 1 |

Interpretation:

At this small domain/grid/window size, surface extrema selection is not the
runtime bottleneck. MATLAB/GPU candidate precompute is useful as an optional
candidate source and matrix backend, but its startup cost dominates very short
smoke runs.

## Monthly Benchmark: 1993-01-01 To 1993-01-31

Command output root:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\benchmarks_19930101_19930131_d1000_c80
```

Settings:

```text
max_depth_m = 1000
max_candidates_per_day = 80
lifetime_min_days = 5
radius_min_m = 10000
detect_parallel = 1
detect_shard_mode = year
```

Results:

| Run | Return Code | Elapsed |
| --- | ---: | ---: |
| MATLAB/GPU candidate precompute | 0 | 20.49 s |
| Python year-shard, no candidate cache | 0 | 15.11 s |
| Python year-shard, MATLAB candidate cache | 0 | 14.85 s |

Output counts:

| Run | Pass Layers | Vertical Objects | Tracks | Shape Tracks |
| --- | ---: | ---: | ---: | ---: |
| No candidate cache | 3721 | 172 | 58 | 9 |
| MATLAB candidate cache | 3470 | 158 | 56 | 8 |

Interpretation:

The MATLAB candidate cache does not materially reduce wall time for this
Kuroshio grid at 80 candidates/day. The larger speed gains are from the Python
pipeline changes that reduce I/O churn and avoid plotting overhead.

## Full-Year Validation: 1993

Output root:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\fast_chain_1993_d1000_c80_life30
```

Settings:

```text
max_depth_m = 1000
max_candidates_per_day = 80
lifetime_min_days = 30
radius_min_m = 10000
detect_parallel = 1
detect_shard_mode = year
```

Counts:

| Product | Count |
| --- | ---: |
| Days | 365 |
| Surface candidates | 29182 |
| Center rows | 73560 |
| Pass layers | 44514 |
| Vertical frame objects | 2014 |
| Feature tracks | 720 |
| Tracks with length >= 2 | 403 |
| Tracks with length >= 3 | 235 |
| Shape tracks | 0 |

The life30 run produced no eligible shape tracks because the longest track in
this 1000 m / 80 candidates per day validation is 23 days, below the strict
`lifetime_days > 30` eligibility rule.

Stage timing from file timestamps:

| Stage | Approximate Time |
| --- | ---: |
| Detection | 66 s |
| Feature/group tracking | 21 s |
| Catalog + shape adapter | 2 s |
| End-to-end log window | 106 s |

## Non-Empty Shape Validation: 1993 Life20

Output root:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\fast_chain_1993_d1000_c80_life20
```

Settings are the same as the full-year validation, except:

```text
lifetime_min_days = 20
```

Counts:

| Product | Count |
| --- | ---: |
| Pass layers | 44514 |
| Vertical frame objects | 2014 |
| Feature tracks | 720 |
| Eligible shape tracks | 2 |
| Shape class `mixed` | 2 |

This verifies that the final shape stage is functional on the extracted chain;
the empty life30 output is a data/threshold outcome, not a pipeline failure.

## Current Default Validation: Tile10 Top15 Full Source Depth

Output root:

```text
E:\DATA\01_Eddy_correspond\03_Original_detection\tile10_top15_full_depth_1993_life20
```

Settings:

```text
candidate_selection = tile_topn
tile_lon_deg = 10
tile_lat_deg = 10
tile_top_n = 15
max_depth_m = 0
lifetime_min_days = 20
radius_min_m = 10000
min_valid_layers = 1
detect_parallel = 12
detect_shard_mode = year
```

The current source file has 51 depth levels, down to about 1516 m. The output
center table may have a shallower maximum accepted layer because strict
contiguous extension stops an object at its first failed layer; this is separate
from the source-depth cap.

Counts:

| Product | Count |
| --- | ---: |
| Days | 365 |
| Surface candidates | 27961 |
| Center rows | 71894 |
| Pass layers | 44002 |
| Vertical frame objects | 1956 |
| Feature tracks | 695 |
| Tracks with length >= 2 | 393 |
| Tracks with length >= 3 | 229 |
| Eligible shape tracks | 2 |
| Shape class `mixed` | 2 |

Stage timing from file timestamps:

| Stage | Approximate Time |
| --- | ---: |
| Detection | 81 s |
| Feature/group tracking | 22 s |
| Catalog + shape adapter | 3 s |
| End-to-end log window | 127 s |

Because this validation covers one year, `detect_parallel = 12` cannot show its
multi-year benefit. A 30-year run will create one shard per year, so annual
workers can run concurrently until either CPU or F: drive NetCDF I/O becomes the
limiting factor.

## Current Recommendation

For this local Kuroshio dataset, the default fast path should remain pure Python
with:

```text
--candidate-selection tile_topn
--tile-lon-deg 10
--tile-lat-deg 10
--tile-top-n 15
--max-depth-m 0
--detect-parallel 12
--detect-shard-mode year
--netcdf-chunk-cache-mb 512
--skip-axis-examples
--skip-plots
```

Keep MATLAB/GPU candidate precompute as an optional experimental backend. It is
not yet the primary acceleration lever unless the candidate-generation stage
becomes much larger, or unless the Hua circle-check/vertical-extension stages
are also moved into MATLAB/GPU kernels later.
