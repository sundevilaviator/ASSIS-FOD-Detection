# Benchmark Methodology: FAA AC 150/5220-24

`src/benchmark_faa.py` scores a trained model against FAA Advisory Circular
150/5220-24, "Airport Foreign Object Debris (FOD) Detection Equipment,"
because that AC is the standard every commercial system in the Phase 2 gap
analysis is implicitly measured against, and none of them publish results in
its terms.

## What the AC actually requires

| Requirement | Standard |
|---|---|
| Minimum detectable object | Unpainted metal cylinder, 3.1 cm (h) × 3.8 cm (dia.); white/grey/black sphere, 4.3 cm dia. |
| Size-distribution coverage | ≥90% of mixed debris items ≤4 in (10.2 cm) in any dimension |
| Multi-object discrimination | Two qualifying objects ≤10 ft (3 m) apart identified as separate objects |
| False alarm rate | ≤1/day (visual-equipped systems), ≤3/day (non-visual), 90-day average |
| Location accuracy | Within 16 ft (5.0 m) of actual location |
| Response time | Continuous for always-on systems; otherwise ≤4 min inspection cycle |
| Environmental range | -25°F to +123°F, 90–100% RH, day/night/dusk, local 2-year storm rain/snow |

Source: FAA AC 150/5220-24 (see the link in the main README).

## What this repository's benchmark measures, and how it maps

| AC requirement | This repo's proxy metric | Gap between proxy and the real thing |
|---|---|---|
| Minimum detectable object size | Bounding-box area as % of frame, bucketed via `configs/fod.yaml: size_buckets` | No calibrated ground-sample-distance in FOD-A → can't convert pixels to centimeters without knowing camera height/angle/focal length per shot |
| Size-distribution coverage (≥90%) | Detection rate (recall) in the "small" bucket, checked against `faa_ac_150_5220_24.min_detection_rate_small_object` | Same calibration gap; also evaluated on a held-out split of the training dataset, not an independent site |
| False alarm rate | False positives per evaluated image | The AC defines this per full runway scan per day; converting requires knowing a real deployment's images-per-day, which is deployment-specific |
| Location accuracy (5 m) | Mean pixel-center error as % of frame diagonal | Same calibration gap — meters require a calibrated camera |
| Multi-object discrimination | Not yet implemented | Planned: would require identifying whether the model resolves two nearby ground-truth boxes as two predictions rather than one merged box |
| Environmental range | Detection rate stratified by FOD-A's own light-level (bright/dim/dark) and weather (dry/wet) metadata, via `--metadata-csv` | Not the AC's full range (no -25°F to +123°F temperature data, no snow) — only the light/weather categories FOD-A itself ships. Also: the metadata CSV's real column names were not independently verified in the session that wrote this integration (no network access to the live dataset) — auto-detection with an explicit override exists for this reason; check the "[metadata] Loaded light/weather labels for N images" line against the real image count before trusting this breakdown |

## Why report a proxy at all, instead of waiting for calibrated data

Two reasons. First, a proxy metric that is explicit about what it isn't
still lets performance be tracked consistently run-over-run — did the
small-object detection rate go up after the last training change, yes or no.
Second, it keeps the repository's claims honest by construction: the script
prints its own limitations every time it runs (see `LIMITATIONS` in
`src/benchmark_faa.py`), so a proxy number can't accidentally get quoted as
a verified FAA-compliance result without the caveat attached.

## Closing the gap to a real compliance-style test

1. Deploy against a fixed or PTZ camera with known height, tilt angle, and
   focal length (or run a one-time calibration pass with reference objects
   of known size at known distances).
2. Re-derive the pixel-to-centimeter conversion per camera position and
   re-bucket ground truth using actual centimeters instead of area
   percentage.
3. Log real deployment scan cadence (images or frames evaluated per day) to
   convert false-positives-per-image into a per-day figure.
4. Run the same benchmark script across at least one additional airport's
   camera footage (cross-site validation), not just a held-out split of the
   same source dataset.

None of this is done yet — it's the explicit roadmap for turning this
proxy benchmark into something closer to the AC's real test protocol.
