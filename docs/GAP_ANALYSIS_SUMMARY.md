# FOD Detection — Competitive Landscape & Gap Summary

Condensed from the full ASSIS Phase 2 gap analysis (project research, August
2026). This is what scoped the module in this repository — kept here so the
reasoning behind the code travels with the code.

**Verification status:** the vendor descriptions, cost tiers, and technology
claims below come from vendor marketing pages and trade press gathered
during that research pass — they have NOT yet been independently re-verified
against primary sources (vendor spec sheets, FAA/GAO/ACI-NA/IATA data, or
direct vendor contact). Treat every specific number here (cost ranges,
detection ranges, deployment counts) as provisional until checked, same
standard applied to every other citation in this project. Do not cite a
specific figure from this table in RFE material without that check.

## Commercial / government-funded landscape

| System | Technology | Cost tier | Target | Limitation |
|---|---|---|---|---|
| Tarsier (QinetiQ, exclusively licensed to and commercialized by Moog Digital Airfield Solutions) | mmWave radar + MIL-SPEC day/night EO with NIR illumination | $1M–$5M+ | Large hubs | Fixed towers only, high capital cost. Publishes "100% detection out to 3,168 feet" and claims best-in-class ranking in FAA testing of fixed runway surveillance — a range-based figure, not stated by object size class, with no false-alarm rate or location accuracy given |
| FODetect/RunWize (Xsight) | Radar + EO fusion | $3M–$8M+ | Medium–large hubs | No published size/false-alarm data |
| iFerret (Stratech) | Vision-only EO | $1M–$3M | Regional–large | Degrades ~20% range in rain, weak at night |
| FOD Finder V2 (Trex) | Mobile radar + camera | $250K–$500K/vehicle | Smaller airports | Periodic sweep only, not continuous |
| ELVA-1 | Radar OEM module | $50K–$150K/sensor | Integrators | Component only, not a full system |
| FOD AI, on the InspectEx platform (Illuminex AI) | Vehicle-mounted sensors + edge AI, cloud analytics | Undisclosed; Standard and Premium tiers | Marketed to airports "both large and small" | Vehicle-mounted and inspection-pass based, not continuous. Standard: 50 ft inspection width at up to 25 mph; Premium: 100 ft at up to 50 mph, expandable with LiDAR/thermal. Deploys on existing airport vehicles or FOD sweepers, so no new *vehicle* is required — but dedicated sensor hardware is. No performance figures published in AC 150/5220-24 terms |
| Safe Pro (SPOTD/AFWERX SBIR) | Drone imagery CV | Government SBIR | Military airfields | Pre-commercial, dataset still in construction |
| sUAS + FastFlow ML (FAA/Volpe) | Drone CV | Federal R&D | FAA research only | 96% detection but high false positives, fails AC weather/low-light criteria |

## Confirmed open gaps this module targets

- **No system uses existing airport CCTV/PTZ cameras** for continuous FOD
  scanning — every reviewed system requires new dedicated hardware (radar
  tower, inspection vehicle, or drone).
- **Small-object detection is the documented weak point.** FOD-A, the most
  common public benchmark, is dominated by large objects; real airfield FOD
  is predominantly small (nuts, bolts, screws, fragments).
- **No vendor publishes results in FAA AC 150/5220-24's own three-part terms**
  (detection rate by size class, false-alarm rate per 90-day average,
  location accuracy in meters). Stated carefully, because vendors *do*
  publish performance claims: Moog's Tarsier page advertises "100% detection
  out to 3,168 feet" and a best-in-class ranking in FAA testing. That is a
  detection-range claim with no object size class attached, no false-alarm
  rate, and no location accuracy — so it is not comparable across systems
  and does not answer the AC's questions. The gap is the absence of a
  *common measure*, not the absence of any published numbers, and this
  module's claim should be worded that way everywhere it appears.
- ~~**No existing system amortizes FOD detection cost across other safety
  functions.**~~ **CORRECTED 2026-08-28 — this claim was wrong and is
  withdrawn.** Illuminex AI's InspectEx is explicitly a shared platform
  amortizing multiple airfield applications across common sensor and cloud
  infrastructure: FOD AI, PIDS AI (perimeter intrusion), SnowPro AI
  (snowbank profiling), EdgeGuard (edge-light protection), and Surface AI
  (pavement condition, announced as forthcoming). Its FOD collateral markets
  "additional plug and play applications" and added sensors as an expansion
  path. Multi-function amortization on shared infrastructure is therefore
  *not* an ASSIS-unique property, and must not be claimed as one. This
  correction is recorded rather than silently edited, because the earlier
  claim propagated into outreach documents.

- **The surviving infrastructure differentiator is narrower, and should be
  stated only in this narrow form.** Every reviewed system — Illuminex
  included — requires *dedicated sensor hardware*: a radar tower, a drone,
  or a sensor package mounted on a vehicle. Illuminex reduces the vehicle
  cost by mounting on vehicles an airport already operates, but the sensors
  are still a new purchase, and coverage is still a periodic inspection pass
  rather than continuous. No reviewed system performs FOD scanning on the
  fixed CCTV/PTZ cameras an airport has already installed and already
  maintains. That distinction — **existing fixed cameras and continuous
  coverage, versus new sensors and inspection passes** — is the real
  differentiator, and it is a narrower claim than the one it replaces.
- **No vendor or paper reviewed reports performance stratified by lighting
  or weather condition, even though FOD-A itself ships that metadata.**
  FOD-A (Munyer et al. 2021) includes light-level (bright/dim/dark) and
  weather (dry/wet) categorization separately from its bounding-box
  annotations — confirmed via the dataset's own paper and GitHub repo. None
  of the 8 systems in the landscape table above publish results broken out
  this way — re-confirmed 2026-08-28 against Illuminex AI's own FOD, PIDS and
  platform collateral and Moog's Tarsier product page. `src/benchmark_faa.py`'s environmental breakdown (added this
  session; see `docs/RESEARCH_LOG.md`) is a concrete, checkable way to close
  that specific reporting gap using data the field already has — not a
  claim of solving environmental robustness, just of measuring and
  publishing it honestly where nobody else currently does.

## What this module does NOT attempt (roadmap, not current scope)

- All-weather / night parity with $3M–$8M radar systems.
- Thermal-RGB sensor fusion.
- Cross-site validation across multiple airports.
- Open-world / unseen-debris generalization.
- Manufacturing- or assembly-phase FOD (a different environment and dataset
  problem from airfield FOD).
- **Tire-fragment / rubber-debris FOD (burst retreads, "alligator" rubber
  left on taxiways/runways).** Confirmed absent from FOD-A's 31 real object
  classes (verified directly against the live dataset, 2026-08-19 — see
  `docs/RESEARCH_LOG.md`). This is a genuine, previously undocumented gap in
  the field's own standard public benchmark, not just this module's scope
  choice. No public labeled dataset covering this was identified. Closing it
  requires a separately sourced and hand-annotated image set — not started.

Full analysis with sources: see the project's
`ASSIS Phase 2 - FOD Competitive Landscape and Gap Analysis` document.
