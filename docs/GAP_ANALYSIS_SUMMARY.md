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
| Tarsier (QinetiQ/Moog) | mmWave radar + EO | $1M–$5M+ | Large hubs | Fixed towers only, high capital cost |
| FODetect/RunWize (Xsight) | Radar + EO fusion | $3M–$8M+ | Medium–large hubs | No published size/false-alarm data |
| iFerret (Stratech) | Vision-only EO | $1M–$3M | Regional–large | Degrades ~20% range in rain, weak at night |
| FOD Finder V2 (Trex) | Mobile radar + camera | $250K–$500K/vehicle | Smaller airports | Periodic sweep only, not continuous |
| ELVA-1 | Radar OEM module | $50K–$150K/sensor | Integrators | Component only, not a full system |
| FODᴬᴵ (Illuminex AI) | Vehicle camera + edge AI | Undisclosed, low-cost positioning | 2026 trials (Pittsburgh, Toronto Pearson, Columbus, Savannah) | New entrant, vehicle-hardware-dependent, not continuous |
| Safe Pro (SPOTD/AFWERX SBIR) | Drone imagery CV | Government SBIR | Military airfields | Pre-commercial, dataset still in construction |
| sUAS + FastFlow ML (FAA/Volpe) | Drone CV | Federal R&D | FAA research only | 96% detection but high false positives, fails AC weather/low-light criteria |

## Confirmed open gaps this module targets

- **No system uses existing airport CCTV/PTZ cameras** for continuous FOD
  scanning — every reviewed system requires new dedicated hardware (radar
  tower, inspection vehicle, or drone).
- **Small-object detection is the documented weak point.** FOD-A, the most
  common public benchmark, is dominated by large objects; real airfield FOD
  is predominantly small (nuts, bolts, screws, fragments).
- **No vendor publishes results in FAA AC 150/5220-24's own terms**
  (detection rate by size class, false-alarm rate per 90-day average,
  location accuracy in meters).
- **No existing system amortizes FOD detection cost across other safety
  functions.** Every vendor reviewed sells FOD detection as a standalone
  purchase. Even the cheapest commercial option (~$250K–$500K/vehicle) is a
  dedicated line item. An airport of any size already running camera-based
  PPE, badge-misuse, or fall detection has no way to add FOD coverage as an
  incremental cost on that same infrastructure — this is a deployability
  differentiator based on shared-infrastructure economics, not a claim that
  the module targets small airports specifically.
- **No vendor or paper reviewed reports performance stratified by lighting
  or weather condition, even though FOD-A itself ships that metadata.**
  FOD-A (Munyer et al. 2021) includes light-level (bright/dim/dark) and
  weather (dry/wet) categorization separately from its bounding-box
  annotations — confirmed via the dataset's own paper and GitHub repo. None
  of the 8 systems in the landscape table above publish results broken out
  this way. `src/benchmark_faa.py`'s environmental breakdown (added this
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
