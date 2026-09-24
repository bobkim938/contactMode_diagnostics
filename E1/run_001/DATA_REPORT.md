# E1 data-generation report

Generated 23 candidates: **23 accepted**, **0 rejected**. No replacement sampling.

## Frozen calibration

- MuJoCo: 3.12.0; primary timestep: 0.000125 s.
- Quasi-stick ceiling: 0.000612227425 m/s; clear-slide floor: 0.0024489097 m/s.
- Friction-utilization boundary: 0.98; minimum loaded force: 0.1 N.
- Full sensitivity ranges: frozen_thresholds.json. Thresholds were fixed before candidate simulation and before any model fitting.

## Acceptance by family

| Family | Attempted | Accepted | Rejected |
|---|---:|---:|---:|
| a2 | 8 | 8 | 0 |
| a2_quasi_control | 3 | 3 | 0 |
| a3 | 6 | 6 | 0 |
| a3_contact_control | 3 | 3 | 0 |
| a3_free_control | 3 | 3 | 0 |

## Rejected episodes

None in this batch. This is not a guarantee for every parameter combination.

## Startup and data use

All startup and rejected rows are retained. Training eligibility is the intersection of an accepted manifest entry, analysis_mask, and valid_transition. Splits were assigned by whole episode before simulation; all cycles in an episode stay together.

Calibration traces and finer-step verification traces are excluded from accepted_splits.json. mode, rho, forces, cycle ID, and support phase are diagnostics; the mode-agnostic model uses qpos/qvel and actual ctrl to predict qpos_next/qvel_next.

## A3 resolution checks

Every A3 candidate (including same-mode controls) was rerun at half the primary timestep. Transition episodes require exactly one complete separation/recontact per retained drive cycle, gap >= 1 mm, free duration >= 20 ms with >= 40 samples, and contact dwell >= 50 ms. Matched exit/entry and duration must agree within 1 ms; peak gap within max(2%, 50 micrometres); the first 20 ms normal impulse after recontact within max(5%, 0.005 N s). Detailed comparisons are in each episode.json.

## Limits

These are operational labels and numerical acceptance tests, not physical validation of a marker. E1b matching/common-support checks, model capacity checks, and prediction-error analysis remain subsequent work. The included same-mode control families do not guarantee matched state changes for every transition. Rejected parameter regions must be reported when interpreting the accepted dataset.
