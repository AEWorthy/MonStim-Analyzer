# Diagnostic Notice System

This document describes the cross-level diagnostic notice architecture used to surface data integrity and heterogeneity conditions via small icons in the GUI.

## Overview
Three hierarchical levels emit notices:

| Level | Object | Method | Icon Placement |
|-------|--------|--------|----------------|
| Session | `Session` | `Session.collect_notices()` | Session selector icon |
| Dataset | `Dataset` | `Dataset.collect_notices()` | Dataset selector icon |
| Experiment | `Experiment` | `Experiment.collect_notices()` | Experiment selector icon |

Each `collect_notices()` returns a list of dictionaries:
```jsonc
{ "level": "warning" | "info", "code": "short_snake_case", "message": "Human readable message" }
```

Icon colors:
* Yellow (⚠) if any `warning` present
* Light grey (ℹ) if only `info` notices

Commands (execute / undo / redo) trigger a refresh so icons always reflect the current state after structural edits (e.g., excluding recordings, changing latency windows, adding sessions).

---
## Canonical M-wave Window Recognition
A canonical M-wave window is considered present if any latency window name (case-insensitive) matches one of:
```
{ "m-wave", "m_wave", "m wave", "mwave", "m-response", "m_response", "m response" }
```
Absence at a hierarchy level produces `missing_m_wave_window`.

---
## Notice Code Glossary

### Session-Level Codes
| Code | Level | Meaning |
|------|-------|---------|
| `zero_or_negative_window` | warning | A latency window has duration ≤ 0 ms. |
| `missing_m_wave_window` | info | No canonical M-wave window variant defined. |
| `no_active_recordings` | warning | All recordings excluded (or none loaded). |
| `window_out_of_bounds` | warning | A window start/end exceeds acquisition time window. |
| `excessive_window_overlap` | info | ≥50% overlap (shorter window) between two windows on a channel. |
| `inconsistent_scan_rate` | warning | Mixed scan rates inside the Session. |
| `inconsistent_num_channels` | warning | Recordings have different channel counts. |
| `inconsistent_stim_delay` | warning | Stim delay differs across recordings. |
| `duplicate_stim_voltages` | warning | (Currently suppressed) Duplicate stimulus voltages detected. |
| `unsorted_stim_voltages` | warning | Voltages not monotonic non‑decreasing. |

### Dataset-Level Codes
| Code | Level | Meaning |
|------|-------|---------|
| `heterogeneous_latency_windows` | warning | Sessions define different window sets/order. |
| `mixed_scan_rates` | info | Sessions have different scan rates. |
| `missing_m_wave_window` | info | No Session provides a canonical M-wave window. |
| `no_active_session` | warning | Dataset contains zero active Sessions. |
| `single_session_only` | warning | Only one active Session (reduced statistical strength). |
| `heterogeneous_mmax_failures` | info | Some Sessions have calculable M-max, others fail. (NOTE: REMOVED FOR NOW, for performance reasons) |
| `high_latency_window_name_churn` | info | High fraction of rarely-used window names; aggregation may be unstable. |
| `inconsistent_scan_rate` | warning | Underlying inconsistency surfaced. |
| `inconsistent_num_channels` | warning | Channel mismatch across Sessions. |
| `inconsistent_stim_start` | warning | Start time mismatch (if implemented). |

### Experiment-Level Codes
| Code | Level | Meaning |
|------|-------|---------|
| `heterogeneous_latency_windows` | warning | Datasets (or their Sessions) differ in window sets/order. |
| `mixed_scan_rates` | info | Datasets differ in scan rate. |
| `missing_m_wave_window` | info | No canonical M-wave anywhere in the experiment. |
| `no_active_datasets` | warning | All datasets excluded or none loaded. |

---
## Heterogeneity Semantics
* Dataset-level heterogeneity flagged if any Session’s ordered window list differs from the first Session.
* Experiment-level heterogeneity flagged if ordered Dataset lists differ OR any Dataset is heterogeneous.
* Aggregation always uses the union of window names; per-voltage contribution counts (`n_sessions`) indicate breadth of support.

---
## Usage Guidance
* Resolve `warning` notices prior to final analysis whenever possible.
* Treat `info` notices as advisory context; they do not invalidate processing.
* High churn suggests standardizing naming (e.g., prefer a consistent `Late` window label).

---
## Future Extensions
* Per-window targeted notices surfaced in legends
* Caching / selective invalidation for very large experiments
* Optional strict mode blocking analysis on specific warnings

---
Last updated: 2025-09.
