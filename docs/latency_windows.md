# Latency Windows and Latency Window Editor

This document covers two major areas:

1. Editing and reusing latency windows (clipboard workflow)
2. Heterogeneous latency window handling across Sessions / Datasets / Experiments

For the diagnostic notice system (icons, warning/info codes, glossary) see `diagnostic_notices.md`.

---
## 1. Clipboard (Copy / Paste) Workflow

The Latency Windows dialog includes lightweight Copy / Paste functionality so you can transfer window configurations between Sessions, Datasets, or an entire Experiment without creating a preset.

## Why This Exists
Frequently, the same latency windows are appropriate across multiple sessions in a dataset (e.g. homogeneous stimulation paradigms). Previously you either had to: (1) recreate them manually each time, or (2) promote them to a global preset (which adds long‑term clutter). The transient clipboard solves this by keeping a deep copy of the windows only for the current application run.

## How It Works
1. Open the Latency Windows dialog for any Session / Dataset / Experiment.
2. Configure the windows as desired.
3. Click **Copy All** – the current list of windows (including per‑channel start times) is stored in an in‑memory clipboard.
4. Open the Latency Windows dialog for another scope and click **Paste**.
5. Confirm replacement (the paste always replaces the existing list in the dialog view).
6. Press **Apply** or **OK** to persist them via the existing command/undo system.

The clipboard is:
* In‑memory only (not written to disk; cleared on program exit)
* Deep‑copied on copy and on paste (no accidental shared mutation)
* Independent of latency window presets (those remain file‑backed in config)

## Undo / Redo
Paste itself only changes the dialog contents. The change is committed when you click **Apply** or **OK**, which triggers the existing `SetLatencyWindowsCommand` and therefore fully supports Undo/Redo.

## Channel Count Mismatch
If you paste into a session with a different number of channels, the start times & durations are automatically broadcast/truncated to fit the channel count (matching the existing Add/Apply behavior).

## Limitations / Future Ideas
* No multi‑item history – only the most recent copy is stored.
* Clipboard is not shared across separate running instances of the application.
* Could be extended later for other annotation types if needed.

---
## 2. Heterogeneous Latency Window Handling

Historically the application assumed every Session inside a Dataset (and every Dataset inside an Experiment) shared the exact same ordered set of latency windows (e.g. `M-wave`, `H-reflex`). This is no longer required.

### Key Principles
* Latency windows are now defined **at the Session level** only.
* Dataset & Experiment computations build a **case‑insensitive union** of all window names they contain.
* Aggregation for a specific window includes only Sessions (Dataset level) or Sessions inside Datasets (Experiment level) that actually define that window.
* The representative legacy properties (`dataset.latency_windows`, `experiment.latency_windows`) are still exposed for backward compatibility but should not be used for per‑window aggregation logic—use the new helpers instead.

### Helper APIs
| Level | Purpose | Helper |
|-------|---------|--------|
| Dataset | Union of window names across Sessions | `Dataset.unique_latency_window_names()` |
| Dataset | Which Sessions have a given window | `Dataset.window_presence_map()` |
| Dataset | Per-window amplitudes (session granularity) | `Dataset.get_lw_reflex_amplitudes()` |
| Dataset | Aggregated curve (means / stdevs / contributing session counts) | `Dataset.get_average_lw_reflex_curve()` |
| Experiment | Union of window names across all Sessions (all Datasets) | `Experiment.unique_latency_window_names()` |
| Experiment | Window presence at dataset granularity | `Experiment.dataset_window_presence_map()` |
| Experiment | Aggregated multi-dataset curve | `Experiment.get_average_lw_reflex_curve()` |

### Contribution Counting
When producing an average reflex curve for a window, two arrays are returned:
* `voltages`: Binned stimulus voltages (rounded to configured `bin_size`).
* `n_sessions`: Number of Sessions that contributed at least one recording in that voltage bin (skips Sessions missing the window).

This allows the UI to annotate legends like: `Late (n=3 @ 2.0V)` or visually de‑emphasize sparse bins.

### Canonical M-wave Recognition
The system treats the following (case‑insensitive) names as equivalent when determining if the canonical M-wave window is “present”:
```
{ "m-wave", "m_wave", "m wave", "mwave", "m-response", "m_response", "m response" }
```
If none of these variants are present at a hierarchy level, higher-level aggregation may surface a `missing_m_wave_window` notice (see `diagnostic_notices.md`).

### Heterogeneity Detection
Flags propagate upward:
* Dataset: `has_heterogeneous_latency_windows` if any Session’s ordered window name list differs from the first Session.
* Experiment: True if ordered Dataset lists differ, **or** any Dataset is itself heterogeneous.

These flags drive aggregation behavior; for diagnostic visibility refer to the notice system (`diagnostic_notices.md`).

---
## Practical Workflow Examples

### Adding a New Exploratory Window to One Session
1. Open Session A, add a `Late` window.
2. Dataset icon turns yellow (heterogeneous) if other Sessions lack `Late`.
3. Decide whether to replicate the window (Copy → other Sessions) or leave as exploratory (legend will show lower `n_sessions`).

### Diagnosing Flat Normalized Curves
* Check for `heterogeneous_mmax_failures`—some Sessions may lack valid M-max, distorting relative normalization.

### Cleaning Overlaps
* Hover Session icon → find `excessive_window_overlap` details → edit start/duration until code disappears.

---
## Future Extensions (Planned / Possible)
* Per-window legend tagging (visual emphasis per window)
* Optional template synchronization across Datasets
* Optional strict mode: block analysis actions while critical warnings present.

---
Last updated: Added heterogeneous latency window handling & notice glossary (2025-09).

## Troubleshooting
* Paste button disabled: no windows have been copied in the current run.
* Unexpected per‑channel values after paste: verify source windows were in Per‑channel mode; global mode copies a single start time to all channels.

---
Previous clipboard section last updated: transient clipboard feature introduction.
