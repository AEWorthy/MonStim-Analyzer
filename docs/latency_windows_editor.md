# Latency Windows Editor Clipboard

The Latency Windows dialog now includes lightweight Copy / Paste functionality so you can transfer window configurations between Sessions, Datasets, or an entire Experiment without creating a preset.

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

## Troubleshooting
* Paste button disabled: no windows have been copied in the current run.
* Unexpected per‑channel values after paste: verify source windows were in Per‑channel mode; global mode copies a single start time to all channels.

---
Last updated: transient clipboard feature introduction.
