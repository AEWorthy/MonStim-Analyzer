# Documentation screenshots

Screenshots in this directory must be captured from the current application using synthetic or empty-state data only. Never use identifiable research data in public documentation. The capture tool uses a temporary non-sensitive plug-in directory.

Capture all safe static dialogs from the repository root:

```powershell
conda run -n monstim python -m tools.capture_docs_screenshot --all --output-dir docs/assets
```

Or capture one dialog:

```powershell
conda run -n monstim python -m tools.capture_docs_screenshot --screen update-manager --output docs/assets/update-manager.png
conda run -n monstim python -m tools.capture_docs_screenshot --screen addon-source-selection --output docs/assets/addon-source-selection.png
```

The same command also captures the real latency-window editor, Recording Exclusion Editor, and bulk-export dialog using deterministic in-memory synthetic EMG recordings and disposable temporary folders. It never loads user experiments.

## Demo screenshots

`tools.capture_demo_screenshots` captures full application views using deterministic, in-memory synthetic experiments. It includes H-reflex recruitment, a 100 Hz vibration train, and a stretch ramp-hold-release protocol. The plots and latency editor are the real MonStim UI; no researcher data or experiment files are read, and no user profile is created or modified.

```powershell
conda run -n monstim python -m tools.capture_demo_screenshots --all --output-dir docs/assets/demo
```

It produces H-reflex EMG, individual-recording, reflex-curve, recruitment, M-max, and latency-window views plus vibration and stretch EMG views. The vibration and stretch captures intentionally hide PTT extrema; the stretch view includes TA, LG, force, and length. To regenerate one image after a UI change:

```powershell
conda run -n monstim python -m tools.capture_demo_screenshots --screen single-recording --output docs/assets/demo/single-recording.png
```

## Bundled protocol demos

`tools.generate_demo_experiments` creates
`docs/resources/demo_experiments/monstim-synthetic-protocol-demos.zip`. It
contains three native, synthetic experiments: H-reflex recruitment, a 100 Hz
vibration intensity series with TA/LG bulk EMG, and a stretch ramp-hold-release
intensity series with TA, LG, force, and length. The archive intentionally
contains no research data.

```powershell
conda run -n monstim python -m tools.generate_demo_experiments
```
