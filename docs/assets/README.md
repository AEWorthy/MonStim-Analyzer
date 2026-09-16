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

`tools.capture_marketing_screenshots` captures full application views using a deterministic, in-memory synthetic H-reflex experiment. It activates an in-memory **Demo H-reflex** profile with enough pre-/post-stimulus display time to show both the M-wave and delayed H-reflex. The plots and latency editor are the real MonStim UI; no researcher data or experiment files are read, and no user profile is created or modified.

```powershell
conda run -n monstim python -m tools.capture_marketing_screenshots --all --output-dir docs/assets/demo
```

It produces `session-emg.png`, `single-recording.png`, `recruitment-curves.png`, and `latency-windows.png`. To regenerate one image after a UI change:

```powershell
conda run -n monstim python -m tools.capture_marketing_screenshots --screen single-recording --output docs/assets/demo/single-recording.png
```
