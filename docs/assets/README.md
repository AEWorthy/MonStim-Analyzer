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
