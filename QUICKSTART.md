# MonStim Analyzer 0.6.0 — Quick Start Guide

This guide gets a new user from an installed copy of MonStim Analyzer to a first reviewed plot. For the complete workflow and feature reference, open **Help > Show Help** in the application or read [Using MonStim Analyzer](docs/user/using_monstim.md).

## Install and launch

1. Download the release archive from [GitHub Releases](https://github.com/AEWorthy/MonStim-Analyzer/releases).
2. Extract the complete archive to a location where you can keep the application data.
3. Launch the MonStim Analyzer executable in the extracted folder. Keep the files in that folder together; do not move only the executable.

The first launch may take longer than later launches and Windows may show a warning because the release is not code-signed. The application currently reports itself as **v0.6.0 (beta)**. macOS distribution is not currently available.

## First analysis

### 1. Import an experiment

Choose **File > Import an Experiment** for one experiment, or **File > Import Multiple Experiments** when a parent folder contains several experiment folders. MonStim copies imported recordings into its managed data store; retain the original acquisition files as your archive.

The expected structure is:

```text
Experiment name/
  Dataset name/
    SessionA-1.csv
    SessionA-2.csv
    SessionB-1.csv
```

Files with the same session identifier form a session. Other file types in a dataset folder are ignored. Keep the hierarchy meaningful: **experiment > dataset > session > recording**.

### 2. Select and inspect data

Choose an **Analysis Profile** in the main window, then select an **Experiment**, **Dataset**, and **Session**. Inspect a raw and filtered trace before changing settings or interpreting a summary. Check channel identity, polarity, sampling rate, stimulus alignment, and response timing.

### 3. Check latency windows

Open **Edit > Session > Manage Latency Windows**. Start at Session scope while checking or developing timing. Dataset and Experiment scopes are bulk actions that copy the displayed window set to every included child session, so use them only when standardization is intentional.

Use a consistent name such as `M-wave` for the M-response window. M-max recognizes names listed in **File > Settings Center > Global Analysis > Latency windows > M-wave Recognition Names**. An empty recognition-name list disables automatic M-wave recognition.

### 4. Choose the analysis settings

Use **File > Settings Center** to review global defaults and manage reusable profiles. A profile can override eligible analysis settings, but it does not replace latency-window annotations already applied to sessions. Choose the amplitude method required by your analysis plan and keep the method, profile, bin size, windows, and M-max choices with your results.

### 5. Plot and review

1. Select **Session**, **Dataset**, or **Experiment** in the plot controls.
2. Choose a plot type and its options.
3. Click **Plot**.
4. Inspect the traces, contribution counts, missing values, and diagnostic notices before interpreting the result.

The **Reports** panel provides Session, Dataset, Experiment, and M-max reports. Use **Plot & Extract Data** for data represented by the current plot.

### 6. Export and save

- Use **File > Bulk Export Data…** for aggregated output across datasets or experiments.
- Use **File > Save Current Experiment** to save the selected experiment after reviewing changes.
- Record the active profile, amplitude method, latency windows, exclusions, bin size, and any M-max plateau or fallback result with the export.

## Common maintenance actions

- Use **Edit > Data Curation > Recording Exclusion Editor** to preview and review recording exclusions. Click **Preview** after changing criteria; **Apply** commits one undoable bulk action.
- Use **Edit > Data Curation > Manage Data…** to organize experiments and datasets.
- Use **File > Force Rebuild Data Catalog** when the active experiment listing is stale or incomplete.
- Use **Tools > Force Rebuild All Data Catalogs…** only for an expensive all-experiment rebuild.
- Use **Edit > Undo** and **Edit > Redo** for supported edits.

## Help and support

- **Help > Show Help** opens the in-app help library.
- **Help > Open Log Folder** opens the application logs.
- **Help > Save Error Report** creates a report for troubleshooting.

When reporting a problem, include the MonStim version, operating system, active profile, data level, exact steps, and any error message. Do not include identifiable data unless appropriate.

For detailed topics, see the [Help Library](docs/user/index.md), especially [Getting started](docs/user/getting_started.md), [Experiment import](docs/user/importing_experiments.md), [Latency windows](docs/user/latency_windows.md), [Settings Center](docs/user/settings_center.md), and [Troubleshooting](docs/user/troubleshooting.md).

## Running from source

Development commands must use the `monstim` Conda environment:

```pwsh
conda activate monstim
python main.py --debug
python -m pytest -q
```

The source application uses PySide6 and PyQtGraph for its GUI and interactive plots.
