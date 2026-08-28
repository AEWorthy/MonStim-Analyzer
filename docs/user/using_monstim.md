# Using MonStim Analyzer

## Purpose

MonStim organizes stimulation recordings as **experiment > dataset > session > recording**. Select the level you want to inspect or plot, then choose a plot type and its options. This guide describes the normal user workflow; use the linked topics when you need detailed instructions.

## Before you begin

Import an experiment and select a session. Inspect at least one raw and filtered trace before changing analysis settings or interpreting a summary plot.

## Steps

1. Import data from **File** and select an experiment, dataset, and session.
2. Inspect a raw and filtered session trace before changing analysis settings.
3. Set latency windows at Session scope, then use broader scope only when standardizing child sessions is intentional.
4. Choose the amplitude method required by the analysis plan.
5. Review exclusions and diagnostic notices.
6. Plot, export, and record the profile and choices used.

Use **File > Save Current Experiment** to save the selected experiment. Most editing actions are also saved as part of the normal workflow and can be undone with **Edit > Undo**; redo is available through **Edit > Redo**.

## Common tasks

| Goal | Where to start |
| --- | --- |
| Import one or several experiments | **File > Import an Experiment** or **Import Multiple Experiments**; see [Importing experiments](importing_experiments.md). |
| Change profiles, analysis defaults, or application behavior | **File > Settings Center**; see [Settings Center](settings_center.md). |
| Create or standardize timing windows | **Edit > Session/Dataset/Experiment > Manage Latency Windows**; see [Latency windows](latency_windows.md). |
| Exclude a set of recordings after review | **Edit > Data Curation > Recording Exclusion Editor**; see [Recording exclusion editor](recording_exclusion_editor.md). |
| Change channel names or invert polarity | **Edit > Session/Dataset/Experiment**. |
| Export aggregate data | **File > Bulk Export Data**. |
| Rebuild a stale active-experiment listing | **File > Force Rebuild Data Catalog**. |

## Plots and their levels

The available plot types change with the selected level.

| Level | Typical plots |
| --- | --- |
| Session | EMG overlays, individual recordings, reflex curves, reflex averages, suspected H-reflexes, and latency-window distributions. |
| Dataset | Average reflex curves, M-max, maximum H-reflex, and latency-window distributions. |
| Experiment | Average reflex curves, M-max, and maximum H-reflex. |

Open several plots if useful. Plot options determine the calculation method, channels, signal form, display flags, normalization, and other plot-specific choices. When exporting plotted data, inspect the columns and settings so the export remains interpretable outside MonStim.

## Related topics and support

Use **Help > Show Help** for this library, **Help > Open Log Folder** to find application logs, and **Help > Save Error Report** to prepare diagnostic information for support. A useful report includes the program version, operating system, selected profile, data level, exact steps, and a screenshot or error message. Do not include identifiable data unless it is appropriate to share.

[Back to Help Library](index.md)
