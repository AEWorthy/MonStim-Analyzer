# Using MonStim Analyzer 0.6.0

## Purpose

MonStim organizes stimulation recordings as **experiment > dataset > session > recording**. Select the level you want to inspect or plot, then choose a plot type and its options. This guide describes the normal workflow and the main actions available in the current application; linked topics provide detailed instructions.

## Before you begin

Import an experiment and select a session. Inspect at least one raw and filtered trace before changing analysis settings or interpreting a summary plot. Confirm that the selected hierarchy represents the biological and acquisition units you intend to compare.

## Recommended workflow

1. **Import and verify.** Use **File > Import an Experiment** for one experiment or **File > Import Multiple Experiments** for a parent folder containing several experiments. Confirm the experiment, dataset, and session selections after import.
2. **Inspect the signal.** Check a raw and filtered trace, channel identity, polarity, sampling rate, stimulus alignment, and response timing.
3. **Set latency windows.** Open **Edit > Session > Manage Latency Windows** while developing or checking timing. Dataset and Experiment scopes copy the draft to every included child session and should be used only for deliberate standardization. See [Latency windows](latency_windows.md).
4. **Select the analysis profile.** Use the main-window **Analysis Profile** selector to activate a profile. Use **File > Settings Center > Profiles** to create, duplicate, import, export, or edit profiles. See [Analysis profiles](analysis_profiles.md).
5. **Choose the amplitude method.** Use the method specified by the analysis plan. Review the equations, units, and limitations in [Analysis methods](../science/analysis_methods.md).
6. **Review quality and exclusions.** Read diagnostic notices and exclude recordings only with a documented reason. Use [Recording exclusion editor](recording_exclusion_editor.md) for previewed, reviewable bulk decisions.
7. **Review M-max before normalization.** Confirm the M-wave window and method, then check whether the estimate came from a detected plateau or the high-stimulus fallback. See [M-max estimation and review](../science/mmax_estimation.md).
8. **Plot, export, and document.** Inspect the plot or report, then retain the profile, method, windows, exclusions, bin size, and M-max choices with any export. See [Exporting results](exporting_results.md).

## Common tasks

| Goal | Where to start |
| --- | --- |
| Import one or several experiments | **File > Import an Experiment** or **Import Multiple Experiments**; see [Experiment import](importing_experiments.md). |
| Change program behavior or analysis defaults | **File > Settings Center**; see [Settings Center](settings_center.md). |
| Activate an analysis profile | Main-window **Analysis Profile** selector. |
| Create or standardize timing windows | **Edit > Session/Dataset/Experiment > Manage Latency Windows**; see [Latency windows](latency_windows.md). |
| Change channel names or polarity | **Edit > Session/Dataset/Experiment > Change Channel Names** or **Invert Channel Polarity**. |
| Edit dataset metadata | **Edit > Dataset > Edit Metadata**. |
| Exclude recordings after review | **Edit > Data Curation > Recording Exclusion Editor**; see [Recording exclusion editor](recording_exclusion_editor.md). |
| Organize experiments and datasets | **Edit > Data Curation > Manage Data…**. |
| Export current plotted data | **Plot & Extract Data**. |
| Export aggregated data | **File > Bulk Export Data…**; see [Exporting results](exporting_results.md). |
| Refresh the experiment list | **File > Refresh Experiments list**. |
| Repair the active experiment listing | **File > Force Rebuild Data Catalog**. |
| Repair every experiment listing | **Tools > Force Rebuild All Data Catalogs…**; this can be expensive. |
| Save or undo edits | **File > Save Current Experiment**, **Edit > Undo**, and **Edit > Redo**. |

## Plots, reports, and levels

The available plot types depend on the selected level. Typical uses include:

| Level | Typical plots and reports |
| --- | --- |
| Session | EMG overlays, individual recordings, reflex curves, reflex averages, suspected H-reflexes, latency-window distributions, and Session Info. Report. |
| Dataset | Average reflex curves, M-max, maximum H-reflex, latency-window distributions, and Dataset Info. Report. |
| Experiment | Average reflex curves, M-max, maximum H-reflex, and Experiment Info. Report. |

Plot options control the calculation method, channels, signal form, display flags, normalization, stimulus binning, and other plot-specific choices. Aggregate values can have different contribution counts when child sessions do not share the same named windows or exclusions. Review those counts before comparing curves.

## Latency-window and M-wave rules

Every latency-window set belongs to a session. The Dataset and Experiment editors are hierarchy-wide copy operations, not merely views of one shared object. Read the editor’s **Active**, **Values from**, and **Apply target** details before applying a change.

M-max uses the first latency window whose name matches a global **M-wave Recognition Names** entry, without regard to case. Configure those names in **File > Settings Center > Global Analysis > Latency windows**. The configured list replaces the shipped aliases; an empty list intentionally disables automatic M-wave recognition. Use one recognized M-wave window per session and inspect its placement on the trace.

## Curation and data integrity

The Recording Exclusion Editor stages criteria and decisions until **Apply**. Changing a criterion makes the preview stale, so run **Preview** again before applying. Existing exclusions are preserved unless you deliberately include a recording, and the complete applied bulk action can be reversed with **Edit > Undo**.

Catalog files are rebuildable listings of the managed data. Rebuild the active catalog from **File** when it is stale. The all-catalog operation in **Tools** is intended for maintenance and may take substantially longer.

## Saving, help, and support

Use **File > Save Current Experiment** after reviewing edits. Most editing actions are included in the normal undo history. Use **Help > Show Help** for this library, **Help > Open Log Folder** for application logs, and **Help > Save Error Report** for a support package.

A useful report includes the program version, operating system, selected profile, data level, exact steps, and a screenshot or error message. Do not include identifiable data unless it is appropriate to share.

## Related topics

- [Getting started](getting_started.md)
- [Troubleshooting](troubleshooting.md)
- [Help Library](index.md)
