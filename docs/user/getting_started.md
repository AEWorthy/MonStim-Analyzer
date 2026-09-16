# Getting started with a defensible analysis

## Purpose

This guide is the shortest path from imported recordings to results that can be reviewed later. Inspect the signal and settings before relying on a summary plot or export.

## 1. Import and verify the hierarchy

Import an experiment, then confirm that **experiment**, **dataset**, and **session** identify the intended biological and acquisition units. Use [Experiment import](importing_experiments.md) for the required folder layout. A dataset does not guarantee that its sessions share identical latency windows.

For a supported non-MonStim source, use **File > Import using Add-on…** instead. Select the source file or folder; MonStim checks enabled compatible add-ons and uses the only high-confidence match. It asks you to choose when recognition is ambiguous and refuses to overwrite an existing experiment. See [Importer add-ons](importer_addons.md).

## 2. Inspect a representative session

Open a filtered EMG or Single EMG plot. Check the stimulus-aligned trace, channel identity, polarity, sampling rate, and response timing. Use raw traces to inspect acquisition; most amplitude methods operate on filtered, unrectified data.

## 3. Set and audit latency windows

Open **Edit > Session > Manage Latency Windows**. The live context states the active data, the representative values loaded into the draft, and the scope that **Apply** overwrites. Start at Session scope. Choose Dataset or Experiment only when replacing every affected child-session window set is intended. See [Latency windows](latency_windows.md).

## 4. Choose and record an amplitude method

Choose the method specified by the analysis plan; do not switch methods merely because one gives a preferred result. [Analysis methods](../science/analysis_methods.md) gives the equations, units, and limits for every available method.

## 5. Review exclusions, notices, and M-max

Exclude recordings only with a documented reason. Review [Diagnostic notices](diagnostic_notices.md). Before using relative-to-M-max results, inspect the M-wave window and whether M-max used a detected plateau or high-stimulus fallback; see [M-max algorithm](../science/mmax_estimation.md).

## 6. Preserve an audit trail

Record the profile, window scope, amplitude method, exclusion rules, and exported-data identifier. MonStim saves annotation changes. For unexpected results, start with [Troubleshooting](troubleshooting.md) rather than changing several settings at once.

For further reference on the main window, plot levels, reports, and common commands, see [Using MonStim Analyzer](using_monstim.md).

## Related topics

- [Analysis profiles](analysis_profiles.md)
- [Exporting results](exporting_results.md)
- [Importer add-ons](importer_addons.md)
- [Updating MonStim](updates.md)
- [Back to Help Library](index.md)
