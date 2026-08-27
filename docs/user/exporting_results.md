# Exporting results

## Purpose

Use exports to move an analysis result into a spreadsheet, statistics workflow, or figure pipeline. Exported values are only interpretable with the selections and analysis choices that produced them.

## Choose an export path

- Use **Plot & Extract Data** when you want the data represented by the current plot.
- Use **File > Bulk Export Data** when you need aggregated output across multiple datasets or experiments.
- Choose a longform reflex-amplitude export when downstream analysis needs one row per active recording, channel, latency window, and method rather than a summarized curve.

## Before exporting

1. Confirm the selected experiment, dataset, or session.
2. Confirm included recordings, sessions, and datasets.
3. Record the active profile, amplitude method, bin size, latency windows, and any M-max normalization or manual override.
4. Inspect the plot or table for missing values, `NaN`, and unexpected contribution counts.
5. Use a descriptive filename and store the export with its analysis notes.

## Review the output

Open the exported file before distributing it. Check units, channel labels, stimulus-voltage grouping, window names, and row granularity. For extrema-based longform output, retain the selected-extrema fields and result reasons when the result will be audited.

## Related topics

- [Using MonStim Analyzer](using_monstim.md)
- [Analysis methods](../science/analysis_methods.md)
- [Troubleshooting](troubleshooting.md)
- [Back to Help Library](index.md)
