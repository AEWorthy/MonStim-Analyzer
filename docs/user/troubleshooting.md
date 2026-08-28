# Troubleshooting and quality checks

## Purpose

Use this page to isolate an unexpected result without changing several variables at once.

## Before you begin

Record the active experiment, dataset, session, profile, and selected method before changing settings. Those facts explain many apparent discrepancies.

## Empty plot or report

1. Confirm that data are loaded and not excluded.
2. Check the selected plot level and active child data.
3. Inspect latency-window names and timing; an out-of-bounds window can produce `NaN` or no usable result.
4. Review [Diagnostic notices](diagnostic_notices.md) and **Help → Open Log Folder**.

## Windows changed more data than expected

Dataset and Experiment scope copy the draft to every affected active child session. Check **Values from** and **Apply target** before Apply. A heterogeneity warning means differing child window sets will be replaced. See [Latency windows](latency_windows.md).

## Zero, `NaN`, or inconsistent amplitudes

- Confirm that the window contains enough samples at the session sampling rate.
- Inspect raw and filtered traces separately; filtered values depend on cutoffs/order and unrectified methods depend on polarity.
- For `extrema_ptt`, a valid window with no complete eligible pair is `0.0`; invalid/out-of-bounds windows are `NaN`. See [Extrema peak-to-trough methods](../science/extrema_peak_to_trough.md).
- For aggregate plots, verify bin size and included/excluded sessions.

## Implausible relative-to-M-max values

Confirm the M-wave window and amplitude method, then inspect the M-max plot and its plateau/fallback. If the protocol does not reach a stable high-stimulus response, do not interpret a fallback estimate as a confirmed physiological plateau. Review [M-max algorithm](../science/mmax_estimation.md) and [Configuration reference](../science/configuration_reference.md).

## Import or interface problems

Use [Experiment import](importing_experiments.md) for missing/misgrouped files. Use the [UI scaling guide](ui_scaling_guide.md) for a clipped interface or stale saved position. For support, include program version, OS, profile, data level, reproducible steps, and an error report; do not share identifiable data without permission.

## Related topics

- [Diagnostic notices](diagnostic_notices.md)
- [Exporting results](exporting_results.md)
- [Back to Help Library](index.md)
