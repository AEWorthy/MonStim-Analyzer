# Analysis profiles

## Purpose

Analysis profiles let you save a consistent set of analysis preferences for a protocol or project. Use a profile to make deliberate choices repeatable; do not treat a profile name as evidence that its defaults are scientifically appropriate.

## Before you begin

Start from a representative session. Confirm channel mapping, filter settings, latency windows, amplitude method, and stimulus binning before turning those choices into a reusable profile.

## Create or update a profile

1. Open **File > Analysis Preferences**.
2. Select the default profile to review shipped settings, or create/select a named profile.
3. Adjust only settings supported by the analysis plan.
4. Save the profile with a descriptive name and, where available, a description of the protocol or condition.
5. Replot a representative session and an aggregate level to verify the result.

Profiles can supply analysis defaults and latency-window presets. Session annotations remain the authority for applied latency windows; loading a profile does not silently overwrite every session's existing windows.

## Record and review

Record the selected profile with each export. If a profile changes filter cutoffs, bin size, amplitude method, M-max settings, or timing templates, analyses made before and after the change may not be directly comparable.

## Related topics

- [Analysis preferences and defaults](../science/configuration_reference.md)
- [Latency windows](latency_windows.md)
- [Exporting results](exporting_results.md)
- [Back to Help Library](index.md)
