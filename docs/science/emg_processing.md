# EMG processing and transformations

## Purpose

MonStim keeps the acquired recording available and derives display/analysis signals from it. Check the raw trace when investigating acquisition problems; use the filtered trace when checking the timing and morphology used by most amplitude methods.

## Processing order

For EMG channels, MonStim applies the configured Butterworth bandpass filter using forward-and-reverse filtering. This avoids a net phase shift but does not make an unsuitable cutoff valid. The shipped filter is 100–3500 Hz, fourth order; confirm that the cutoffs are appropriate for the acquisition rate and scientific protocol.

Identified force and length channels use pre-stimulus baseline subtraction instead of the EMG bandpass filter. Other non-EMG channel types pass through without this filtering. A configured polarity inversion is applied after the channel's processing step.

## Signal forms in plots

- **Raw** shows the acquired signal.
- **Filtered** shows the processed signal used by the standard amplitude methods.
- **Rectified** shows the absolute value of raw or filtered samples for visual review or methods that explicitly use rectification.

Rectification is not a permanent change to your source recording. Likewise, polarity inversion changes the sign of the processed signal; it changes signed averages but not RMS, average rectified amplitude, AUC, or peak-to-trough magnitude.

## What amplitude methods use

Unless a method explicitly says otherwise, MonStim calculates amplitude inside the selected latency window from the **filtered, unrectified** trace. The method determines whether sign, absolute magnitude, or extrema are used. See [Analysis methods](analysis_methods.md) for the exact calculations and [Extrema peak-to-trough methods](extrema_peak_to_trough.md) for the two extrema methods.

## Review before analysis

1. Confirm the channel type and physical units.
2. Inspect the raw trace for clipping, artifacts, and timing problems.
3. Inspect the filtered trace for an appropriate response window and unexpected ringing or distortion.
4. Confirm polarity and latency windows before comparing amplitudes.
5. Record any profile changes with the exported analysis.

For the shipped filter and other profile defaults, see [Configuration reference](configuration_reference.md).

## Related topics

- [Analysis methods](analysis_methods.md)
- [Latency windows](../user/latency_windows.md)
- [Back to Help Library](../user/index.md)
