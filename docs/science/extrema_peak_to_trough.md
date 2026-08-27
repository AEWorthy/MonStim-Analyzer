# Extrema peak-to-trough methods

## Purpose

Use these methods when the intended amplitude is the distance between a physiologically meaningful adjacent peak and trough, rather than simply the largest and smallest samples in the window. They analyze the filtered, unrectified EMG trace.

## Two choices

- **Extrema peak-to-trough** evaluates each latency window independently. Overlapping windows may use the same detected extremum.
- **Exclusive extrema peak-to-trough** evaluates all windows for one recording and channel together. Earlier windows have priority; if starts tie, the order in the latency-window table breaks the tie. Once a pair is selected for an earlier window, its two extrema are unavailable to later windows.

Exclusive mode prevents reuse of the same detected samples. It does not separate superimposed physiological responses, so inspect the trace rather than treating exclusivity as proof of attribution.

## How MonStim chooses the pair

MonStim detects maxima and minima in the complete recording, then considers extrema strictly inside the latency window. It chooses the largest eligible adjacent maximum/minimum pair. A window edge is not treated as an extremum.

A valid window with no complete pair returns `0.0`. A window that is invalid or outside the recording returns `NaN`. These values mean different things: zero means no eligible pair was found; `NaN` means the requested measurement could not be made.

## Review and export

For filtered EMG and individual-recording plots, enable **Show PTT Extrema** to see the selected pair. The option is unavailable for raw or rectified signals because these methods are defined on the filtered, unrectified trace.

Longform exports include the selected extrema and the reason for the result, including exclusion by an earlier window in exclusive mode. Selected times are relative to the primary stimulus; sample indices refer to the full recording.

When normalizing to M-max, use the same amplitude method for the numerator and denominator. See [Analysis methods](analysis_methods.md) for the full method comparison and [Latency windows](../user/latency_windows.md) for window ordering.

## Related topics

- [Analysis methods](analysis_methods.md)
- [Latency windows](../user/latency_windows.md)
- [Back to Help Library](../user/index.md)
