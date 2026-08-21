# Extrema peak-to-trough methods

MonStim provides two filtered, unrectified EMG amplitude methods: `extrema_ptt`
and `exclusive_extrema_ptt`.  Both find maxima and minima on the complete
recording first, then consider only extrema strictly inside each latency window.
They measure the largest adjacent maximum/minimum pair, so a window edge cannot
act as a peak or trough.

`extrema_ptt` evaluates every window independently; overlapping windows may use
the same exact extremum.  `exclusive_extrema_ptt` evaluates all windows for a
recording and channel together.  Windows whose start time is earlier have
priority (ties use the configured window order), and only the two extrema in a
selected earlier pair become unavailable to later windows.  This prevents reuse
of exact extrema; it does not mathematically separate superimposed physiological
responses.

A valid window without a complete pair has amplitude `0.0`. An invalid or
out-of-bounds window has `NaN`. The longform export records the reason, selected
sample/time/value extrema, counts excluded by earlier windows, and their owners
so an extrema amplitude can be audited. Selected extrema times are relative to
the primary stimulus, while sample indices remain full-recording indices.

M-max normalization always uses the same selected method for numerator and
denominator. Enable **Show PTT Extrema** in filtered EMG or Single EMG plot
options to display the selected extrema; these annotations are intentionally
unavailable on raw or rectified traces.
