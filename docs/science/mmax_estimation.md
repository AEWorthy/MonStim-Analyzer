# M-max estimation and review

## Purpose

M-max in MonStim is an estimate derived from M-wave amplitudes across stimulus levels. It is useful for normalization only after the M-wave timing, amplitude method, stimulus range, and resulting plateau/fallback have been reviewed. It is not a direct physiological validation of saturation.

## Inputs and scope

For each channel, MonStim takes paired stimulus voltages and M-wave amplitudes calculated with the current amplitude method. Dataset and experiment results aggregate child values; changing method, latency windows, exclusions, or binning can therefore change M-max.

## Step 1: smooth and search for a low-variation run

The M-wave amplitude sequence is smoothed with a third-order Savitzky–Golay filter. If no explicit smoothing length is supplied, MonStim uses `int(number_of_points × savgol_window_ratio)`, enforces a minimum length of 5, and makes the length odd.

MonStim tests contiguous runs beginning at the configured largest window size. A run is a candidate plateau when its standard deviation is below the configured absolute threshold. If no run is found, it retries with one fewer point down to the configured smallest window size. When more than one run qualifies at that size, the later qualifying run is used.

The threshold is not normalized by curve SD. Its numeric meaning depends on the amplitude unit and scale. A qualifying run is an algorithmic low-variation region, not proof of a physiological plateau.

## Step 2: estimate from a detected plateau

For the selected run, MonStim computes a maximum, a 95th percentile, a mean of values at/above the 80th percentile (top 20%), and a corrected mean. The corrected mean starts as the run mean and may add the difference between the mean of globally higher values and the mean of non-maximum run values.

Selection order is maximum, 95th percentile, top-20% mean, then corrected mean. A candidate is accepted only when it is no greater than `plateau_mean × validation_tolerance`. The final value is capped at the global maximum amplitude.

## Step 3: fallback when no plateau is found

MonStim sorts by stimulus voltage and uses the highest 25% of stimulus levels. It evaluates the maximum, 95th percentile, and mean in that region. It prefers the maximum, then the 95th percentile, when each is within `region_mean × validation_tolerance`; otherwise it uses the region mean.

This fallback is deterministic, but it is a high-stimulus heuristic. Treat it as a prompt to inspect the curve, not evidence that a plateau was established.

## Review before normalization

1. Confirm the M-wave latency window and amplitude method.
2. Check that the stimulus range reaches the intended high-response region.
3. Inspect whether the estimate used a plateau or fallback.
4. Record M-max settings and manual overrides with exported results.
5. Do not compare normalized values from materially different settings without an explicit rationale.

See [Configuration reference](configuration_reference.md) for the shipped parameters and their defaults.

## Related topics

- [Analysis preferences and defaults](configuration_reference.md)
- [Exporting results](../user/exporting_results.md)
- [Back to Help Library](../user/index.md)
