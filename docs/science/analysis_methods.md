# Analysis methods: equations, units, and limits

## Purpose

MonStim calculates each method over the selected latency window on one channel and recording. Unless a method says otherwise, the input is the **filtered, unrectified** trace. Let the samples in the selected window be \(x_1, \ldots, x_n\), sampled at \(f_s\) Hz.

| Method | Calculation | Units | Important limit |
| --- | --- | --- | --- |
| `rms` | \(\sqrt{\frac{1}{n}\sum_i x_i^2}\) | signal units | Positive and negative deflections both contribute. |
| `average_rectified` | \(\frac{1}{n}\sum_i \left\vert x_i \right\vert\) | signal units | Baseline/noise contributes positively. |
| `average_unrectified` | \(\frac{1}{n}\sum_i x_i\) | signal units | Opposing deflections can cancel. |
| `peak_to_trough` | \(\max(x_i)-\min(x_i)\) | signal units | Any noise extrema in the window can determine the result. |
| `auc` | \(\sum_i \left\vert x_i \right\vert / f_s\) | signal units·s | Depends on window duration as well as response size. |
| `extrema_ptt` | Largest adjacent eligible maximum/minimum pair | signal units | Uses detected extrema strictly inside the window. |
| `exclusive_extrema_ptt` | `extrema_ptt` with selected samples claimed by earlier windows | signal units | Requires all windows for one recording/channel together. |

An empty ordinary-method window returns `NaN`. The extrema methods have additional result semantics; see [Extrema peak-to-trough methods](extrema_peak_to_trough.md).

## Choose a method

The program does not infer the scientifically correct method. Choose one from the protocol or analysis plan and retain it for comparable analyses. `rms` is the shipped default because it is a stable magnitude measure, not because it is universally preferable. Use `average_unrectified` only when signed mean deflection is the intended quantity. Use `auc` only when duration dependence is intended rather than a nuisance.

For peak-based methods, inspect the plotted extrema and latency window. A large value can reflect the desired waveform, noise, movement artifact, or a window that includes an adjacent response; the calculation alone cannot decide.

## Review filtering, polarity, and aggregation

Filtering and polarity are applied before the usual amplitude calculations. Polarity inversion changes the sign of unrectified averages but not RMS, rectified mean, AUC, or peak-to-trough magnitude. Correct channel units and amplifier scaling remain the user’s responsibility.

Dataset and experiment curves group stimulus voltages by rounding to the configured `bin_size`: \(\mathrm{round}(V / b) b\). This is an aggregation policy, not an estimate of the original stimulus voltage. Changing bin size can change group membership, means, error bars, and M-max inputs; record it with exported results.

## Related topics

- [EMG processing and transformations](emg_processing.md)
- [Exporting results](../user/exporting_results.md)
- [Back to Help Library](../user/index.md)
