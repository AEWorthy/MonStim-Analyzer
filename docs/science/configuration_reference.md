# Analysis preferences and defaults

## Purpose

Open **File > Settings Center** to review Program settings, Global Analysis defaults, and analysis profiles. A profile records software choices; it does not establish that a choice is scientifically correct for every muscle, species, amplifier, sampling rate, or protocol. Validate and record the profile used for each analysis.

## Core defaults

| Setting | Shipped default | What it changes |
| --- | ---: | --- |
| Stimulus bin size | 0.01 V | Groups nearby stimulus voltages for aggregate curves. Changing it can change means, error bars, and M-max inputs. |
| Displayed post-stimulus time | 8.0 ms | Default time shown after the stimulus in plots. |
| Displayed pre-stimulus time | 2.0 ms | Default baseline shown before the stimulus. |
| Default amplitude method | RMS | The starting method for plots; choose the method required by the analysis plan. |
| Default channel names | LG, TA, SOL | Labels used when the imported data do not provide channel names. |

## EMG filter defaults

| Setting | Shipped default | Review before use |
| --- | ---: | --- |
| Low cutoff | 100 Hz | It should retain the signal content you need. |
| High cutoff | 3500 Hz | It must remain below the Nyquist frequency for the recording. |
| Filter order | 4 | Changing it alters the transition region and transient behavior. |

MonStim uses forward-and-reverse filtering to avoid net phase shift. It does not silently substitute invalid cutoffs. See [EMG processing and transformations](emg_processing.md) for the signal path.

## M-max defaults

| Setting | Shipped default | What to know |
| --- | ---: | --- |
| Largest plateau run | 15 points | The first plateau length MonStim tests. |
| Smallest plateau run | 2 points | The shortest run it will consider after no larger run qualifies. Two points are weak evidence. |
| Plateau variation threshold | 0.3 | An absolute standard-deviation limit; its meaning depends on your signal scale. |
| Validation tolerance | 1.05 | Allows a candidate up to 5% above the relevant mean. |
| Smoothing-window ratio | 0.25 | Sets the Savitzky–Golay window from the number of stimulus levels. |
| M-wave recognition names | M-wave aliases | Global names that identify the M-response latency window for M-max. The user list replaces the shipped aliases; an empty list disables automatic recognition. |

The M-wave recognition names are global user settings, not profile settings. Edit them in **File > Settings Center > Global Analysis > Latency windows**. The M-max estimator also uses a third-order smoothing polynomial, a 95th-percentile candidate, a top-20% candidate, and a high-stimulus fallback. These are reproducibility settings, not physiological constants. See [M-max estimation](mmax_estimation.md) before using M-max normalization.

## Presets and appearance

Latency-window presets are templates for their named protocols, not universal timing recommendations. Colors and plot-style settings change presentation, not calculated amplitudes. A Dataset- or Experiment-level latency-window application copies the selected draft to included child sessions; review the target carefully before applying.

## Related topics

- [Analysis profiles](../user/analysis_profiles.md)
- [Settings Center](../user/settings_center.md)
- [M-max estimation and review](mmax_estimation.md)
- [Back to Help Library](../user/index.md)
