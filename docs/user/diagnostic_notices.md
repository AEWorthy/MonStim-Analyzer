# Diagnostic notices

## Purpose

MonStim shows notices beside the selected session, dataset, or experiment when it finds a condition worth reviewing. A warning means that the condition can prevent a valid analysis or materially affect it. An information notice is a useful check, not a diagnosis and not an automatic reason to discard data.

Read the notice text itself: it identifies the affected level and usually the window, channel, or recording involved. Notices update after relevant edits and after undo or redo.

## Steps: what to do first

1. Resolve warnings before creating a final plot or export whenever possible.
2. Inspect the affected session before changing a dataset or experiment.
3. Keep an intentional exception only when it is justified in the analysis record.
4. Do not change windows simply to silence an overlap notice; overlapping M- and H-response windows can be scientifically appropriate. Review the trace and method instead.

## Session notices

| Notice | What it means | Suggested check |
| --- | --- | --- |
| No active recordings | Every recording is excluded, or none is available. | Review recording exclusions and restore recordings if appropriate. |
| Non-positive window duration | A latency window has zero or negative duration. | Correct the duration before analysis. |
| Window outside acquisition bounds | A window begins before the available signal or ends after it. | Check the recording duration and the window start/duration. |
| Windows overlap substantially | Two windows overlap by at least half of the shorter window on a channel. | Confirm that overlap is intended; see [Extrema peak-to-trough methods](../science/extrema_peak_to_trough.md) if using an extrema method. |
| Missing M-wave window | No window has a recognized M-wave name. | Add or rename the M-wave window if M-max normalization is intended. |
| Recording consistency warning | Recordings differ in sampling rate, channel count, stimulus delay, or stimulus ordering. | Verify import files and avoid treating incompatible recordings as interchangeable. |

## Dataset and experiment notices

| Notice | What it means | Suggested check |
| --- | --- | --- |
| Different latency-window sets | Child sessions or datasets do not use the same named, ordered windows. | Inspect the affected sessions. Standardize windows only if a shared analysis definition is intended. |
| Mixed sampling rates | Child sessions or datasets have different sampling rates. | Confirm that the comparison remains appropriate after filtering and windowing. |
| No active sessions or datasets | Nothing remains included at that level. | Review session or dataset exclusions. |
| One active session | A dataset has only one active session. | Interpret any aggregate result as a single-session result, not a replicate estimate. |
| High diversity of window names | Many window names occur only rarely within a dataset. | Check for spelling variants and unintended exploratory windows before aggregating. |

## M-wave naming

For M-max-related checks, MonStim recognizes the global **M-wave Recognition Names** without regard to case. The shipped list is `M-wave`, `M_wave`, `M wave`, `Mwave`, `M-response`, `M_response`, and `M response`, and you can replace it in **File > Settings Center > Global Analysis > Latency windows**. An empty list disables automatic M-wave recognition. Use a consistent name across comparable sessions to make plots and notices easier to interpret. See [M-max estimation and review](../science/mmax_estimation.md) for the complete setup guidance.

## Related topics

- [Troubleshooting](troubleshooting.md)
- [Latency windows](latency_windows.md)
- [Back to Help Library](index.md)
