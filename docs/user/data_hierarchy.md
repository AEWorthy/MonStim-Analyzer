# Understanding the data hierarchy

## Purpose

MonStim organizes data as **experiment > dataset > session > recording**. The hierarchy distinguishes an individual acquired signal from repeated measurements, the biological unit under its condition, and an aggregate collection of those units. Choose the level that matches the question you want to ask before plotting, editing, or exporting.

```text
Experiment: usually one condition across biological units; also a flexible dataset container
  Dataset: one biological unit x one experimental condition
    Session: one repeated run of a protocol
      Recording: one acquired EMG signal
```

## Recording

A **recording** is one acquired EMG trace. In MonStim's managed store, it has the raw EMG data in a binary data file and per-recording metadata, including its identifiers, sampling and channel information, and stimulus information. Recording annotations store non-destructive, recording-specific decisions such as curation information; they do not replace the raw signal.

Recordings within a session can differ in stimulus intensity. For example, a recruitment series may contain one recording at each stimulus voltage.

## Session

A **session** is a set of recordings made with a common set of recording parameters and protocol context. Its recordings can use different stimulus intensities, but their acquisition timing, channels, sampling, and stimulus structure should be compatible enough to interpret together.

Sessions hold session-level analysis context, including latency windows, channel display/polarity choices, and recording inclusion or exclusion decisions. Multiple sessions let you repeat the same protocol, such as repeated runs from the same preparation or repeat measurements made under the same condition.

## Dataset

A **dataset** is MonStim's base unit of analysis: one biological unit under one experimental condition. In a typical animal experiment, it represents one **animal x condition** combination, and contains that animal's repeated session/protocol measurements. Its metadata identifies the biological unit and condition—for example, an animal identifier together with genotype, ablation status, drug administration, or another experimental manipulation. MonStim aggregates the dataset's included sessions when you use dataset-level plots or results.

Keep sessions together in a dataset only when they belong to the same biological unit under the same condition. An experiment will commonly contain several datasets from the same condition, one per animal or other biological unit. A dataset can have session-specific latency windows, so shared membership alone does not mean every analysis setting is identical.

## Experiment

An **experiment** usually groups datasets from the same condition across biological units. For example, an experiment can hold every animal that received one drug treatment, so its aggregate plots show a typical response for that condition.

An experiment can also be a general-purpose container for datasets. You may use it only to organize and curate data before exporting it to a separate statistical or custom-analysis pipeline. MonStim does not require experiment-level aggregation or comparison. When you do use experiment-level results, they aggregate the included datasets and sessions; experiment-wide edits are bulk actions on their children, not one shared session setting.

## Plan the hierarchy before import

For the built-in importer, this hierarchy is reflected in the source folder layout: select an experiment folder, place each dataset in its own child folder, and use the acquisition file names to group recordings into sessions. See [Importing experiments](importing_experiments.md) for the default MonStim V3D/V3H LabVIEW CSV layout and preparation checklist.

Other acquisition systems can require a different source layout or grouping rule because they encode recording, session, and stimulus information differently. A compatible [importer add-on](importer_addons.md) maps that system into MonStim's hierarchy; follow the add-on's own preparation instructions rather than assuming the LabVIEW layout applies.

## Related topics

- [Getting started](getting_started.md) — confirm the hierarchy after import.
- [Using MonStim Analyzer](using_monstim.md) — choose a level for plotting and editing.
- [Importing experiments](importing_experiments.md) — prepare default LabVIEW exports for import.
- [Importer add-ons](importer_addons.md) — import other acquisition systems safely.

[Back to Help Library](index.md)
