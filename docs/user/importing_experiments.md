# Importing experiments

## Purpose

Use **File > Import an Experiment** to import one experiment folder. Use **File > Import Multiple Experiments** when one parent folder contains several experiment folders. Import copies the source recordings into MonStim's managed data store; keep the original acquisition files as your archive.

Before arranging folders, read [Understanding the data hierarchy](data_hierarchy.md). It defines the recording, session, dataset, and experiment units that this layout represents.

This page covers native MonStim V3D/V3H CSV exports. For another acquisition system, do not rename files to resemble MonStim CSVs; use **File > Import using Add-on…** and follow [Importer add-ons](importer_addons.md). If no add-on recognizes your source, [request an importer](https://github.com/AEWorthy/MonStim-Analyzer/issues/new?template=importer_request.yml).

## Folder layout

For a single import, select the experiment folder:

```text
Experiment name/
  Dataset name/
    SessionA-1.csv
    SessionA-2.csv
    SessionB-1.csv
```

For a multi-experiment import, select the folder above the experiment folders:

```text
Study root/
  Experiment 1/
    Dataset 1/
      session files...
  Experiment 2/
    Dataset 1/
      session files...
```

Each dataset folder contains recording CSV files. Files with the same session identifier form a session; retain the acquisition-system naming convention when possible. Other file types in a dataset folder are ignored.

## Before you begin

- Use descriptive experiment and dataset folder names.
- Make each dataset folder one biological unit x one condition (for example, one animal under one treatment); place that unit's repeated sessions inside it.
- Verify that the CSV files use the expected MonStim export format and represent the intended channel mapping.
- Confirm that recordings grouped into one session belong to the same acquisition run and have compatible timing and channels.
- Keep each biological/experimental unit in the hierarchy you intend to analyze: experiment > dataset > session > recording. See [Understanding the data hierarchy](data_hierarchy.md) if you need to decide which level a measurement or condition belongs to.

## Steps and review

For multiple imports, choose the experiments to include and review any duplicate-name prompt carefully. The progress dialog can cancel work that has not yet completed; inspect the completion summary for any failures.

After import, select a session and inspect a raw and filtered trace before editing windows or exporting results. If a file is missing, grouped unexpectedly, or reports inconsistent acquisition settings, see [Diagnostic notices](diagnostic_notices.md) and [Troubleshooting](troubleshooting.md).

## Add-on import safety

Add-on imports normalize and validate incoming recordings in a temporary staging location. MonStim activates the completed experiment only after the whole import passes validation. Cancellation or failure does not modify the source files and does not replace an existing managed experiment.

[Back to Help Library](index.md)
