# Importing experiments

## Purpose

Use **File > Import an Experiment** to import one experiment folder. Use **File > Import Multiple Experiments** when one parent folder contains several experiment folders. Import copies the source recordings into MonStim's managed data store; keep the original acquisition files as your archive.

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
- Verify that the CSV files use the expected MonStim export format and represent the intended channel mapping.
- Confirm that recordings grouped into one session belong to the same acquisition run and have compatible timing and channels.
- Keep each biological/experimental unit in the hierarchy you intend to analyze: experiment > dataset > session > recording.

## Steps and review

For multiple imports, choose the experiments to include and review any duplicate-name prompt carefully. The progress dialog can cancel work that has not yet completed; inspect the completion summary for any failures.

After import, select a session and inspect a raw and filtered trace before editing windows or exporting results. If a file is missing, grouped unexpectedly, or reports inconsistent acquisition settings, see [Diagnostic notices](diagnostic_notices.md) and [Troubleshooting](troubleshooting.md).

[Back to Help Library](index.md)
