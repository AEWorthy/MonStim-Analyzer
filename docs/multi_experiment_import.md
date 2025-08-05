# Multi-Experiment Import Feature

## Overview

The multi-experiment import feature allows users to import multiple experiments simultaneously from a root directory containing several experiment folders. This is particularly useful when you have organized your data with multiple experiments in separate directories and want to import them all at once.

## How It Works

### Directory Structure Expected

The feature expects the following directory structure:

```
Root Directory/
├── Experiment_1/
│   ├── Dataset_1/
│   │   ├── file1.csv
│   │   ├── file2.csv
│   │   └── ...
│   ├── Dataset_2/
│   │   ├── file1.csv
│   │   └── ...
│   └── ...
├── Experiment_2/
│   ├── Dataset_1/
│   │   ├── file1.csv
│   │   └── ...
│   └── ...
└── ...
```

Each experiment directory should contain dataset subdirectories, and each dataset subdirectory should contain CSV files.

### Features

- **Batch Processing**: Import multiple experiments simultaneously with parallel processing for faster import times
- **Conflict Detection**: Automatically detects existing experiments and allows you to choose whether to overwrite them
- **Progress Tracking**: Real-time progress updates showing which experiment is currently being processed
- **Validation**: Validates that each experiment contains valid dataset directories with CSV files before starting import
- **Selective Import**: Choose which experiments to import using an intuitive checkbox interface
- **Error Handling**: Continues processing other experiments even if one fails, with detailed error reporting

### Usage Steps

1. **Access the Feature**: Go to `File → Import Multiple Experiments` in the menu bar
2. **Select Root Directory**: Choose the directory that contains all your experiment folders
3. **Review Found Experiments**: The system will scan the directory and show you all valid experiments found, along with the number of datasets in each
4. **Select Experiments**: Use the checkbox interface to select which experiments you want to import. Use "Select All" or "Select None" buttons for convenience
5. **Handle Conflicts**: If any experiments already exist, you'll be prompted whether to overwrite them
6. **Monitor Progress**: A progress dialog will show the import status for all selected experiments, including:
   - Overall progress across all experiments
   - Current experiment being processed
   - Ability to cancel the operation
7. **Import Summary**: After completion, you'll see a summary showing how many experiments were successfully imported
