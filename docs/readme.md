# MonStim EMG Analyzer

## Overview
MonStim EMG Analyzer is a graphical user interface (GUI) application designed to facilitate the analysis of electromyography (EMG) data obtained from LabView MonStim experiments. The software allows users to import, process, visualize, and report on EMG data effectively.

## Installation
1. **Download** the MonStim EMG Analyzer zip file from GitHub: https://github.com/AEWorthy/MonStim_Analysis/releases
    - Windows Users: Download `MonStim-Analyzer-v0.4.0-WIN.zip`
    - Mac Users: A public Mac version is currently unavailable.
2. **Extract** the contents of the zip file to a location of your choice on your computer.
3. **Navigate** to the extracted folder and locate the `MonStim Analyzer v0.4.0.exe` file.
    - Note: Please keep all program files in the unzipped directory and work directly from there.

## Features
- **Data Import**: Import EMG data from CSV files. *You must first convert STM files to CSVs*.
- **Data Processing**: Automatically process and store EMG data.
- **Data Visualization**: Plot various types of hierarchical EMG data including raw EMG signals, reflex curves, M-max, max H-reflex, and average reflex curves.
- **Report Generation**: Generate and display detailed M-max reports, and session or dataset parameters.
- **Hierarchical Data Management**: Load and manage multiple EMG recording sessions as a dataset, and store multiple datasets as experiments. Exclude unwanted recordings/sessions/datasets at runtime and reload from the originals if desired.

## Running the Application
1. Double-click on `MonStim Analyzer v0.4.0.exe` to launch the application.
    - Note: Your system may aggresively warn you that this program may be a virus. It is an unsigned program because I am an individual developer.
    - Additional Note: The program may take a while to load, especially the first time you use it.
2. The MonStim EMG Analyzer main window will appear, ready for use.

## Usage
1. **Launch the Application**: Start the MonStim EMG Analyzer by running the application executable file.
2. **Import Data**: Use the "File" menu to import CSV data from your specified directory. Your data should be stored in a single folder with subfolders for each "dataset." A dataset contains all the individual recording CSV files for one or more recording sessions (e.g., a session could be a stimulation ramp to get an M-reflex curve).
3. **Select Dataset**: Choose a dataset from the available list.
4. **Select Session**: Select a session within the chosen dataset.
5. **Change Channel Names**: Optionally, customize the channel names via the settings dialog. This change is persistent even if you close the program.
6. **Generate Reports**: Access various reports through the "Reports" menu.
7. **Visualize Data**: Use dynamic plotting options to visualize different aspects of your EMG data.
8. **Export Raw Data**: Click the "Plot/Extract Raw Data" button in the plotting menu to easily export a plot's raw data to a CSV file.

# Quick Start Guide

1. **Import Data**
   - Click on "File" > "Import an Experiment" in the menu bar.
   - Select the directory containing the experiment directory. An experiment level directory should contain one or more dataset folders each containing one or more session folders with recording CSV files. 
   - ***See the `Important Information` section below for a guide on how to structure your data.*** In short, an "EMG Experiment" is a group of "EMG Datasets" that all have similar experimental parameters/conditions but are from different animals. Each "EMG Dataset" may contain multiple replicate "EMG Sessions" which are each stimulation ramps where each stimulus/response is an "EMG Recording".
   - Wait for the import process to complete. Processed data will be stored in the `/data` directory in the same folder as the `.EXE` file.

2. **Select Dataset and Session**
   - Use the dropdown menus in the "Data Selection" section to choose an experiment, a dataset, and a session of interest for plotting, report generation, etc.

3. **View Reports**
   - Click on the "Show M-max Report", "Show Session Report", or "Show Dataset Report" buttons to view detailed information about the selected data.

4. **Plot Data**
   - Choose whether to plot data from the selected session or from the entire dataset. *Note: Future versions will also allow analysis/plotting at the experiment level.*
   - Choose the desired plot type from the dropdown menu. (see the `Plot Types` section for more info).
   - Adjust any additional plot options.
   - Click the "Plot" button to generate the visualization.
        - Note: You can open more than one plot at a time. They will always open in new windows.

5. **Export Plotted Data**
    - Click the "Plot/Export Raw Data" button to simultaneously plot data and extract the plotted values.
    - In the pop-up window, press "Export Dataframe..." to easily export this data to a CSV file.
    - Note: The extracted data are an exact copy of the data plotted in the plotting frame.

6. **Customization Settings**
    - Use "Remove Session" in the "Data Selection" panel to remove a session from the dataset. This is useful if one of your recording sessions is too noisy or had other issues and you want to exclude it from the dataset.
        - Use "Undo"/"Redo" to undo/redo the removal/addition of sessions.
        - Alternatively use the "Edit" > "Reload Current Dataset" button to restore all sessions to the dataset. This will preserve any changes you made to the dataset such as channel names or reflex time windows.
    - Use "Edit" > "Change Channel Names" to modify channel labels.
    - Use "Edit" > "Manage Latency Windows" to create, remove, or edit latency windows for EMG measurements. Each window can have channel specific start times while the duration stays the same. Changes can be undone with the Undo action.
    - Use "Edit" > "Invert Channel Polarity" to select any desired channels in the currently selected dataset for which you would like to invert the polarity. This will be applied to all sessions in the dataset for each of the selected channels.
    - To view/exclude individual recordings, use the Single Session plot type called "Single EMG Recordings". Cycle through the individual recordings and exclude/include any that you desire. Use "Edit" > "Reload Current Session" to reset all changes.
    - For advanced users, set user preferences in "File" > "Preferences".

7. **Special Note**
    - Any changes made to your experiments (such as altering reflex window settings or channel names) are persistent (this data is saved in the ./data/bin folder). If you want to reset/reload experiments or restore them to default, you should use "Edit" > "Reload Current Experiment".

# Important Information and Tips

## Definitions for Common Terms

### EMG Data Types
- **EMG Experiment:** A set of EMG Datasets. These are biological replicates of the same protocol/condition across different animals.
- **EMG Dataset:** A set of EMG sessions. These are replicates of a single protocol/condition within a single animal.
- **EMG Session:** A set of EMG Recordings. All the recordings will share the same recording parameters like stimulus duration/recording duration; the only variable parameter should be the stimulus intensity. And individual EMG session should contain enough datapoints to create a Reflex Curve plot (see the `EMG Plot Types` section for details).
- **EMG Recording:** A single peri-stimulus recording at a defined stimulus intensity. Many recordings (often with varying stimulus intensity) will make up a session.

## How to structure your CSV data

- The algorithm for importing experiment data from CSV files makes certain assumptions about how your data is saved. You should store your data in the following structure:
    - Experiment 1/
        - Dataset 1/
            - XX001-1.CSV  # These are recordings files for Session 'XX001'
            - XX001-2.CSV
            - ...
            - XX002-1.CSV  # These are recordings files for Session 'XX002'
            - XX002-2.CSV
            - ...
        - Dataset 2/
            - session files...
        - other datasets...

- The experiment folder name (the directory you import under "File" > "Import an Experiment") is used to name the imported experiment, so make sure it is sufficiently descriptive.
- Dataset folder names should all have the following format: '[YYMMDD] [AnimalID] [Experimental condition or other desired info...]'.
    - You can leave `.STM` or any other non-`.CSV` filetypes in the datset folders. Any non-CSV files will be ignored.
- Session file names will be used to create a [`SessionID`]. It is best to leave them as the default names given by MonStim.


## EMG Plot Types

This EMG analysis program offers various plotting options at both the session and dataset levels. Below is an overview of the available plot types for each level.

### General Features

- All plots support customization of channel names and reflex window times (under the 'Edit' section of the menu bar).
- Various reflex amplitude calculation methods are available: RMS, average rectified, average unrectified, and peak-to-trough.
- Most plots offer the option to display results relative to M-max.

### Session-Level Plots

The `Single Session` view provides the following plot types:

1. **EMG Overlay Plot**
   - Function: `EMG`
   - Description: Plots EMG data for a specified time window, overlaying all recordings.
   - Options:
     - Display M-wave and H-reflex flags
     - Choose data type (filtered, raw, rectified_raw, or rectified_filtered)

2. **Suspected H-Reflex Plot**
   - Function: `Suspected H-reflexes`
   - Description: Detects and plots session recordings with potential H-reflexes.
   - Options:
     - Adjust H-reflex detection threshold
     - Choose reflex amplitude calculation method
     - Display figure legend

3. **Reflex Curves Plot**
   - Function: `Reflex Curves`
   - Description: Plots overlayed M-response and H-reflex curves for each recorded channel.
   - Options:
     - Choose reflex amplitude calculation method
     - Plot relative to M-max
     - Display legend

4. **M-max Plot**
   - Function: `M-Max`
   - Description: Plots the average M-response values at M-max for each channel.
   - Options:
     - Choose reflex amplitude calculation method

### Dataset-Level Plots

The `Entire Dtaset` view offers the following plot types:

1. **Reflex Curves Plot**
   - Function: `Average Reflex Curves`
   - Description: Plots the average M-response and H-reflex curves for each channel across all sessions in the dataset.
   - Options:
     - Choose reflex amplitude calculation method
     - Display legend
     - Plot relative to M-max

2. **Max H-Reflex Plot**
   - Function: `plot_maxH()`
   - Description: Plots the M-wave and H-response amplitudes at the stimulation voltage range where the average H-reflex is maximal.
   - Options:
     - Choose reflex amplitude calculation method
     - Plot relative to M-max



# Support
For additional help or feature requests, or to report issues and bugs, please contact aeworthy@emory.edu.

# License
© 2024 Andrew Worthy. All rights reserved.

---

Thank you for using the MonStim Analyzer!
