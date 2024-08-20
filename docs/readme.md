# MonStim EMG Analyzer

## Overview
MonStim EMG Analyzer is a graphical user interface (GUI) application designed to facilitate the analysis of electromyography (EMG) data obtained from LabView MonStim experiments. The software allows users to import, process, visualize, and report on EMG data effectively.

## Features
- **Data Import**: Import EMG data from CSV files. *You must first convert .STM files to CSVs*.
- **Data Processing**: Automatically process and store EMG data.
- **Data Visualization**: Plot various types of EMG data including raw EMG signals, suspected H-reflexes, reflex curves, M-max, max H-reflex, and average reflex curves.
- **Session Management**: Load and manage multiple EMG data sessions under datasets. Remove unwanted session at runtime and reload if necessary.
- **Report Generation**: Generate and display detailed reports on  M-max, session parameters, and dataset parameters.
- **Customizable Channel Names**: Change and save custom channel names for datasets and sessions.

## Installation
1. **Download** the MonStim EMG Analyzer zip file from the provided source: https://github.com/AEWorthy/MonStim_Analysis/releases/tag/Releases
    - Windows Users: Download `MonStim-Analyzer-v1.x-WIN.zip`
    - Mac Users: Dowload `MonStim-Analyzer-v1.x-OSX.zip`
2. **Extract** the contents of the zip file to a location of your choice on your computer.
3. **Navigate** to the extracted folder and locate the `MonStim Analyzer v1.X.exe` file.
    - Note: Please keep all program files in the unzipped directory and work from there.

## Running the Application
1. Double-click on `MonStim Analyzer v1.X.exe` to launch the application.
    - Note: If you 
2. The MonStim EMG Analyzer main window will appear, ready for use.

## Usage
1. **Launch the Application**: Start the MonStim EMG Analyzer by running the application executable file.
2. **Import Data**: Use the "File" menu to import CSV data from your specified directory. Your data should be stored in a single folder with subfolders for each "dataset." A dataset contains all the individual recording CSV files for one or more recording sessions (e.g., a session could be a stimulation ramp to get an M-reflex curve).
3. **Select Dataset**: Choose a dataset from the available list.
4. **Select Session**: Select a session within the chosen dataset.
5. **Change Channel Names**: Optionally, customize the channel names via the settings dialog. This change is persistent even if you close the program.
6. **Generate Reports**: Access various reports through the "Reports" menu.
7. **Visualize Data**: Use the plotting options to visualize different aspects of your EMG data.

# Quick Start Guide

1. **Import Data**: 
   - Click on "File" > "Import CSV Data" in the menu bar.
   - Select the directory containing your directory containing dataset folders with CSV files. ***See the `Important Information` section below for a guide on how to structure your data.***
   - Wait for the import process to complete. Processed data will be stored in the `/data` directory in the same folder as the `.EXE` file.

2. **Select Dataset and Session**:
   - Use the dropdown menus in the "Data Selection" section to choose a dataset and session of interest.

3. **View Reports**:
   - Click on the "Show M-max Report", "Show Session Report", or "Show Dataset Report" buttons to view detailed information about the selected data.

4. **Plot Data**:
   - Choose whether to plot data from the selected session or from the entire dataset.
   - Choose the desired plot type from the dropdown menu. (see the `Plot Types` section for more info).
   - Adjust any additional plot options.
   - Click the "Plot" button to generate the visualization.
        - Note: You can open more than one plot at a time. They will always open in new windows.

5. **Customization Settings**:
    - Use "Remove Session" in the "Data Selection" panel to remove a session from the dataset. This is useful if one of your recording sessions is too noisy or had other issues and you want to exclude it from the dataset.
        - Use "Undo"/"Redo" to undo/redo the removal/addition of sessions.
        - Alternatively use the "Reload All Sessions" button to restore all sessions to the dataset. This will preserve any changes you made to the dataset such as channel names or reflex time windows.
    - Use "Tools" > "Change Channel Names" to modify channel labels.
    - Use "Tools" > "Update Reflex Settings" to adjust reflex window parameters.

# Important Information and Tips

## Definitions for Common Terms

### EMG Data Types
- **EMG Recording:** A single peri-stimulus recording at a defined stimulus value. Many recordings (sometimes with varying stimulus intensity) will make up a session.
- **EMG Session:** A set of EMG Recordings. All the recordings will share the same recording parameters like stimulus/recording duration; the only variable parameter should be the stimulus intensity. And EMG session may contain enough datapoints to create a Reflex Curve plot (see the `EMG Plot Types` section for details).
- **EMG Dataset:** A set of EMG sessions. These are replicates of the same protocol/condition within a single animal.
- **EMG Experiment:** *(NOT YET IMPLEMENTED)* A set of EMG Datasets. These are replicates of the same protocol/condition on different animal.

## How to structure your CSV data

- Importing data from CSV files makes certain assumptions about how your data is saved. You should store your data in the following structure:
    - my_project/
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

- The project folder name is not used at all by the program. Later versions may use it to create an `Experiment` data type.
- Dataset folder names should all have the following format: '[YYMMDD] [AnimalID] [Experimental Condition]'.
    - You can leave `.STM` or any other non-`.CSV` filetypes in the datset folders.
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


## Other Tips

- Changes made to dataset (such as altering reflex window settings or channel names) are persistent and saved in the ./data/bin folder. If you want to reset/reload datasets or restore any to default, you should close the program, delete the appropriate '.pickle' file, and re-open the program.



# Support
For additional help or feature requests, or to report issues and bugs, please contact aeworthy@emory.edu.

# License
© 2024 Andrew Worthy. All rights reserved.

---

Thank you for using the MonStim Analyzer!
