# MonStim EMG Analyzer

## Overview
MonStim EMG Analyzer is a graphical user interface (GUI) application designed to facilitate the analysis of electromyography (EMG) data obtained from LabView MonStim experiments. The software allows users to import, process, visualize, and report on EMG data effectively.

## Installation
1. **Download** the MonStim EMG Analyzer zip file from GitHub: https://github.com/AEWorthy/MonStim-Analyzer/releases
    - Windows Users: Download `MonStim-Analyzer-v0.4.3-WIN.zip`
    - Mac Users: A public Mac version is currently unavailable.
2. **Extract** the contents of the zip file to a location of your choice on your computer.
3. **Navigate** to the extracted folder and locate the `MonStim Analyzer v0.4.3.exe` file.
    - Note: Please keep all program files in the unzipped directory and work directly from there.

## Features
- **Data Import**: Import EMG data from CSV files. *You must first convert STM files to CSVs*. Supports both single experiment import and multi-experiment import.
- **Analysis Profiles**: Create, manage, and apply different analysis configurations for different experimental conditions.
- **Data Processing**: Automatically process and store EMG data with configurable parameters.
- **Data Visualization**: Plot various types of hierarchical EMG data including raw EMG signals, reflex curves, M-max, max H-reflex, and average reflex curves.
- **Report Generation**: Generate and display detailed M-max reports, and session or dataset parameters.
- **Hierarchical Data Management**: Load and manage multiple EMG recording sessions as a dataset, and store multiple datasets as experiments. Exclude unwanted recordings/sessions/datasets at runtime and reload from the originals if desired.
- **Recording Exclusion Editor**: Advanced tool for excluding recordings based on stimulus amplitude and other criteria, with support for bulk operations across sessions, datasets, or entire experiments.
- **Undo/Redo System**: Comprehensive undo/redo functionality for data modifications.
- **Responsive UI**: Automatically adapts to different screen sizes and DPI settings.

## Running the Application
1. Double-click on `MonStim Analyzer v0.4.3.exe` to launch the application.
    - Note: Your system may aggressively warn you that this program may be a virus. It is an unsigned program because I am an individual developer.
    - Additional Note: The program may take a while to load, especially the first time you use it.
2. The MonStim EMG Analyzer main window will appear, ready for use.

## Usage
1. **Launch the Application**: Start the MonStim EMG Analyzer by running the application executable file.
2. **Import Data**: Use the "File" menu to import CSV data from your specified directory. You can import single experiments or multiple experiments at once.
3. **Analysis Profiles**: Select or create analysis profiles to apply specific configurations for different experimental conditions.
4. **Select Data**: Choose an experiment, dataset, and session from the available dropdowns in the Data Selection panel.
5. **Customize Settings**: Modify channel names, latency windows, and other parameters via the Edit menu or right-click context menus.
6. **Generate Reports**: Access various reports through the Reports panel or by right-clicking on data elements.
7. **Visualize Data**: Use dynamic plotting options to visualize different aspects of your EMG data.
8. **Export Raw Data**: Click the "Plot & Extract Data" button to export plot data to CSV files.
9. **Undo/Redo**: Use Ctrl+Z/Ctrl+Y or the Edit menu to undo/redo modifications.

# Quick Start Guide

1. **Import Data**
   - Click on "File" > "Import an Experiment" for single experiment import, or "File" > "Import Multiple Experiments" for batch import.
   - For single import: Select the directory containing the experiment. For multi-import: Select the root directory containing multiple experiment folders.
   - ***See the `Important Information` section below for a guide on how to structure your data.*** In short, an "EMG Experiment" is a group of "EMG Datasets" that all have similar experimental parameters/conditions but are from different animals. Each "EMG Dataset" may contain multiple replicate "EMG Sessions" which are each stimulation ramps where each stimulus/response is an "EMG Recording".
   - Wait for the import process to complete. Processed data will be stored in the `/data` directory in the same folder as the `.EXE` file.

2. **Select Analysis Profile and Data**
   - Choose an analysis profile from the dropdown at the top of the left panel, or use "(default)" for standard settings.
   - Use the dropdown menus in the "Data Selection" section to choose an experiment, a dataset, and a session of interest for plotting, report generation, etc.

3. **View Reports**
   - Click on the "Show M-max Report", "Show Session Report", or "Show Dataset Report" buttons to view detailed information about the selected data.

4. **Plot Data**
   - Choose whether to plot data from the selected session, from the entire dataset, or from the entire experiment.
   - Choose the desired plot type from the dropdown menu. (see the `Plot Types` section for more info).
   - Adjust any additional plot options.
   - Click the "Plot" button to generate the visualization.
        - Note: You can open more than one plot at a time. They will always open in new windows.

5. **Export Plotted Data**
    - Click the "Plot & Extract Data" button to simultaneously plot data and extract the plotted values.
    - In the pop-up window, press "Export Dataframe..." to easily export this data to a CSV file.
    - Note: The extracted data are an exact copy of the data plotted in the plotting frame.

6. **Preferences and Settings**
    - Use "File" > "Preferences" to configure analysis parameters and create custom analysis profiles.
    - Use "File" > "Display Preferences" to adjust UI scaling and display settings.
    - Use "File" > "Program Preferences" to control application behavior and data tracking preferences.
    - Access context menus by right-clicking on experiments, datasets, or sessions for quick actions like marking complete/incomplete, excluding, or restoring items.

7. **Data Management and Editing**
    - Use Undo (Ctrl+Z) and Redo (Ctrl+Y) to undo/redo data modifications.
    - Right-click on datasets or sessions in the dropdown menus to access context-specific actions.
    - Use "Edit" > "Experiment/Dataset/Session" submenus to manage latency windows, change channel names, invert polarity, exclude/restore items, or reload data.
    - To exclude individual recordings, use the Single Session plot type called "Single EMG Recordings" and exclude unwanted recordings directly from the plot interface.
    - For bulk recording exclusion based on criteria like stimulus amplitude, use "Edit" > "Data Curation" > "Recording Exclusion Editor..." to exclude multiple recordings at once across sessions, datasets, or entire experiments.

8. **Special Notes**
    - Changes made to experiments (channel names, latency windows, exclusions) are automatically saved.
    - Use "Edit" > "Reload Current Experiment/Dataset/Session" to reset changes and restore from the original, raw data.
    - The application remembers your last session state, analysis profile, and selected data when you restart by default. This can be changed in Program Preferences.

# Important Information and Tips

## Definitions for Common Terms

### EMG Data Types
- **EMG Experiment:** A set of EMG Datasets. These are biological replicates of the same protocol/condition across different animals.
- **EMG Dataset:** A set of EMG sessions. These are replicates of a single protocol/condition within a single animal.
- **EMG Session:** A set of EMG Recordings. All the recordings will share the same recording parameters like stimulus duration/recording duration; the only variable parameter should be the stimulus intensity. And individual EMG session should contain enough datapoints to create a Reflex Curve plot (see the `EMG Plot Types` section for details).
- **EMG Recording:** A single peri-stimulus recording at a defined stimulus intensity. Many recordings (often with varying stimulus intensity) will make up a session.

## How to structure your CSV data

- The algorithm for importing experiment data from CSV files makes certain assumptions about how your data is saved. You should store your data in the following structure:
    - **For single experiment import:**
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
    
    - **For multi-experiment import:**
        - Root Directory/
            - Experiment 1/
                - Dataset 1/
                    - XX001-1.CSV
                    - XX001-2.CSV
                    - ...
                - Dataset 2/
                    - session files...
            - Experiment 2/
                - Dataset 1/
                    - session files...
            - other experiments...

- The experiment folder name (the directory you import under "File" > "Import an Experiment") is used to name the imported experiment, so make sure it is sufficiently descriptive.
- Dataset folder names should all have the following format: '[YYMMDD] [AnimalID] [Experimental condition or other desired info...]'.
    - You can leave `.STM` or any other non-`.CSV` filetypes in the datset folders. Any non-CSV files will be ignored.
- Session file names will be used to create a [`SessionID`]. It is best to leave them as the default names given by MonStim.


## EMG Plot Types

This EMG analysis program offers various plotting options at session, dataset, and experiment levels. Below is a comprehensive overview of all available plot types.

### General Features

- All plots support customization of channel names and reflex window times (under the 'Edit' section of the menu bar).
- Various reflex amplitude calculation methods are available: RMS, average rectified, average unrectified, and peak-to-trough.
- Most plots offer the option to display results relative to M-max.
- Interactive cursor support for precise measurements (PyQtGraph plots).
- Automatic data export functionality - plotted data can be saved to CSV files that can be imported into MonStim Plotter (https://github.com/AEWorthy/MonStim-Plotter/) for publication-ready figure generation.

### Session-Level Plots

The `Single Session` view provides the following plot types:

1. **EMG Overlay Plot**
   - Function: `EMG`
   - Description: Plots EMG data for a specified time window, overlaying all recordings from a session.
   - Options:
     - Choose data type (filtered, raw, rectified_raw, or rectified_filtered)
     - Display reflex latency window flags
     - Show/hide plot legend
     - Enable colormap display showing stimulus voltage gradients
     - Filter specific stimuli to plot

2. **Single EMG Recordings**
   - Function: `Single EMG Recordings`
   - Description: Plots EMG data from a single recording at a specific stimulus intensity.
   - Options:
     - Select recording index (stimulus intensity)
     - Choose data type (filtered, raw, rectified_raw, or rectified_filtered)
     - Fixed or auto-scaling Y-axis
     - Display latency window flags
     - Show/hide plot legend
     - Enable interactive cursor for measurements

3. **Suspected H-Reflex Plot**
   - Function: `Suspected H-reflexes`
   - Description: Detects and plots session recordings with potential H-reflexes based on amplitude threshold.
   - Options:
     - Adjust H-reflex detection threshold (default: 0.3 mV)
     - Choose reflex amplitude calculation method
     - Display latency window flags
     - Show/hide figure legend

4. **Reflex Curves Plot**
   - Function: `Reflex:Stimulus Curves`
   - Description: Plots M-response and H-reflex amplitudes vs. stimulus voltage for each recorded channel.
   - Options:
     - Choose reflex amplitude calculation method
     - Plot relative to M-max
     - Display legend
     - Manual M-max override
     - Interactive cursor for precise measurements

5. **M-max Plot**
   - Function: `M-Max`
   - Description: Plots the average M-response values at M-max for each channel.
   - Options:
     - Choose reflex amplitude calculation method

6. **Reflex Averages**
   - Function: `Reflex Averages`
   - Description: Plots average reflex amplitudes across the session for different latency windows.
   - Options:
     - Choose reflex amplitude calculation method
     - Display legend
     - Plot relative to M-max

### Dataset-Level Plots

The `Entire Dataset` view offers the following plot types:

1. **Average Reflex Curves**
   - Function: `Average Reflex:Stimulus Curves`
   - Description: Plots the average M-response and H-reflex curves for each channel across all sessions in the dataset, with standard error bars.
   - Options:
     - Choose reflex amplitude calculation method
     - Display legend and error bars
     - Plot relative to M-max
     - Manual M-max override

2. **Max H-Reflex Plot**
   - Function: `Max H-reflex`
   - Description: Plots the M-wave and H-response amplitudes at the stimulation voltage where the average H-reflex is maximal.
   - Options:
     - Choose reflex amplitude calculation method
     - Plot relative to M-max
     - Manual M-max override

3. **M-max Plot**
   - Function: `M-max`
   - Description: Plots dataset-averaged M-max values for each channel.
   - Options:
     - Choose reflex amplitude calculation method

### Experiment-Level Plots

The `Experiment` view provides aggregated analysis across all datasets in an experiment:

1. **Average Reflex Curves**
   - Function: `Average Reflex:Stimulus Curves`
   - Description: Plots experiment-wide averages of M-response and H-reflex curves across all datasets and sessions, with standard error bars representing biological variation.
   - Options:
     - Choose reflex amplitude calculation method
     - Display legend and error bars
     - Plot relative to M-max
     - Manual M-max override

2. **Max H-Reflex Plot**
   - Function: `Max H-reflex`
   - Description: Plots experiment-wide M-wave and H-response amplitudes at optimal stimulation voltage.
   - Options:
     - Choose reflex amplitude calculation method
     - Plot relative to M-max
     - Manual M-max override

3. **M-max Plot**
   - Function: `M-max`
   - Description: Plots experiment-wide M-max averages across all datasets.
   - Options:
     - Choose reflex amplitude calculation method

## User Settings and Preferences

MonStim Analyzer provides comprehensive customization options through multiple preference dialogs accessible from the File menu.

### Analysis Profiles (File > Preferences)

Analysis profiles allow you to save and reuse specific analysis configurations for different experimental conditions. Each profile can contain:

- **Analysis Parameters**: Custom settings for reflex calculations, filtering, and processing
- **Latency Window Presets**: Predefined time windows for M-wave and H-reflex measurements
- **Channel Configurations**: Default channel names and settings
- **Plot Preferences**: Default plotting options and visualization settings
- **Stimulus Selection**: Specify which stimulus voltages to include in plots

#### Managing Analysis Profiles

- Access profile management through "File" > "Preferences"
- **Global Configuration**: Select "(default)" to modify system-wide settings
- **Profile Management**: Create, edit, duplicate, and delete custom profiles
- **Profile Features**:
  - Custom profile names and descriptions
  - Analysis parameter overrides (method, thresholds, colors, etc.)
  - Stimulus filtering for focused analysis
  - Latency window configurations
- The application remembers your last selected profile between sessions

#### Available Settings in Profiles

**Basic Plotting Parameters:**
- Bin size for stimulus voltage grouping
- Time window and pre-stimulus time settings
- Default amplitude calculation method
- Default channel names

**EMG Filter Settings:**
- Butterworth filter parameters
- Filter order and frequency ranges

**M-max Calculation Settings:**
- Detection window sizes
- Threshold parameters
- Plateau detection settings

**Plot Style Settings:**
- Font sizes for titles, axis labels, and ticks
- M-wave and H-reflex colors
- Latency window visual styles
- Subplot adjustment parameters

### Display Preferences (File > Display Preferences)

Controls UI scaling and visual presentation:

#### UI Scaling Settings
- **Auto Scale**: Automatically detect and scale UI based on screen DPI and resolution
- **Manual Scale Factor**: Set custom scaling (0.5x to 3.0x) when auto-scale is disabled
- **Current Scale Display**: Shows active scaling factor

#### Panel and Layout Settings
- **Left Panel Width**: Adjust control panel width as percentage of screen (15-40%)
- **Window Positioning**: Automatic centering and size optimization
- **Responsive Design**: Automatic adaptation to different screen sizes and DPI settings

#### Font and Sizing Controls
- **Base Font Size**: Set application font size (6-16 pt)
- **Maximum Font Scale**: Limit font scaling for readability (1.0x-2.0x)
- **Dynamic Scaling**: Proportional scaling of spacing, margins, and UI elements

#### High DPI Support
- **DPI Awareness**: Automatic detection of high DPI displays
- **Multi-Monitor Support**: Consistent scaling across different monitor setups
- **4K/5K Display Optimization**: Resolution-based scaling for ultra-high resolution displays

### Program Preferences (File > Program Preferences)

Controls application behavior and data tracking:

#### Session Restoration
- **Track Session State**: Remember last opened experiment, dataset, and session
- **Automatic Restoration**: Restore previous session on application startup
- **Profile Memory**: Remember last selected analysis profile

#### File and Path Tracking
- **Import/Export Paths**: Remember last used directories for data import/export
- **Recent Files**: Track recently opened experiments for quick access
- **Path Restoration**: Automatically navigate to previously used locations

#### Privacy and Data Options
- **Selective Tracking**: Enable/disable specific tracking features independently
- **Data Persistence**: Control what information is saved between sessions
- **Reset Options**: Clear saved preferences and restore defaults

### UI Responsiveness Features

The application includes comprehensive responsive design features:

#### Automatic Scaling
- **DPI Detection**: Automatic detection of screen DPI and scaling factors
- **Resolution Adaptation**: Intelligent sizing for different screen resolutions
- **Component Scaling**: Proportional scaling of fonts, spacing, and UI elements

#### Responsive Widgets
- **Smart Combo Boxes**: Dynamic popup sizing and tooltips for long text
- **Scroll Areas**: Automatic scroll bars for content that exceeds available space
- **Collapsible Sections**: Space-saving expandable/collapsible groups

#### Layout Optimization
- **Flexible Panel Sizing**: Panels adapt to screen size with min/max constraints
- **Window State Management**: Automatic saving/restoration of window positions and sizes
- **Content Adaptation**: UI elements adjust to available space intelligently

### Settings Persistence

All user settings are automatically saved and persist between application sessions:

- **QSettings Integration**: Uses system-appropriate settings storage (Windows Registry, macOS Preferences, Linux config files)
- **Automatic Backup**: Settings are synced and backed up automatically
- **Import/Export**: Settings can be reset to defaults through preference dialogs
- **Session Memory**: Application remembers window positions, panel sizes, and user selections

### Keyboard Shortcuts and Accessibility

- **Undo/Redo**: Ctrl+Z/Ctrl+Y for data modifications
- **Menu Access**: Standard keyboard navigation through all menus
- **Dialog Navigation**: Tab navigation through all preference dialogs
- **Tooltips**: Comprehensive tooltips explaining all settings and options


## Log Files
Application errors are written to a log file named `app.log` in your user
application data directory. On Windows this is usually:
```
%APPDATA%\WorthyLab\MonStimAnalyzer\logs
```
Use **Help > Open Log Folder** to quickly open this location. You can also
generate a zip archive of these logs using **Help > Save Error Report** when
requesting support.

# Support
For additional help or feature requests, or to report issues and bugs, please:
- Visit the GitHub repository: https://github.com/AEWorthy/MonStim-Analyzer
- Open an issue on GitHub for bug reports or feature requests
- Contact aeworthy@emory.edu for direct support

Use **Help > Open Log Folder** to access application logs, or **Help > Save Error Report** to generate a support package when reporting issues.

# License
This software is licensed under the **BSD 2-Clause License** ("Simplified BSD License").

© 2024 Andrew Worthy. All rights reserved.

For full license terms, see the `license.txt` file included with the software.

---

Thank you for using the MonStim Analyzer!
