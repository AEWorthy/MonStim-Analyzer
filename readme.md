# MonStim EMG Analyzer

## Overview
MonStim EMG Analyzer is a graphical user interface (GUI) application designed to facilitate the analysis of electromyography (EMG) data obtained from LabView MonStim experiments. The software allows users to import, process, visualize, and report on EMG data effectively.

## Features
- **Data Import**: Import EMG data from CSV files. You must first convert .STM files to CSVs.
- **Data Processing**: Automatically process and store EMG data. Changes are persistent and saved in the ./data/bin folder.
- **Data Visualization**: Plot various types of EMG data including raw EMG signals, suspected H-reflexes, reflex curves, M-max, max H-reflex, and average reflex curves.
- **Session Management**: Load and manage multiple EMG data sessions under datasets. Remove unwanted session at runtime and reload if necessary.
- **Report Generation**: Generate and display detailed reports on  M-max, session parameters, and dataset parameters.
- **Customizable Channel Names**: Change and save custom channel names for datasets and sessions.

## Usage
1. **Launch the Application**: Start the MonStim EMG Analyzer by running the application executable file.
2. **Import Data**: Use the "File" menu to import CSV data from your specified directory. Your data should be stored in a single folder with subfolders for each "dataset." A dataset contains all the individual recording CSV files for one or more recording sessions (e.g., a session could be a stimulation ramp to get an M-reflex curve).
3. **Select Dataset**: Choose a dataset from the available list.
4. **Select Session**: Select a session within the chosen dataset.
5. **Change Channel Names**: Optionally, customize the channel names via the settings dialog. This change is persistent even if you close the program.
6. **Generate Reports**: Access various reports through the "Reports" menu.
7. **Visualize Data**: Use the plotting options to visualize different aspects of your EMG data.

## GUI Components

### Main Window (EMGAnalysisGUI)
The main window consists of several key widgets:
- **Menu Bar**: Provides options to import data, change settings, and access reports.
- **Data Selection Widget**: Allows users to select and manage datasets and sessions.
- **Reports Widget**: Displays available reports for the selected dataset/session.
- **Plot Widget**: Provides options to plot different types of EMG data.

### Menu Bar Commands
- **Undo/Redo**: Available through the command invoker for reverting or reapplying actions.
- **Remove Session**: Command to remove a session from the dataset.

### Data Selection Options
- **Undo/Redo**: Available through the command invoker for reverting or reapplying actions.
- **Remove Session**: Command to remove a session from the dataset.

### Reports
- **Undo/Redo**: Available through the command invoker for reverting or reapplying actions.
- **Remove Session**: Command to remove a session from the dataset.

### Plot Options
- **Undo/Redo**: Available through the command invoker for reverting or reapplying actions.
- **Remove Session**: Command to remove a session from the dataset.

## License
© 2024 Andrew Worthy

## Support
For issues and feature requests, please contact aeworthy@emory.edu.
