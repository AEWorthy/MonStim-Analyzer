# MonStim EMG Analyzer

## Overview
MonStim EMG Analyzer is a graphical user interface (GUI) application designed to facilitate the analysis of electromyography (EMG) data obtained from LabView MonStim experiments. The software allows users to import, process, visualize, and report on EMG data effectively.

## Features
- **Data Import**: Import EMG data from CSV files.
- **Data Processing**: Automatically process and store EMG data.
- **Data Visualization**: Plot various types of EMG data including raw EMG signals, suspected H-reflexes, reflex curves, M-max, max H-reflex, and average reflex curves.
- **Session Management**: Load and manage multiple EMG data sessions.
- **Report Generation**: Generate and display detailed reports on session parameters, M-max, and dataset parameters.
- **Customizable Channel Names**: Change and save custom channel names for datasets and sessions.

## Installation
1. **Install Python 3.8+**: Ensure you have Python 3.8 or higher installed.
2. **Install Dependencies**: Run the following command to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**: Execute the following command to start the application:
   ```bash
   python main.py
   ```

## Usage
1. **Launch the Application**: Start the MonStim EMG Analyzer by running the application script.
2. **Import Data**: Use the "File" menu to import CSV data from your specified directory.
3. **Select Dataset**: Choose a dataset from the available list.
4. **Select Session**: Select a session within the chosen dataset.
5. **Change Channel Names**: Optionally, customize the channel names via the settings dialog.
6. **Generate Reports**: Access various reports through the "Reports" menu.
7. **Visualize Data**: Use the plotting options to visualize different aspects of your EMG data.

## GUI Components
### SplashScreen
Displays a splash screen with program information, logo, version, and description.

### Main Window (EMGAnalysisGUI)
The main window consists of several key widgets:
- **Menu Bar**: Provides options to import data, change settings, and access reports.
- **Data Selection Widget**: Allows users to select and manage datasets and sessions.
- **Reports Widget**: Displays available reports for the selected dataset/session.
- **Plot Widget**: Provides options to plot different types of EMG data.

### Commands
- **Undo/Redo**: Available through the command invoker for reverting or reapplying actions.
- **Remove Session**: Command to remove a session from the dataset.

### Data Processing
Handles the loading and processing of existing datasets and importing new data from CSV files. Uses multithreading to process data efficiently.

## License
© 2024 Andrew Worthy

## Support
For issues and feature requests, please contact aeworthy@emory.edu.
