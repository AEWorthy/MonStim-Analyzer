# MonStim Analyzer - Quick Start Guide

Get up and running with MonStim EMG Analyzer in minutes!

## 📥 Installation

1. **Download** the latest release from [GitHub Releases](https://github.com/AEWorthy/MonStim-Analyzer/releases)
   - Windows: `MonStim-Analyzer-v0.4.1-WIN.zip`
   - Mac: Currently unavailable

2. **Extract** the zip file to your desired location

3. **Run** `MonStim Analyzer v0.4.1.exe` from the extracted folder
   - ⚠️ Keep all files in the extracted directory
   - ⚠️ First launch may take longer and trigger security warnings (unsigned software)

## 🚀 Quick Start (5 Steps)

### 1. Import Your Data
- **File** → **Import an Experiment** (single) or **Import Multiple Experiments** (batch)
- Select your experiment directory
- Wait for processing to complete

### 2. Structure Your CSV Data
```
Experiment Folder/
├── Dataset 1/
│   ├── XX001-1.CSV  # Session XX001 recordings
│   ├── XX001-2.CSV
│   └── XX002-1.CSV  # Session XX002 recordings
└── Dataset 2/
    └── session files...
```
- Dataset names: `[YYMMDD] [AnimalID] [Condition]`
- Only CSV files are imported (STM files ignored)

### 3. Select Your Data
- Choose **Analysis Profile** (or use "default")
- Select **Experiment** → **Dataset** → **Session**

### 4. Generate Reports
Click any report button:
- **Show M-max Report**
- **Show Session Report** 
- **Show Dataset Report**

### 5. Plot Your Data
- Choose plot scope: **Single Session** / **Entire Dataset** / **Experiment**
- Select **Plot Type** from dropdown
- Click **Plot** button
- Use **Plot & Extract Data** to export plot data to CSV

## 📊 Key Plot Types

| Level | Plot Type | Description |
|-------|-----------|-------------|
| **Session** | EMG Overlay | All recordings overlaid |
| | Reflex Curves | M-response & H-reflex vs stimulus |
| | Single EMG Recordings | Individual stimulus responses |
| **Dataset** | Average Reflex Curves | Averaged across sessions |
| | Max H-Reflex | Peak H-reflex response |
| **Experiment** | Average Reflex Curves | Biological replicates averaged |

## ⚙️ Essential Settings

### Analysis Profiles
- **File** → **Preferences** to create custom analysis configurations
- Save different settings for different experimental conditions

### Display Scaling
- **File** → **Display Preferences** for UI scaling
- Auto-detect or manual scaling (0.5x - 3.0x)

### Data Management
- **Right-click** experiments/datasets/sessions for context menus
- **Ctrl+Z/Ctrl+Y** for undo/redo
- **Edit** menu for channel names, latency windows, exclusions

## 🔧 Data Definitions

- **EMG Experiment**: Multiple datasets (biological replicates)
- **EMG Dataset**: Multiple sessions from one animal
- **EMG Session**: Multiple recordings at different stimulus intensities
- **EMG Recording**: Single stimulus-response pair

## 📁 File Locations

- **Processed Data**: `/data` folder (same directory as .exe)
- **Logs**: `%APPDATA%\WorthyLab\MonStimAnalyzer\logs` (Windows)
- Use **Help** → **Open Log Folder** for quick access

## 🆘 Need Help?

- 📖 **Full Documentation**: [README.md](README.md)
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/AEWorthy/MonStim-Analyzer/issues)
- 📧 **Direct Support**: aeworthy@emory.edu
- 📋 **Error Reports**: **Help** → **Save Error Report**

## 🔗 Related Tools

- **[MonStim Plotter](https://github.com/AEWorthy/MonStim-Plotter/)**: Publication-ready figures from exported data

---

**License**: BSD 2-Clause | **© 2024 Andrew Worthy**
