# Recording Exclusion Editor

The Recording Exclusion Editor is a powerful tool for excluding recordings from analysis based on various criteria. It provides a user-friendly interface for bulk recording management with support for different exclusion criteria.

## Features

### Current Capabilities
- **Stimulus Amplitude Exclusion**: Exclude recordings based on primary stimulus voltage
  - Above threshold: Exclude recordings with stimulus above a specified value
  - Below threshold: Exclude recordings with stimulus below a specified value  
  - Outside range: Exclude recordings outside a voltage range
  - Inside range: Exclude recordings within a voltage range

### Application Levels
The exclusion criteria can be applied at three different levels:
- **Current Session Only**: Apply to the currently selected session
- **Entire Dataset**: Apply to all sessions in the current dataset
- **Entire Experiment**: Apply to all sessions across all datasets in the experiment

### Preview and Confirmation
- **Real-time Preview**: See which recordings will be affected before applying changes
- **Summary Statistics**: View total recordings, currently excluded, and recordings that will be excluded
- **Confirmation Dialog**: Confirm changes before applying them
- **Undo Support**: All changes support undo/redo through the command pattern

## Usage

### Accessing the Editor
1. Launch MonStim Analyzer
2. Load an experiment with data
3. Select a session
4. Go to **Edit** > **Data Curation** > **Recording Exclusion Editor...**

### Using Stimulus Amplitude Exclusion
1. In the dialog, click the **Stimulus Amplitude** tab
2. Check **"Exclude recordings by stimulus amplitude"** to enable the criteria
3. Choose the exclusion type from the dropdown:
   - **Above threshold**: Excludes recordings with stimulus > threshold
   - **Below threshold**: Excludes recordings with stimulus < threshold  
   - **Outside range**: Excludes recordings outside the range [lower, upper]
   - **Inside range**: Excludes recordings inside the range [lower, upper]
4. Set the threshold value(s) using the spinbox controls
5. Choose the application level (Session, Dataset, or Experiment)
6. Click **Preview** to see which recordings will be affected
7. Click **Apply** to execute the exclusions

### Preview Table
The preview table shows:
- **Recording ID**: Unique identifier for each recording
- **Session**: Which session the recording belongs to
- **Stimulus (V)**: The primary stimulus voltage for the recording
- **Status**: Current status (Included, Excluded, or Will exclude)

Records that will be excluded are highlighted in gray.

## Extensibility

The Recording Exclusion Editor is designed to be easily extensible for future criteria types:

### Adding New Criteria Types

1. **Add a new tab** in `create_criteria_widget()`:
```python
# Add new criteria tab
quality_tab = self.create_quality_criteria_tab()
self.criteria_tabs.addTab(quality_tab, "Recording Quality")
```

2. **Create the criteria tab method**:
```python
def create_quality_criteria_tab(self) -> QWidget:
    tab_widget = QWidget()
    layout = QVBoxLayout(tab_widget)
    
    # Add your criteria controls here
    self.quality_group = QGroupBox("Exclude recordings by quality metrics")
    self.quality_group.setCheckable(True)
    self.quality_group.setChecked(False)
    
    # Add controls for your criteria
    # Connect changes to self.update_preview()
    
    layout.addWidget(self.quality_group)
    return tab_widget
```

3. **Update the exclusion logic** in `should_exclude_recording()`:
```python
def should_exclude_recording(self, recording) -> bool:
    # Existing stimulus amplitude criteria...
    
    # Add new quality criteria
    if self.quality_group.isChecked():
        # Implement your quality criteria logic here
        quality_score = calculate_quality_score(recording)
        threshold = self.quality_threshold.value()
        if quality_score < threshold:
            return True
    
    return False
```

### Potential Future Criteria
- **Signal Quality**: Signal-to-noise ratio, artifact detection
- **Channel-specific**: Exclude based on individual channel criteria
- **Temporal**: Time-based exclusions (first/last N recordings)
- **Statistical**: Outlier detection based on amplitude distributions
- **Manual Selection**: Interactive selection from plots
- **Metadata-based**: Exclude based on recording metadata

## Technical Details

### Architecture
- **Dialog**: `RecordingExclusionEditor` - Main dialog interface
- **Command**: `BulkRecordingExclusionCommand` - Handles undo/redo for bulk operations
- **Menu Integration**: Added to Edit > Data Curation submenu

### Data Flow
1. User sets criteria in the dialog
2. `should_exclude_recording()` evaluates each recording
3. Preview table updates to show proposed changes
4. User applies changes, creating a `BulkRecordingExclusionCommand`
5. Command executes, applying exclusions to session annotations
6. UI refreshes to reflect changes

### Key Methods
- `get_sessions_for_level()`: Gets sessions based on application level
- `should_exclude_recording()`: Core logic for determining exclusions
- `update_preview()`: Updates the preview table
- `apply_exclusions()`: Applies the exclusion criteria

## Integration Points

The Recording Exclusion Editor integrates with several existing systems:

### Command Pattern
- Uses `BulkRecordingExclusionCommand` for undo/redo support
- Integrates with the existing `CommandInvoker` system

### Session Management
- Works with existing `Session.exclude_recording()` and `Session.restore_recording()` methods
- Respects existing exclusion states

### UI Updates
- Automatically refreshes plot displays after applying exclusions
- Updates data selection widgets to reflect changes

## Error Handling

The editor includes robust error handling:
- Validates session availability before opening
- Shows appropriate error messages for failures
- Gracefully handles missing dependencies
- Provides fallback behavior when command pattern is unavailable

## Performance Considerations

- Real-time preview updates only when criteria change
- Efficient bulk operations through single command
- Lazy evaluation of exclusion criteria
- Minimal memory overhead for large datasets
