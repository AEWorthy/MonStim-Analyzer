# M-max Detection Algorithm Documentation

## Overview

The M-max (maximum M-wave amplitude) detection algorithm in MonStim Analyzer employs a sophisticated multi-level approach to accurately determine the maximum motor response across different hierarchical levels of the data structure: individual recordings, sessions, datasets, and experiments.

## Algorithm Architecture

### Design Philosophy

The M-max detection algorithm is built on three core principles:

1. **Multi-Approach Robustness**: Rather than relying on a single calculation method, the algorithm evaluates multiple approaches and selects the most appropriate one based on data characteristics and validation criteria.

2. **Local Context Validation**: Validation compares results against local context (plateau mean or region mean) rather than global metrics, ensuring that selected values are consistent with their immediate neighborhood and resistant to distant artifacts.

3. **Consistent Methodology**: The same analytical framework is applied across all scenarios - whether plateau detection succeeds or fails, the algorithm uses identical validation logic and selection criteria.

### Two-Tier System

1. **Session Level**: Advanced plateau detection with multi-approach validation 
3. **Dataset/Experiment Level**: Mean-based aggregation for population-level normalization

## Individual Recording Algorithm

### Core Algorithm Flow

```
Input: stimulus_voltages[], m_wave_amplitudes[]
│
├─ Step 1: Plateau Detection
│   ├─ Apply Savitzky-Golay smoothing
│   ├─ Sliding window standard deviation analysis  
│   └─ Identify stable region (low variability)
│
├─ Step 2: Multi-Approach Calculation
│   ├─ Mean Corrected: Traditional average + outlier correction
│   ├─ 95th Percentile: Robust high-end estimate
│   ├─ Maximum: Most aggressive approach
│   └─ Top 20% Mean: Balanced high-value average
│
├─ Step 3: Intelligent Selection
│   ├─ Validate each approach against plateau/region mean (configurable tolerance)
│   ├─ Prefer maximum if ≤ validation_tolerance of plateau mean (improved validation)
│   └─ Fall back through hierarchy if validation fails
│
└─ Output: Selected M-max value + metadata
```

### Mathematical Details

#### Plateau Detection
The plateau detection uses a sliding window approach with adaptive sizing:

```python
def detect_plateau(y, max_window_size=20, min_window_size=3, threshold=0.3):
    y_filtered = savitzky_golay_filter(y)
    
    for window_size in range(max_window_size, min_window_size-1, -1):
        for i in range(len(y_filtered) - window_size):
            window = y_filtered[i:i + window_size]
            if std(window) < threshold * std(y_filtered):
                return i, i + window_size  # plateau found
    
    return None, None  # no plateau detected
```

The Savitzky-Golay filter parameters:
- **Window length**: 25% of data points (minimum 5, always odd)
- **Polynomial order**: 3 (or window_length-1 if window is small)

#### Multi-Approach Calculation

**1. Mean Corrected Approach**
```
plateau_mean = mean(plateau_data)
outliers = all_data[all_data > plateau_mean]  
below_max_plateau = plateau_data[plateau_data < max(plateau_data)]

if outliers exist and below_max_plateau exist:
    correction = mean(outliers) - mean(below_max_plateau)
    M_max = plateau_mean + correction
else:
    M_max = plateau_mean
```

**2. Percentile Approaches**
```
M_max_p95 = percentile(plateau_data, 95)
M_max_max = max(plateau_data)
```

**3. Top Percentage Mean**
```
threshold_80 = percentile(plateau_data, 80)
top_20_percent = plateau_data[plateau_data >= threshold_80]
M_max_top20 = mean(top_20_percent)
```

#### Selection Logic
```python
plateau_mean = mean(plateau_data)
tolerance = validation_tolerance  # Configurable tolerance (default: 1.05 = 5%)

# Improved validation: compare against plateau mean, not global maximum
# This prevents artifacts from dominating and ensures plateau consistency
if M_max_maximum <= plateau_mean * tolerance:
    selected = M_max_maximum, "maximum"
elif M_max_p95 <= plateau_mean * tolerance:
    selected = M_max_p95, "95th_percentile"  
elif M_max_top20 <= plateau_mean * tolerance:
    selected = M_max_top20, "top_20_percent_mean"
else:
    selected = M_max_corrected, "mean_corrected"  # fallback
```

**Validation Logic Rationale:**
- **Local Context**: Compares plateau maximum against plateau mean (5% tolerance)
- **Artifact Resistance**: Movement artifacts typically affect single points, not entire plateau regions
- **Physiological Constraint**: True plateau should be relatively flat - large deviations suggest measurement problems
- **Tighter Validation**: Much more restrictive than comparing against global maximum

#### Fallback Algorithm
When no plateau is detected, the algorithm applies the same multi-approach methodology to the high-stimulus region:

```python
# Use high-stimulus region (top 25% of stimulus range)
sorted_indices = argsort(stimulus_voltages)
top_25_percent_idx = int(len(sorted_indices) * 0.75)
high_stim_indices = sorted_indices[top_25_percent_idx:]
high_stim_amplitudes = m_wave_amplitudes[high_stim_indices]

# Apply same multi-approach methodology
approaches = []
m_max_mean = mean(high_stim_amplitudes)
m_max_p95 = percentile(high_stim_amplitudes, 95)
m_max_max = max(high_stim_amplitudes)

# Same validation logic as main algorithm
region_mean = mean(high_stim_amplitudes)
tolerance = validation_tolerance  # Configurable tolerance

if m_max_max <= region_mean * tolerance:
    selected = m_max_max, "maximum"
elif m_max_p95 <= region_mean * tolerance:
    selected = m_max_p95, "95th_percentile"
else:
    selected = m_max_mean, "mean"
```

**Benefits of Multi-Approach Fallback:**
- **Consistency**: Same methodology regardless of plateau detection success
- **Robustness**: Multiple approaches provide better estimates than single percentile
- **Validation**: Same quality control applied to fallback estimates
- **Debugging**: Detailed logging for troubleshooting difficult cases

## Dataset and Experiment Aggregation

### Design Philosophy
The aggregation algorithm uses mean-based calculations to provide proper population-level normalization for H/M ratio analysis. This approach ensures that normalized reflex measurements represent true population characteristics rather than individual maxima.

### Mean-Based Aggregation Approach

#### Algorithm Steps

**1. Dataset Level Aggregation**
```python
def get_dataset_mmax(sessions):
    session_mmaxes = [session.get_m_max() for session in sessions]
    dataset_mmax = mean(session_mmaxes)
    return dataset_mmax
```

**2. Experiment Level Aggregation**
```python
def get_experiment_mmax(datasets):
    dataset_mmaxes = [dataset.get_avg_m_max() for dataset in datasets]
    experiment_mmax = mean(dataset_mmaxes)
    return experiment_mmax
```

#### Rationale for Mean-Based Aggregation

**Population-Level Normalization:**
- M-max for population studies represents average muscle response capacity
- H-reflex normalized by mean M-max gives true population H/M ratios
- Individual subject variability is properly captured in the normalization

**Inclusivity Benefits:**
- All sessions contribute regardless of stimulus range achieved
- Sessions with exploratory high-voltage tests don't bias results
- Natural experimental variation is preserved

**Statistical Validity:**
- Mean aggregation appropriate for population parameters
- Enables proper statistical comparisons across conditions
- Maintains relationship between individual and group measures

#### Example Calculation
```
Dataset with 3 sessions:
Session A: M-max = 2.4 mV
Session B: M-max = 2.8 mV  
Session C: M-max = 3.2 mV

Dataset M-max = (2.4 + 2.8 + 3.2) / 3 = 2.8 mV

Experiment with 2 datasets:
Dataset 1: M-max = 2.8 mV
Dataset 2: M-max = 3.1 mV

Experiment M-max = (2.8 + 3.1) / 2 = 2.95 mV
```

### Benefits for Research Applications

**H-reflex Studies:**
- Population H/M ratios properly normalized
- Cross-subject comparisons statistically valid
- Condition effects not confounded by individual M-max variation

**Longitudinal Analysis:**
- Consistent normalization across time points
- Individual subject changes preserved in group analysis
- No bias from protocol variations between sessions

**Multi-Site Studies:**
- Standardized normalization approach
- Comparable results across different stimulation protocols
- Reduced inter-site variability in normalized measures

## Debugging and Logging

### Debug Output Format
```
DEBUG: Plateau region detected with window size 15. Threshold: 0.3 times SD.
DEBUG: M-max calculation: selected 'maximum' approach, value: 2.4843907139884704
DEBUG:   Validation: within 105.0% of plateau mean
DEBUG:   mean_corrected: 2.403818
DEBUG:   95th_percentile: 2.477664  
DEBUG:   maximum: 2.484391
DEBUG:   top_20_percent_mean: 2.457541
DEBUG:   plateau_mean: 2.350000
DEBUG:   validation_tolerance: 1.050
DEBUG: Final M-max amplitude: 2.4843907139884704

# Fallback case example:
WARNING: No plateau detected, using fallback multi-approach detection in high-stimulus region
DEBUG: Fallback M-max calculation: selected 'mean' approach, value: 1.863727
DEBUG:   Validation: fallback to mean - other approaches exceeded tolerance
DEBUG:   mean: 1.863727
DEBUG:   95th_percentile: 2.393524
DEBUG:   maximum: 2.475991
DEBUG:   high_stim_region_mean: 1.863727
DEBUG:   validation_tolerance: 1.050

DEBUG: Dataset M-max: Using mean from 4 sessions
DEBUG:   M-max values: [2.484391, 2.692246, 2.683790, 2.740320]
DEBUG:   Mean M-max: 2.650187

DEBUG: Experiment M-max: Using mean from 3 datasets
DEBUG:   M-max values: [2.650187, 1.985432, 3.125789]
DEBUG:   Mean M-max: 2.587136
```

### Performance Validation
Expected behavior with the improved algorithm:
- Individual M-max values should be close to maximum recorded amplitudes
- Normalized maximum M-wave amplitudes should be ≈ 1.0 (range 0.95-1.05)
- H-reflex curves should display physiologically reasonable amplitudes when normalized

## Configuration Parameters

### Individual Algorithm Parameters
```yaml
m_max_args:
  max_window_size: 20         # Maximum plateau detection window
  min_window_size: 3          # Minimum plateau detection window  
  threshold: 0.3              # SD threshold for plateau (0.3 * overall_SD)
  validation_tolerance: 1.05  # Validation tolerance (5% above plateau/region mean)
```

### Analysis Parameters
```yaml
analysis:
  default_method: 'rms'       # Default amplitude calculation method
  time_window: 8.0            # EMG analysis time window (ms)
  pre_stim_time: 2.0          # Pre-stimulus baseline period (ms)
```

## Integration with Plotting

### H-reflex Normalization
The experiment plotter now properly normalizes H-reflex data:

```python
if relative_to_mmax and experiment_mmax is not None:
    h_response_means = h_response_means / experiment_mmax
    h_response_errors = h_response_errors / experiment_mmax
    ylabel = "H-reflex Amplitude (relative to M-max)"
```

This ensures that when "Relative to M-max" is enabled, the H-reflex curves display correctly normalized values based on the improved M-max calculations.


This comprehensive approach ensures accurate, robust, and physiologically meaningful M-max calculations across all levels of the data hierarchy.
