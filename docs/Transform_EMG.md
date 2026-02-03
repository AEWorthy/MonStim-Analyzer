# EMG Processing and Analysis

## Data Acquisition
Electromyographic (EMG) signals were recorded using custom equipment and the LabView 'MonStim' program created by Bill Goolsby. The sampling rate varies by experiment but is typically around 30 kHz.

## EMG Pre-processing
Raw signals are pre-processed using different methods depending on the channel type to improve signal quality and prepare the data for analysis.

### Channel-Specific Processing

**EMG Channels:**
- **Bandpass Filtering**: A fourth-order Butterworth bandpass filter is applied to remove low-frequency motion artifacts and high-frequency noise. The filter is designed with cutoff frequencies of 100 Hz (low cut) and 3500 Hz (high cut).

**Force and Length Channels:**
- **Baseline Correction**: Non-EMG channels (force, length) undergo baseline correction relative to the pre-stimulus amplitude instead of bandpass filtering.

**Other Channel Types:**
- Channels not specifically identified as EMG, force, or length are passed through without filtering but may have polarity inversion applied if specified by the user.

### Bandpass Filtering (EMG Channels Only)
The Butterworth bandpass filter transfer function is given by:

$$H(s) = \frac{1}{\sqrt{1 + \epsilon^2 (\frac{s}{j\omega_c})^{2N}}}$$

where $\epsilon$ is the maximum passband gain, $\omega_c$ is the cutoff frequency, and $N$ is the filter order (in this case, $N = 4$).

The filter is applied using a forward-backward technique (`scipy.signal.filtfilt`) to achieve zero phase distortion.

### Baseline Correction (Non-EMG Channels)
For force and length channels, baseline correction is applied to account for any DC offset. The signals are corrected relative to the pre-stimulus baseline amplitude. The baseline is calculated as the average amplitude of the signal during the pre-stimulus period (from 0 ms to the stimulus onset):

$$Signal_{corrected}(t) = Signal_{raw}(t) - \overline{Signal_{baseline}}$$

where $\overline{Signal_{baseline}}$ is the mean amplitude during the pre-stimulus period.

### Signal Rectification
For certain analyses, EMG signals can be full-wave rectified by taking the absolute value of each data point:

$$EMG_{rectified}(t) = |EMG_{filtered}(t)|$$

This rectification is applied on-demand during analysis calculations and is not part of the standard preprocessing pipeline (Note: raw user data is stored locally in H5 formatted binary files during import and is used by the program on a *read-only* basis).

### Channel Polarity Inversion
Individual channels can have their polarity inverted if specified in the channel annotations:

$$Signal_{inverted}(t) = -Signal_{processed}(t)$$

This inversion is applied after filtering or baseline correction, depending on the channel type.

## EMG Analysis

### Amplitude Calculations
Various methods are employed to calculate EMG amplitude, depending on the specific analysis requirements:

1. **Average Rectified Amplitude:**
   $$A_{avg} = \frac{1}{T_2 - T_1} \int_{T_1}^{T_2} |EMG(t)| dt$$

2. **Root Mean Square (RMS) Amplitude:**
   $$A_{RMS} = \sqrt{\frac{1}{T_2 - T_1} \int_{T_1}^{T_2} EMG^2(t) dt}$$

3. **Peak-to-Trough Amplitude:**
   $$A_{p-t} = max(EMG(t)) - min(EMG(t)), \quad T_1 \leq t \leq T_2$$

4. **Average Unrectified Amplitude:**
   $$A_{unrect} = \frac{1}{T_2 - T_1} \int_{T_1}^{T_2} EMG(t) dt$$

5. **Area Under Curve (AUC) of Rectified Signal:**
   $$A_{AUC} = \int_{T_1}^{T_2} |EMG(t)| dt$$
   
   This method calculates the total area under the rectified EMG curve, providing a measure of the cumulative muscle activation over the analysis window.

Where $T_1$ and $T_2$ represent the start and end times of the analysis window, respectively. The default method is RMS amplitude.

### M-max Determination

#### Individual Recording M-max Algorithm
The maximum M-wave amplitude (M-max) for individual recordings is determined using a multi-approach algorithm that identifies the plateau region in the stimulus-response curve and selects the most appropriate calculation method:

**Step 1: Plateau Detection**
1. The stimulus-response curve is smoothed using a Savitzky-Golay filter with a window length of 25% of the total data points and a polynomial order of 3.

2. A sliding window approach detects the plateau region. The window size starts at 20 data points (configurable) and decreases to a minimum of 3 points if no plateau is detected.

3. For each window, the standard deviation of M-wave amplitudes is calculated. A plateau is identified when the standard deviation falls below a threshold of 0.3 times the overall standard deviation (configurable).

**Step 2: Multi-Approach M-max Calculation**
Once a plateau is detected, four different approaches are calculated:

1. **Mean Corrected Approach:**
   $$M_{max\_mean} = \frac{1}{n} \sum_{i=1}^{n} A_i + correction$$
   
   where the correction factor accounts for outliers above the plateau mean:
   $$correction = \overline{A_{outliers}} - \overline{A_{plateau\_below\_max}}$$

2. **95th Percentile Approach:**
   $$M_{max\_p95} = P_{95}(A_{plateau})$$

3. **Maximum Approach:**
   $$M_{max\_max} = \max(A_{plateau})$$

4. **Top 20% Mean Approach:**
   $$M_{max\_top20} = \frac{1}{k} \sum_{i=1}^{k} A_{top20\%}$$
   
   where $A_{top20\%}$ are amplitudes ≥ 80th percentile of the plateau region.

**Step 3: Selection Logic**
The algorithm selects the most appropriate approach using improved validation criteria:

- **Primary choice**: Maximum approach if ≤ validation_tolerance of plateau mean (default: 105%)
- **Secondary choice**: 95th percentile if ≤ validation_tolerance of plateau mean  
- **Tertiary choice**: Top 20% mean if ≤ validation_tolerance of plateau mean
- **Fallback**: Mean corrected approach

**Validation Improvement**: The algorithm now compares each approach against the plateau mean rather than the global maximum. This provides much tighter validation since the plateau region should be relatively flat, and prevents movement artifacts or stimulus artifacts (which might be the global maximum) from influencing the selection criteria.

The validation tolerance is configurable via the `validation_tolerance` parameter (default: 1.05 = 5% tolerance).

**Step 4: Fallback for No Plateau**
If no plateau is detected, the algorithm applies the same multi-approach methodology to the high-stimulus region (top 25% of stimulus intensities):

1. **Mean Approach:** $M_{max\_mean} = \frac{1}{n} \sum_{i=1}^{n} A_{high\_stim}$

2. **95th Percentile Approach:** $M_{max\_p95} = P_{95}(A_{high\_stim})$

3. **Maximum Approach:** $M_{max\_max} = \max(A_{high\_stim})$

**Selection Logic for Fallback:** Same validation as main algorithm - maximum approach preferred if ≤ validation_tolerance of high-stimulus region mean, otherwise 95th percentile, with mean as final fallback.

This ensures consistent methodology whether or not a plateau is detected, providing more robust and reliable M-max estimates even in challenging data conditions.

#### Dataset and Experiment Level M-max Aggregation

**Dataset Level Aggregation:**
Dataset M-max is calculated as the mean of all constituent session M-max values:

$$M_{max\_dataset} = \frac{1}{n} \sum_{i=1}^{n} M_{max\_session_i}$$

where $n$ is the number of sessions in the dataset.

**Experiment Level Aggregation:**
Experiment M-max is calculated as the mean of all constituent dataset M-max values:

$$M_{max\_experiment} = \frac{1}{m} \sum_{j=1}^{m} M_{max\_dataset_j}$$

where $m$ is the number of datasets in the experiment.

**Rationale for Mean-Based Aggregation:**
This approach provides proper population-level normalization for M/H ratio calculations:

- **Population Representation**: Mean M-max represents the average muscle response capacity across the population
- **Proper Normalization**: H-reflex values normalized by mean M-max give true population-level H/M ratios
- **Inclusivity**: All sessions and datasets contribute to the aggregate, regardless of stimulus range achieved
- **Statistical Validity**: Mean aggregation is appropriate for population-level parameters

This approach ensures that:
- **H-reflex Normalization**: Provides accurate population-level "Relative to M-max" calculations
- **Cross-session Comparisons**: Maintains proper scaling across different experimental sessions
- **Population Studies**: Enables meaningful comparisons across different subjects/conditions

## Configuration Parameters

All processing parameters can be customized through the application's configuration system:

### Filter Parameters
- **lowcut**: Low cutoff frequency for EMG bandpass filter (default: 100 Hz)
- **highcut**: High cutoff frequency for EMG bandpass filter (default: 3500 Hz)  
- **order**: Filter order (default: 4)

### M-max Detection Parameters
- **max_window_size**: Maximum sliding window size for plateau detection (default: 20 data points)
- **min_window_size**: Minimum sliding window size for plateau detection (default: 3 data points)  
- **threshold**: Standard deviation threshold for plateau identification (default: 0.3)
- **validation_tolerance**: Tolerance factor for plateau validation (default: 1.05 = 5% above plateau mean)

### Analysis Parameters
- **default_method**: Default amplitude calculation method (default: 'rms')
- **time_window**: EMG analysis time window (default: 8.0 ms)
- **pre_stim_time**: Pre-stimulus baseline period (default: 2.0 ms)

These parameters can be adjusted globally or on a per-analysis-profile basis to accommodate different experimental conditions and signal characteristics.