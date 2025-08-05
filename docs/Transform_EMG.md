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

Where $T_1$ and $T_2$ represent the start and end times of the analysis window, respectively. The default method is RMS amplitude.

### M-max Determination
To determine the maximum M-wave amplitude (M-max), an adaptive algorithm is employed that identifies the plateau region in the stimulus-response curve:

1. The stimulus-response curve is smoothed using a Savitzky-Golay filter with a window length of 25% of the total data points and a polynomial order of 3.

2. A sliding window approach is used to detect the plateau region. The window size starts at 15 data points (configurable) and decreases to a minimum of 2 points if no plateau is detected.

3. For each window, the standard deviation of the M-wave amplitudes is calculated. A plateau is identified when the standard deviation falls below a threshold of 0.2 times the overall standard deviation of the data (configurable).

4. The M-max is calculated as the mean amplitude within the detected plateau region:

   $$M_{max} = \frac{1}{n} \sum_{i=1}^{n} A_i$$

   where $A_i$ are the M-wave amplitudes within the plateau region and $n$ is the number of points in the plateau.

5. If the calculated M-max is lower than the maximum recorded M-wave amplitude, a correction is applied:

   $$M_{max_{corrected}} = M_{max} + (\overline{A_{outliers}} - \overline{A_{plateau_{below\_max}}})$$

   where $\overline{A_{outliers}}$ is the mean amplitude of M-waves exceeding the initial M-max, and $\overline{A_{plateau_{below\_max}}}$ is the mean amplitude of the plateau region excluding its maximum value.

This adaptive approach ensures robust M-max determination across varying stimulus-response curve shapes and noise levels. The algorithm parameters (window sizes, threshold) can be adjusted in the configuration settings.

## Configuration Parameters

All processing parameters can be customized through the application's configuration system:

### Filter Parameters
- **lowcut**: Low cutoff frequency for EMG bandpass filter (default: 100 Hz)
- **highcut**: High cutoff frequency for EMG bandpass filter (default: 3500 Hz)  
- **order**: Filter order (default: 4)

### M-max Detection Parameters
- **max_window_size**: Maximum sliding window size for plateau detection (default: 15 data points)
- **min_window_size**: Minimum sliding window size for plateau detection (default: 2 data points)
- **threshold**: Standard deviation threshold for plateau identification (default: 0.2)

### Analysis Parameters
- **default_method**: Default amplitude calculation method (default: 'rms')
- **time_window**: EMG analysis time window (default: 8.0 ms)
- **pre_stim_time**: Pre-stimulus baseline period (default: 2.0 ms)

These parameters can be adjusted globally or on a per-analysis-profile basis to accommodate different experimental conditions and signal characteristics.