# EMG Processing and Analysis

## Data Acquisition
Electromyographic (EMG) signals were recorded using standard bipolar surface electrodes. The sampling rate was set at 4000 Hz.

## EMG Pre-processing
Raw EMG signals were pre-processed using a series of steps to improve signal quality and prepare the data for analysis.

### Bandpass Filtering
A fourth-order Butterworth bandpass filter was applied to remove low-frequency motion artifacts and high-frequency noise. The filter was designed with cutoff frequencies of 100 Hz (low cut) and 3500 Hz (high cut). The transfer function of the Butterworth filter is given by:

$$H(s) = \frac{1}{\sqrt{1 + \epsilon^2 (\frac{s}{j\omega_c})^{2N}}}$$

where $\epsilon$ is the maximum passband gain, $\omega_c$ is the cutoff frequency, and $N$ is the filter order (in this case, $N = 4$).

The filter was applied using a forward-backward technique to achieve zero phase distortion.

### Baseline Correction
[In the end, I decided not to use this function in data processing]To account for any DC offset, the EMG signals were corrected relative to the pre-stimulus baseline amplitude. The baseline was calculated as the average amplitude of the signal during the pre-stimulus period (from 0 ms to the stimulus onset).

### Signal Rectification
For certain analyses, the EMG signals were full-wave rectified by taking the absolute value of each data point:

$$EMG_{rectified}(t) = |EMG_{raw}(t)|$$

## EMG Analysis

### Amplitude Calculations
Various methods were employed to calculate EMG amplitude, depending on the specific analysis requirements:

1. Average Rectified Amplitude:
   $$A_{avg} = \frac{1}{T_2 - T_1} \int_{T_1}^{T_2} |EMG(t)| dt$$

2. Root Mean Square (RMS) Amplitude:
   $$A_{RMS} = \sqrt{\frac{1}{T_2 - T_1} \int_{T_1}^{T_2} EMG^2(t) dt}$$

3. Peak-to-Trough Amplitude:
   $$A_{p-t} = max(EMG(t)) - min(EMG(t)), \quad T_1 \leq t \leq T_2$$

Where $T_1$ and $T_2$ represent the start and end times of the analysis window, respectively.

### M-max Determination
To determine the maximum M-wave amplitude (M-max), we employed an adaptive algorithm that identifies the plateau region in the stimulus-response curve:

1. The stimulus-response curve was smoothed using a Savitzky-Golay filter with a window length of 25% of the total data points and a polynomial order of 3.

2. A sliding window approach was used to detect the plateau region. The window size was initially set to 20 data points and decreased to a minimum of 3 points if no plateau was detected.

3. For each window, the standard deviation of the M-wave amplitudes was calculated. A plateau was identified when the standard deviation fell below a threshold of 0.3 times the overall standard deviation of the data.

4. The M-max was calculated as the mean amplitude within the detected plateau region:

   $$M_{max} = \frac{1}{n} \sum_{i=1}^{n} A_i$$

   where $A_i$ are the M-wave amplitudes within the plateau region and $n$ is the number of points in the plateau.

5. If the calculated M-max was lower than the maximum recorded M-wave amplitude, a correction was applied:

   $$M_{max_{corrected}} = M_{max} + (\overline{A_{outliers}} - \overline{A_{plateau}})$$

   where $\overline{A_{outliers}}$ is the mean amplitude of M-waves exceeding the initial M-max, and $\overline{A_{plateau}}$ is the mean amplitude of the plateau region excluding its maximum value.

This adaptive approach ensures robust M-max determination across varying stimulus-response curve shapes and noise levels.