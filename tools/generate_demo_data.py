"""Generate synthetic H-reflex EMG demo data for portfolio website.

This script creates a realistic synthetic human H-reflex recruitment dataset and exports
it as JSON for use in a Plotly.js interactive demo.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import from monstim_gui
sys.path.insert(0, str(Path(__file__).parent.parent))

from monstim_signals.transform import butter_bandpass_filter  # noqa: E402
from monstim_signals.transform.amplitude import _calculate_peak_to_trough_amplitude, _calculate_rms_amplitude  # noqa: E402

# ============================================================================
# Signal Processing Functions (from monstim_signals)
# ============================================================================
# ============================================================================
# Physiological Parameters
# ============================================================================

SCAN_RATE = 30000  # Hz
RECORDING_DURATION_MS = 40.0  # ms - reduced from 80ms to keep file size under 1.5 MB
NUM_SAMPLES = int(SCAN_RATE * RECORDING_DURATION_MS / 1000)  # 1200 samples
NUM_RECORDINGS = 35

# Timing parameters (adjusted to fit in 40ms window)
STIM_ONSET_MS = 5.0  # Stimulus delivered at 5 ms (moved earlier to fit H-wave)
STIM_INDEX = int(STIM_ONSET_MS * SCAN_RATE / 1000)  # sample 150

# M-wave parameters (compound muscle action potential)
M_ONSET_POST_STIM_MS = 5.0
M_PEAK_POST_STIM_MS = 9.0
M_WINDOW_START_MS = 5.0  # Post-stim
M_WINDOW_END_MS = 11.0  # Post-stim
M_DURATION_MS = 6.0
M_FREQUENCY_HZ = 200.0
M_SIGMA_MS = 1.2

# H-wave parameters (H-reflex)
H_ONSET_POST_STIM_MS = 25.0
H_PEAK_POST_STIM_MS = 29.0
H_WINDOW_START_MS = 24.0  # Post-stim
H_WINDOW_END_MS = 32.0  # Post-stim
H_DURATION_MS = 8.0
H_FREQUENCY_HZ = 100.0
H_SIGMA_MS = 1.8

# Amplitude parameters
NOISE_RMS_MV = 0.05
STIM_ARTIFACT_MIN_MV = 1.0
STIM_ARTIFACT_MAX_MV = 3.0

# M-wave recruitment (sigmoid)
M_MAX_AMPLITUDE_MV = 1.2
M_THRESHOLD_MA = 2.0
M_SATURATION_MA = 7.0
M_SIGMOID_K = 1.2

# H-wave recruitment (inverted-U, bell curve)
H_MAX_AMPLITUDE_MV = 0.4
H_PEAK_MA = 4.0
H_SIGMA_MA = 1.5
H_MIN_AMPLITUDE_MV = 0.02  # Below this, H-wave is not present


# ============================================================================
# Waveform Synthesis
# ============================================================================


def generate_biphasic_wave(time_array, peak_time_ms, sigma_ms, frequency_hz, amplitude_mv):
    """Generate a biphasic Gaussian-modulated sinusoidal waveform.

    Args:
        time_array: Time array in seconds
        peak_time_ms: Time of peak in milliseconds
        sigma_ms: Standard deviation of Gaussian envelope in milliseconds
        frequency_hz: Dominant frequency of the wave
        amplitude_mv: Peak amplitude in millivolts

    Returns:
        np.ndarray: Biphasic waveform
    """
    peak_time_s = peak_time_ms / 1000.0
    sigma_s = sigma_ms / 1000.0

    # Time relative to peak
    t_rel = time_array - peak_time_s

    # Gaussian envelope
    envelope = np.exp(-(t_rel**2) / (2 * sigma_s**2))

    # Sinusoidal component
    sine_wave = np.sin(2 * np.pi * frequency_hz * t_rel)

    # Combined waveform
    waveform = amplitude_mv * sine_wave * envelope

    return waveform


def generate_stimulus_artifact(stim_index, stim_amplitude_ma):
    """Generate stimulus artifact at the stimulus sample.

    Args:
        stim_index: Sample index where stimulus occurs
        stim_amplitude_ma: Stimulus amplitude in milliamps

    Returns:
        np.ndarray: Artifact waveform (full length, mostly zeros)
    """
    artifact = np.zeros(NUM_SAMPLES)

    # Artifact amplitude scales with stimulus intensity
    artifact_amplitude = STIM_ARTIFACT_MIN_MV + ((STIM_ARTIFACT_MAX_MV - STIM_ARTIFACT_MIN_MV) * (stim_amplitude_ma / 12.0))

    # Biphasic spike over 3 samples
    if stim_index >= 1 and stim_index < NUM_SAMPLES - 1:
        artifact[stim_index - 1] = artifact_amplitude * 0.3
        artifact[stim_index] = artifact_amplitude
        artifact[stim_index + 1] = -artifact_amplitude * 0.5

    return artifact


def calculate_m_wave_amplitude(stim_ma):
    """Calculate M-wave amplitude using sigmoid recruitment curve."""
    return M_MAX_AMPLITUDE_MV / (1 + np.exp(-M_SIGMOID_K * (stim_ma - M_THRESHOLD_MA)))


def calculate_h_wave_amplitude(stim_ma):
    """Calculate H-wave amplitude using inverted-U (bell curve) recruitment."""
    amplitude = H_MAX_AMPLITUDE_MV * np.exp(-((stim_ma - H_PEAK_MA) ** 2) / (2 * H_SIGMA_MA**2))

    # Return 0 if below noise threshold
    if amplitude < H_MIN_AMPLITUDE_MV:
        return 0.0

    return amplitude


def generate_recording(stim_amplitude_ma, time_array):
    """Generate a single recording waveform.

    Args:
        stim_amplitude_ma: Stimulus amplitude in milliamps
        time_array: Time array in seconds

    Returns:
        np.ndarray: Synthetic EMG waveform in millivolts
    """
    # Start with baseline noise
    waveform = np.random.normal(0, NOISE_RMS_MV, NUM_SAMPLES)

    # Add stimulus artifact
    artifact = generate_stimulus_artifact(STIM_INDEX, stim_amplitude_ma)
    waveform += artifact

    # Calculate wave amplitudes based on stimulus intensity
    m_amplitude = calculate_m_wave_amplitude(stim_amplitude_ma)
    h_amplitude = calculate_h_wave_amplitude(stim_amplitude_ma)

    # Generate M-wave (if amplitude is significant)
    if m_amplitude > 0.01:
        m_peak_time_ms = STIM_ONSET_MS + M_PEAK_POST_STIM_MS
        m_wave = generate_biphasic_wave(time_array, m_peak_time_ms, M_SIGMA_MS, M_FREQUENCY_HZ, m_amplitude)
        waveform += m_wave

    # Generate H-wave (if present)
    if h_amplitude > 0:
        h_peak_time_ms = STIM_ONSET_MS + H_PEAK_POST_STIM_MS
        h_wave = generate_biphasic_wave(time_array, h_peak_time_ms, H_SIGMA_MS, H_FREQUENCY_HZ, h_amplitude)
        waveform += h_wave

    # Apply bandpass filter
    filtered_waveform = butter_bandpass_filter(waveform, SCAN_RATE, lowcut=100, highcut=3500, order=4)

    return filtered_waveform


# ============================================================================
# Main Generation Logic
# ============================================================================


def round_to_sig_figs(arr, sig_figs=5):
    """Round array elements to specified significant figures."""
    return [round(float(x), sig_figs) for x in arr]


def generate_demo_data():
    """Generate complete demo dataset and return as dictionary."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate stimulus intensities (log-spaced from 0.5 to 12.0 mA)
    stim_amplitudes = np.logspace(np.log10(0.5), np.log10(12.0), NUM_RECORDINGS)

    # Create time array
    time_s = np.arange(NUM_SAMPLES) / SCAN_RATE
    time_ms = time_s * 1000.0

    # Prepare output structure
    output = {
        "meta": {
            "scan_rate": SCAN_RATE,
            "num_samples": NUM_SAMPLES,
            "stim_onset_ms": STIM_ONSET_MS,
            "m_window_ms": [M_WINDOW_START_MS, M_WINDOW_END_MS],
            "h_window_ms": [H_WINDOW_START_MS, H_WINDOW_END_MS],
            "channel_name": "Tibialis Anterior (Synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "recordings": [],
        "recruitment_curve": {
            "stim_ma": [],
            "m_wave_rms_mv": [],
            "h_wave_rms_mv": [],
            "m_wave_p2t_mv": [],
            "h_wave_p2t_mv": [],
        },
    }

    # Generate each recording
    for idx, stim_ma in enumerate(stim_amplitudes):
        print(f"Generating recording {idx + 1}/{NUM_RECORDINGS} (stim = {stim_ma:.2f} mA)...")

        # Generate waveform
        emg_mv = generate_recording(stim_ma, time_s)

        # Calculate M-wave amplitudes
        m_window_start_abs_ms = STIM_ONSET_MS + M_WINDOW_START_MS
        m_window_end_abs_ms = STIM_ONSET_MS + M_WINDOW_END_MS
        m_rms = _calculate_rms_amplitude(emg_mv, m_window_start_abs_ms, m_window_end_abs_ms, SCAN_RATE)
        m_p2t = _calculate_peak_to_trough_amplitude(emg_mv, m_window_start_abs_ms, m_window_end_abs_ms, SCAN_RATE)

        # Calculate H-wave amplitudes
        h_window_start_abs_ms = STIM_ONSET_MS + H_WINDOW_START_MS
        h_window_end_abs_ms = STIM_ONSET_MS + H_WINDOW_END_MS
        h_rms = _calculate_rms_amplitude(emg_mv, h_window_start_abs_ms, h_window_end_abs_ms, SCAN_RATE)
        h_p2t = _calculate_peak_to_trough_amplitude(emg_mv, h_window_start_abs_ms, h_window_end_abs_ms, SCAN_RATE)

        # Determine if waves are present
        m_present = m_rms > H_MIN_AMPLITUDE_MV
        h_present = h_rms > H_MIN_AMPLITUDE_MV

        # Build recording object
        recording = {
            "index": idx,
            "stim_ma": round(float(stim_ma), 5),
            "time_ms": round_to_sig_figs(time_ms, 5),
            "emg_mv": round_to_sig_figs(emg_mv, 5),
            "m_wave": {
                "window_ms": [M_WINDOW_START_MS, M_WINDOW_END_MS],
                "amplitude_rms_mv": round(float(m_rms), 5),
                "amplitude_p2t_mv": round(float(m_p2t), 5),
                "present": bool(m_present),
            },
            "h_wave": {
                "window_ms": [H_WINDOW_START_MS, H_WINDOW_END_MS],
                "amplitude_rms_mv": round(float(h_rms), 5),
                "amplitude_p2t_mv": round(float(h_p2t), 5),
                "present": bool(h_present),
            },
        }

        output["recordings"].append(recording)

        # Add to recruitment curve arrays
        output["recruitment_curve"]["stim_ma"].append(round(float(stim_ma), 5))
        output["recruitment_curve"]["m_wave_rms_mv"].append(round(float(m_rms), 5))
        output["recruitment_curve"]["h_wave_rms_mv"].append(round(float(h_rms), 5))
        output["recruitment_curve"]["m_wave_p2t_mv"].append(round(float(m_p2t), 5))
        output["recruitment_curve"]["h_wave_p2t_mv"].append(round(float(h_p2t), 5))

    return output


def validate_and_report(data, output_path):
    """Validate the generated data and print statistics."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    # File size
    file_size_kb = output_path.stat().st_size / 1024
    print(f"File size: {file_size_kb:.2f} KB")

    # Number of recordings
    num_recordings = len(data["recordings"])
    print(f"Number of recordings: {num_recordings}")

    # Extract recruitment curve data
    stim_ma = np.array(data["recruitment_curve"]["stim_ma"])
    m_rms = np.array(data["recruitment_curve"]["m_wave_rms_mv"])
    h_rms = np.array(data["recruitment_curve"]["h_wave_rms_mv"])

    # Max amplitudes
    max_m_rms = np.max(m_rms)
    max_h_rms = np.max(h_rms)
    print(f"Max M-wave RMS amplitude: {max_m_rms:.5f} mV")
    print(f"Max H-wave RMS amplitude: {max_h_rms:.5f} mV")

    # H-wave maximum location
    h_max_idx = np.argmax(h_rms)
    h_max_stim = stim_ma[h_max_idx]
    print(f"Stimulus intensity at H-wave maximum: {h_max_stim:.3f} mA")

    # M-wave threshold (first time M-wave exceeds 0.1 mV RMS)
    m_threshold_indices = np.where(m_rms > 0.1)[0]
    if len(m_threshold_indices) > 0:
        m_threshold_stim = stim_ma[m_threshold_indices[0]]
        print(f"M-wave threshold (>0.1 mV RMS): {m_threshold_stim:.3f} mA")
    else:
        print("M-wave threshold: NOT REACHED")

    # First recording waveform sanity check
    first_emg = np.array(data["recordings"][0]["emg_mv"])
    print(f"First recording EMG min/max: {np.min(first_emg):.5f} / {np.max(first_emg):.5f} mV")

    print("=" * 70)
    print("Validation complete!\n")


def main():
    """Main execution function."""
    print("MonStim Analyzer Demo Data Generator")
    print("=" * 70)
    print(f"Generating {NUM_RECORDINGS} synthetic H-reflex recordings...")
    print(f"Sampling rate: {SCAN_RATE} Hz")
    print(f"Recording duration: {RECORDING_DURATION_MS} ms")
    print(f"Total samples per recording: {NUM_SAMPLES}")
    print("=" * 70 + "\n")

    # Generate data
    data = generate_demo_data()

    # Write to file
    output_path = Path(__file__).parent / "demo_data.json"
    print(f"\nWriting data to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Validate and report
    validate_and_report(data, output_path)

    print(f"Demo data successfully generated at: {output_path}")


if __name__ == "__main__":
    main()
