# Model heart beat sound to test kardia app and FM encoding

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import sounddevice as sd
import time

# Constants
SAMPLE_RATE = 48000  # Audio sampling rate
CARRIER_FREQ = 19000  # Carrier frequency in Hz
CALIBRATION = 200  # 200 Hz/mV
HEART_RATE = 75  # Beats per minute
ECG_DURATION = 60  # ECG playback duration in seconds


# Generate Artificial Heartbeat Signal
def generate_ecg_signal(duration, heart_rate, fs):
    """Generates an artificial Lead I ECG signal."""
    # Basic parameters
    total_samples = int(fs * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    samples_per_beat = int(fs * 60 / heart_rate)
    
    # PQRST wave definition (Lead I shape approximation)
    p_wave = np.exp(-np.power((np.linspace(0, 1, samples_per_beat) - 0.1), 2) / (2 * 0.005**2)) * 0.1  # P-wave
    q_wave = -np.exp(-np.power((np.linspace(0, 1, samples_per_beat) - 0.2), 2) / (2 * 0.002**2)) * 0.2  # Q-wave
    r_wave = np.exp(-np.power((np.linspace(0, 1, samples_per_beat) - 0.3), 2) / (2 * 0.004**2)) * 1.5  # R-wave
    s_wave = -np.exp(-np.power((np.linspace(0, 1, samples_per_beat) - 0.35), 2) / (2 * 0.003**2)) * 0.5  # S-wave
    t_wave = np.exp(-np.power((np.linspace(0, 1, samples_per_beat) - 0.5), 2) / (2 * 0.015**2)) * 0.2  # T-wave

    # Assemble the waveform
    beat = p_wave + q_wave + r_wave + s_wave + t_wave
    num_beats = total_samples // samples_per_beat + 1
    ecg_signal = np.tile(beat, num_beats)[:total_samples]

    return ecg_signal, t



# Visualization and Playback
def play_and_visualize(ecg_signal, modulated_signal, t, fs, duration):
    """Plays the FM-modulated signal while visualizing the ECG."""
    fig, ax = plt.subplots()
    line, = ax.plot(t, ecg_signal, label="ECG Signal")
    marker, = ax.plot([], [], 'ro')  # Marker for the current point
    ax.set_xlim(0, t[-1])
    ax.set_ylim(np.min(ecg_signal) - 0.1, np.max(ecg_signal) + 0.1)
    ax.set_title("ECG Signal and Real-Time Point")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()

    def audio_callback(outdata, frames, time, status):
        """Stream callback to feed sound device."""
        nonlocal modulated_signal, start_idx
        end_idx = start_idx + frames
        
        # Handle edge cases where the slice might be shorter or longer than `frames`
        if end_idx > len(modulated_signal):
            # Fill the remaining part with zeros if at the end of the signal
            
            signal_slice = modulated_signal[start_idx:]
            padding = np.zeros(frames - len(signal_slice))
            signal_slice = np.concatenate((signal_slice, padding))
        else:
            signal_slice = modulated_signal[start_idx:end_idx]
        
        # Ensure the slice matches the number of frames
        if len(signal_slice) != frames:
            signal_slice = np.resize(signal_slice, frames)

        # Output the slice to the audio buffer
        outdata[:, 0] = signal_slice
        start_idx += frames



    # Initialize sound playback
    start_idx = 0
    with sd.OutputStream(callback=audio_callback, samplerate=fs, channels=1):
        start_time = time.time()
        while time.time() - start_time < duration:
            # Update marker
            elapsed_time = time.time() - start_time
            current_idx = int(elapsed_time * fs)
            if current_idx < len(t):
                marker.set_data([t[current_idx]], [ecg_signal[current_idx]])  # Wrap values in lists
            plt.pause(0.01)


# Main Script
if __name__ == "__main__":
    # Generate the artificial ECG signal
    ecg_signal, t = generate_ecg_signal(ECG_DURATION, HEART_RATE, SAMPLE_RATE)
    
    # FM modulate the signal
    modulated_signal = fm_modulate(ecg_signal, CARRIER_FREQ, SAMPLE_RATE, CALIBRATION)
    
    # Normalize modulated signal for audio playback
    modulated_signal /= np.max(np.abs(modulated_signal))
    
    # Play and visualize
    play_and_visualize(ecg_signal, modulated_signal, t, SAMPLE_RATE, ECG_DURATION)
