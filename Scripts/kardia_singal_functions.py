
#from pydub import AudioSegment
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.signal import butter, lfilter, hilbert, sosfilt, sosfiltfilt
#from scipy.fftpack import fft

from filtering_functions import bandpass_filter

#from scipy.fft import fft, ifft, fftfreq

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




# Stereo Channel Selection
def select_best_channel(data, fs, CARRIER_FREQ):
    if data.ndim == 2:  # Stereo signal
        channel_energies = []
        for channel in range(data.shape[1]):
            channel_data = data[:, channel]
            bandpassed = bandpass_filter(channel_data, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
            snr = np.sum(np.abs(bandpassed)**2) / np.sum(np.abs(channel_data - bandpassed)**2)
            channel_energies.append(snr)
        print('Channel energies')
        print(channel_energies)
        best_channel = np.argmax(channel_energies)
        print(f"Selected Channel: {best_channel}")
        return data[:, best_channel]
    return data  # Already mono
# Mains Frequency Filtering


