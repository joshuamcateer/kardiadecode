
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert, sosfilt
from scipy.fftpack import fft
import os


from scipy.fft import fft, ifft, fftfreq

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

# FM Modulation
def fm_modulate(signal, carrier_freq, fs, calibration):
    """FM modulates the ECG signal."""
    # Convert mV to Hz shift
    deviation = signal * calibration
    # Generate time vector
    t = np.arange(len(signal)) / fs
    # Perform FM modulation
    modulated_signal = np.sin(2 * np.pi * carrier_freq * t + 2 * np.pi * np.cumsum(deviation) / fs)
    return modulated_signal





# Butterworth Bandpass Filter Design
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    
    if not (0 < lowcut < nyquist and 0 < highcut < nyquist):
        raise ValueError("Cutoff frequencies must be between 0 and the Nyquist frequency.")
    
    if highcut - lowcut < 1:  # Adjust threshold as needed for stability
        print("Warning: Narrow bandwidth detected. Consider lowering filter order.")
        order = min(order, 2)  # Reduce order for stability
    
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')

    return sos


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaN or Inf values.")
    max_val = np.max(np.abs(data))

    assert max_val < 1e6, print("Very large values found before filtering")
    # if max_val > 1e6:  # Threshold for large amplitude signals
    #     print("Warning: Large signal amplitude detected. Normalizing input.")
    #     data = data / max_val
    #     b, a = butter_bandpass(lowcut, highcut, fs, order)
    #     return lfilter(b, a, data)*max_val

    sos = butter_bandpass(lowcut, highcut, fs, order)

    return sosfilt(sos, data)


def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    sos = butter(order, low, btype='low', output='sos')
    return sosfilt(sos, data)
