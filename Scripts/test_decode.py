import numpy as np
from scipy.signal import correlate, butter, lfilter, hilbert, resample
from scipy.stats import norm
import matplotlib.pyplot as plt

import pywt
from scipy.optimize import minimize

from kardia_singal_functions import generate_ecg_signal, fm_modulate, butter_bandpass, bandpass_filter, lowpass_filter, frequency_domain_filter
from signal_generation import *

# Constants
CARRIER_FREQ = 19000  # Carrier frequency in Hz
CALIBRATION = 200     # 200 Hz/mV
SAMPLE_RATE = 48000   # Audio sampling rate
FILTER_LOW = 0.5      # Low cutoff for cardiac signal (Hz)
FILTER_LOW = 0.1      # Low cutoff for cardiac signal (Hz)
FILTER_HIGH = 40      # High cutoff for cardiac signal (Hz)
FILTER_HIGH = 150      # High cutoff for cardiac signal (Hz)
MIN_RECORD_GAP_LENGTH = 3  # Minimum gap length in seconds
MIN_RECORD_LENGTH = 10     # Minimum signal length in seconds
NOISE_THRESHOLD = 0.05     # Noise threshold (adjust as needed)




def wavelet_transform(signal, wavelet='db4', level=None):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_flat, slices = pywt.coeffs_to_array(coeffs)
    return coeffs_flat, slices


def inverse_wavelet_transform(coeffs_flat, slices, wavelet='db4'):
    coeffs = pywt.array_to_coeffs(coeffs_flat, slices, output_format='wavedec')
    return pywt.waverec(coeffs, wavelet)


def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def iterative_compressed_sensing(y, wavelet='db4', sparsity_weight=1e-2, max_iter=50, tol=1e-6):
    coeffs_flat, slices = wavelet_transform(y, wavelet)
    x_recovered = np.zeros_like(coeffs_flat)
    
    for i in range(max_iter):
        # Transform reconstructed signal back to the time domain
        reconstructed_signal = inverse_wavelet_transform(x_recovered, slices, wavelet)
        
        # Enforce fidelity: Replace the coefficients in the wavelet domain to match observed data
        fidelity_term = wavelet_transform(reconstructed_signal, wavelet)[0]
        fidelity_term += coeffs_flat - fidelity_term  # Correction step to match original
        
        # Enforce sparsity: Apply soft thresholding to promote sparsity
        x_new = soft_thresholding(fidelity_term, sparsity_weight)
        
        # Check convergence
        if np.linalg.norm(x_new - x_recovered) < tol:
            break
        x_recovered = x_new
    
    return x_recovered, slices



def fm_demodulate(signal, fs, carrier_freq):
    # FM Demodulation hilbert transform based
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0 * np.pi)
    return instantaneous_frequency - carrier_freq




def superheterodyne_demodulate(signal, fs, carrier_freq, if_freq):
    # This looks weird so I'm moving it to inside the scope of 
    # the only function in which it is used

    def fm_demodulate(signal, fs):
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0 * np.pi)
        print(instantaneous_frequency.shape)
        #return np.append(instantaneous_frequency,[0])
        return instantaneous_frequency

    # Step 1: Mix signal to IF
    t = np.arange(len(signal)) / fs
    local_oscillator = np.cos(2 * np.pi * (carrier_freq - if_freq) * t)
    mixed_signal = signal * local_oscillator

    # Step 2: Bandpass filter around IF
    if_bandwidth = 2e3  # Adjust as needed
    filtered_if_signal = bandpass_filter(mixed_signal, if_freq - if_bandwidth/2, if_freq + if_bandwidth/2, fs)

    # Step 3: FM Demodulation
    demodulated_if = fm_demodulate(filtered_if_signal, fs)

    # Step 4: Low-pass filter the demodulated signal
    demodulated_baseband = lowpass_filter(demodulated_if, 500, fs)
    print('length of demodulate baseband', demodulated_baseband.shape)

    return demodulated_baseband


def compressive_sensing_reconstruction(y, wavelet='db4', sparsity_weight=1e-3, max_iter=100):
    # Get the wavelet transform matrix size
    coeffs = wavelet_transform(y, wavelet)
    coeffs_flat, slices = pywt.coeffs_to_array(coeffs)
    
    def objective(x):
        # Objective function: sparsity (L1 norm) + fidelity (L2 norm)
        fidelity = np.linalg.norm(y - inverse_wavelet_transform(pywt.array_to_coeffs(x, slices, wavelet))) ** 2
        sparsity = np.linalg.norm(x, ord=1)
        return fidelity + sparsity_weight * sparsity

    # Initial guess: zeros
    x0 = np.zeros_like(coeffs_flat)
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})
    x_recovered = result.x
    coeffs_recovered = pywt.array_to_coeffs(x_recovered, slices, wavelet)
    return coeffs_recovered

# 
def resample_signal(signal, previous_sample_rate, new_sample_rate=600, get_t=0, plusone=0):
    num_samples = int(len(signal) * new_sample_rate / previous_sample_rate)
    resampled_signal = resample(signal, num_samples)
    if get_t == 0:
        return resampled_signal
    else:
        t_rs = np.arange(num_samples+plusone)/new_sample_rate
        return resampled_signal, t_rs

def sparse_fm_demodulation(signal, fs, carrier_freq, if_freq, resample_rate=600):
    # Step 1: Superheterodyne approach for FM demodulation
    t = np.arange(len(signal)) / fs
    local_oscillator = np.cos(2 * np.pi * (carrier_freq - if_freq) * t)
    mixed_signal = signal * local_oscillator
    if_bandwidth = 2e3
    filtered_if_signal = bandpass_filter(mixed_signal, if_freq - if_bandwidth/2, if_freq + if_bandwidth/2, fs)
    analytic_signal = hilbert(filtered_if_signal)
    demodulated_signal = np.diff(np.unwrap(np.angle(analytic_signal)))

    # Step 2: Bandpass filter demodulated signal to 0.1–40 Hz
    baseband_signal = bandpass_filter(demodulated_signal, 0.1, 40, fs)

    # Step 3: Resample to 600 Hz
    # num_samples = int(len(baseband_signal) * resample_rate / fs)
    # resampled_signal = resample(baseband_signal, num_samples)
    # t_rs = np.arange(num_samples)/resample_rate
    resampled_signal, t_rs = resample_signal(baseband_signal, fs, new_sample_rate=resample_rate, get_t=1)

    # Step 4: Iterative compressed sensing reconstruction
    recovered_coeffs, slices = iterative_compressed_sensing(resampled_signal)
    sparse_signal = inverse_wavelet_transform(recovered_coeffs, slices)
    return sparse_signal, t_rs

def quadrature_demodulation(signal, fs, carrier_freq):
    """
    Perform FM demodulation using quadrature (I/Q) method.
    """
    t = np.arange(len(signal)) / fs
    i_signal = signal * np.cos(2 * np.pi * carrier_freq * t)
    q_signal = signal * np.sin(2 * np.pi * carrier_freq * t)

    # Low-pass filter the I and Q components
    i_filtered = lowpass_filter(i_signal, carrier_freq / 2, fs)
    q_filtered = lowpass_filter(q_signal, carrier_freq / 2, fs)

    # Compute instantaneous frequency
    instantaneous_phase = np.unwrap(np.arctan2(q_filtered, i_filtered))
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0 * np.pi)

    return instantaneous_frequency

def zero_crossing_demodulation(signal, fs):
    """
    Perform FM demodulation using zero-crossing intervals.
    """
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    time_intervals = np.diff(zero_crossings) / fs
    frequency_estimates = 1.0 / time_intervals

    # Resample to match the original signal length
    demodulated_signal = np.interp(np.arange(len(signal)), zero_crossings[:-1], frequency_estimates)
    return demodulated_signal

def time_binned_fft_demodulation(signal, fs, carrier_freq, fft_window=1200, carrier_bandwidth=1000):
    """
    Demodulate FM signal using time-binned FFT to track the peak frequency.
    
    Parameters:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency of the signal.
        carrier_freq (float): Carrier frequency in Hz.
        fft_window (int): Number of samples per FFT window (default: 1200, ~1/40 sec at 48 kHz).
        carrier_bandwidth (float): Bandwidth around carrier to search for peak (default: 1000 Hz).

    Returns:
        np.ndarray: Demodulated signal (instantaneous frequency).
        np.ndarray: Time vector corresponding to the demodulated signal.
    """
    step_size = fft_window // 2  # Overlap by 50%
    num_bins = (len(signal) - fft_window) // step_size + 1
    time = np.arange(num_bins) * step_size / fs
    demodulated_signal = []

    for i in range(num_bins):
        start = i * step_size
        end = start + fft_window
        segment = signal[start:end]

        # Perform FFT
        spectrum = np.fft.fft(segment * np.hanning(len(segment)))
        freqs = np.fft.fftfreq(len(segment), d=1/fs)

        # Focus on positive frequencies in the carrier band
        carrier_range = (freqs >= (carrier_freq - carrier_bandwidth / 2)) & (freqs <= (carrier_freq + carrier_bandwidth / 2))
        freqs_in_band = freqs[carrier_range]
        spectrum_in_band = np.abs(spectrum[carrier_range])

        # Find the peak frequency in the band
        peak_idx = np.argmax(spectrum_in_band)
        peak_freq = freqs_in_band[peak_idx]

        # Convert peak frequency to instantaneous frequency deviation
        demodulated_signal.append(peak_freq - carrier_freq)

    return np.array(demodulated_signal), time


def spectrogram_demodulation(signal, fs, carrier_freq, carrier_bandwidth=1000, nperseg=1024, noverlap=512):
    """
    Demodulate FM signal using a spectrogram to track the peak frequency in the carrier band.
    
    Parameters:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency of the signal.
        carrier_freq (float): Carrier frequency in Hz.
        carrier_bandwidth (float): Bandwidth around carrier to search for peak (default: 1000 Hz).
        nperseg (int): Number of samples per spectrogram segment (default: 1024).
        noverlap (int): Number of overlapping samples (default: 512).
    
    Returns:
        np.ndarray: Demodulated signal (instantaneous frequency).
        np.ndarray: Time vector corresponding to the demodulated signal.
    """
    from scipy.signal import spectrogram

    # Compute the spectrogram
    freqs, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Focus on the carrier band
    carrier_range = (freqs >= (carrier_freq - carrier_bandwidth / 2)) & (freqs <= (carrier_freq + carrier_bandwidth / 2))
    freqs_in_band = freqs[carrier_range]
    Sxx_in_band = Sxx[carrier_range, :]

    # Trace the peak frequency over time
    peak_indices = np.argmax(Sxx_in_band, axis=0)
    peak_freqs = freqs_in_band[peak_indices]

    # Convert peak frequency to instantaneous frequency deviation
    demodulated_signal = peak_freqs - carrier_freq

    return demodulated_signal, times


# --- Sound Sources ---



# --- Signal Recovery Quality ---
def cross_correlation_metric(original, recovered):
    """Compute cross-correlation to handle time delays."""
    print('original')
    print(original.shape)
    print('recovered')
    print(recovered.shape)
    correlation = correlate(recovered, original, mode='full')
    max_corr = np.max(correlation)
    return max_corr / (np.linalg.norm(original) * np.linalg.norm(recovered))


def normalized_mean_squared_error(original, recovered):
    """Compute normalized mean squared error (NMSE)."""
    original = original[:len(recovered)]
    mse = np.mean((original - recovered) ** 2)
    return mse / np.mean(original ** 2)


# --- Simulation and Testing ---
def run_simulation(fs, duration, signal_freq, mic_locs, signal_loc, noise_sources=None, resample_rate=600):
    """Run a full simulation and test signal recovery."""
    # Generate the main signal
    signal_type = 1
    if signal_type == 0:
        main_signal = generate_signal(fs, duration, freq=signal_freq)
    elif signal_type == 1:
        main_signal,_ = generate_ecg_signal(duration, 75, fs)

    main_signal = main_signal/np.max(main_signal)*5
    # Filter to remove spikes
    #main_signal = bandpass_filter(main_signal, FILTER_LOW, FILTER_HIGH, fs)

    main_signal_rs, t_rs = resample_signal(main_signal, fs, new_sample_rate=resample_rate, get_t=1)
    modulated_signal = fm_modulate(main_signal, CARRIER_FREQ, fs, CALIBRATION)
    # Propagate the main signal to the microphones
    propagated_signals = propagate_sound(modulated_signal, mic_locs, signal_loc, fs)

    # Generate and propagate noise signals
    if noise_sources is not None:
        for noise_func, noise_loc, noise_gain in noise_sources:
            noise = noise_func(fs, duration)
            noise_propagated = propagate_sound(noise, mic_locs, noise_loc, fs, gain=noise_gain)
            propagated_signals += noise_propagated

    # Mix signals from all microphones
    # TODO handle multiple microphones. At the moment this doesn't really make sense.
    mixed_signal = mix_signals(propagated_signals)

    # Apply demodulation and sparse recovery

    method = 4
    
    if method == 0:
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        demodulated_signal = fm_demodulate(filtered_data, fs, CARRIER_FREQ)
        ecg_signal = demodulated_signal / CALIBRATION
        # ecg_signal = frequency_domain_filter(ecg_signal, fs, method='fixed')
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)

    elif method == 1:
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        demodulated_signal = superheterodyne_demodulate(filtered_data, fs, carrier_freq=19000, if_freq=2000)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)
        
    elif method == 2:
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        recovered_signal, _ = sparse_fm_demodulation(filtered_data, fs, carrier_freq=19000, if_freq=2000, resample_rate=resample_rate)
        ecg_filtered = bandpass_filter(recovered_signal, FILTER_LOW, FILTER_HIGH, fs)
        _, t_rs2 = resample_signal(ecg_filtered, resample_rate, new_sample_rate=resample_rate, get_t=1, plusone=0)

    elif method == 3:
        print('Quadrature demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        demodulated_signal = quadrature_demodulation(filtered_data, fs, CARRIER_FREQ)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)

    elif method == 4:
        print('Zero crossing demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        demodulated_signal = zero_crossing_demodulation(filtered_data, fs)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)

    elif method == 5:
        print('Fourier window demodulation')
        time_binned_fft_demodulation(signal, fs, carrier_freq, fft_window=1200, carrier_bandwidth=1000)


    # Evaluate signal recovery quality
    cross_corr = cross_correlation_metric(main_signal_rs, recovered_signal)
    cross_corr_end = cross_correlation_metric(main_signal_rs[t_rs>15], recovered_signal[t_rs>15])
    nmse = normalized_mean_squared_error(main_signal_rs, recovered_signal)
    print(t_rs.shape, main_signal_rs.shape, mixed_signal.shape, t_rs.shape, recovered_signal.shape)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Original Signal")
    plt.plot(t_rs, main_signal_rs)
    #plt.ylim([-np.max(main_signal_rs),np.max(main_signal_rs)])
    plt.ylim([-3,6])
    plt.xlim([15,20])
    #plt.subplot(3, 1, 2)
    #plt.title("Mixed Signal at Microphones")
    #plt.plot(mixed_signal[:fs])
    plt.subplot(2, 1, 2)
    plt.title("Recovered Signal")
    plt.plot(t_rs2, recovered_signal)
    #plt.ylim([-np.max(main_signal_rs),np.max(main_signal_rs)])
    plt.ylim([-3,6])
    plt.xlim([15,20])
    plt.xlabel("cross corr total: "+str(np.round(cross_corr,decimals=4))+ " Cross corr end: "+str(np.round(cross_corr_end,decimals=4)))
    plt.tight_layout()
    plt.show()

    print(f"Cross-Correlation: {cross_corr:.3f}")
    print(f"Normalized MSE: {nmse:.3f}")

# Example Simulation
fs = 48000
duration = 20  # seconds
signal_freq = 10
mic_locs = np.array([[0, 0], [0, 0.1]])  # 4 microphones
mic_locs = np.array([[0, 0]])  # 4 microphones
signal_loc = np.array([0, -0.01])  # Signal source location
noise_sources = [
    (generate_mains_hum, np.array([1, 1.5]), 0.5),
    (generate_gwn, np.array([0.2, 0.8]), 0.8),
    (generate_pink_noise, np.array([0.7, 0.3]), 0.6),
    (generate_broad_spectrum_noise, np.array([1.2, 0.7]), 0.7)
]

run_simulation(fs, duration, signal_freq, mic_locs, signal_loc, noise_sources=noise_sources)
