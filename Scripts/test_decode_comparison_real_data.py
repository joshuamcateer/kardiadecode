import numpy as np
from pydub import AudioSegment
from scipy.signal import correlate, butter, lfilter, hilbert, resample
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import pywt
from scipy.optimize import minimize

from kardia_singal_functions import generate_ecg_signal, fm_modulate, butter_bandpass, bandpass_filter, lowpass_filter, frequency_domain_filter
from signal_generation import *
from scipy.fft import fft, ifft, fftfreq
from scipy.fft import dct, idct

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
def resample_signal_old(signal, previous_sample_rate, new_sample_rate=600, get_t=0, plusone=0):
    num_samples = int(len(signal) * new_sample_rate / previous_sample_rate)
    resampled_signal = resample(signal, num_samples)
    if get_t == 0:
        return resampled_signal
    else:
        t_rs = np.arange(num_samples+plusone)/new_sample_rate
        return resampled_signal, t_rs
    

def resample_signal_rate(signal, fs_old, new_sample_rate, get_t=0, plusone=0):
    """
    Resample the signal to a new sample rate.
    """
    num_samples = int(len(signal) * new_sample_rate / fs_old)
    resampled_signal = resample(signal, num_samples)

    if get_t:
        t = np.linspace(0, len(signal) / fs_old, len(resampled_signal))
        if plusone:
            t = np.append(t, t[-1] + (t[1] - t[0]))
        return resampled_signal, t

    return resampled_signal


def denoise_old(signal, wavelet='db4', threshold=0.04, sorh='s'):
    sorh = sorh.lower()
    if sorh == 's':
        sorh = 'soft'
    else:
        sorh = 'hard'
    coeffs = pywt.wavedec(signal, wavelet)
    coeffs[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

import numpy as np
import pywt


def denoise(signal, wavelet='db4', threshold=None, sorh='soft', n_shifts=10):
    """
    Denoise a signal using wavelet thresholding with optional cycle spinning.
    
    Args:
        signal (np.ndarray): Input noisy signal.
        wavelet (str): Type of wavelet to use (default: 'db4').
        threshold (float or None): Threshold for wavelet coefficients (default: None, adaptive).
        sorh (str): 'soft' or 'hard' thresholding (default: 'soft').
        cycle_spinning (bool): Apply cycle spinning to improve shift invariance (default: False).
        n_shifts (int): Number of shifts for cycle spinning (default: 10).

    Returns:
        np.ndarray: Denoised signal.
    """
    # Ensure valid thresholding mode
    sorh = sorh.lower()

    if sorh == 's':
        sorh = 'soft'
    elif sorh == 'h':
        sorh = 'hard'

    if sorh not in ['soft', 'hard']:
        raise ValueError("Thresholding mode must be 's', 'soft' or 'h', 'hard'.")

    # Estimate noise level if threshold is not provided
    def estimate_noise(coeffs):
        # Use the detail coefficients at the first level to estimate noise
        median = np.median(np.abs(coeffs[-1]))
        sigma = median / 0.6745  # Robust estimate of sigma (MAD)
        return sigma

    def wavelet_denoise(sig, thr):
        """Performs standard wavelet denoising."""
        coeffs = pywt.wavedec(sig, wavelet)

        # Estimate threshold adaptively if not provided
        
        sigma = estimate_noise(coeffs)
        thr = sigma * np.sqrt(2 * np.log(len(signal)))*thr  # Universal threshold

        # Apply thresholding to detail coefficients
        coeffs[1:] = [pywt.threshold(c, value=thr, mode=sorh) for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)

    if n_shifts == 0:
        # No cycle spinning: Apply wavelet denoising directly
        return wavelet_denoise(signal, threshold)

    # With cycle spinning: Average denoised outputs across shifts
    denoised_signals = []
    n = len(signal)

    for shift in range(n_shifts):
        # Circularly shift the signal
        shifted_signal = np.roll(signal, shift)
        # Denoise the shifted signal
        denoised_shifted = wavelet_denoise(shifted_signal, threshold)
        # Reverse the shift
        denoised_signals.append(np.roll(denoised_shifted, -shift))

    # Average across all shifted versions
    denoised_signal = np.mean(denoised_signals, axis=0)
    return denoised_signal



def resample_signal(signal, t_signal, t_new):
    """
    Resample a signal to match a new time vector.

    Args:
        signal (np.ndarray): Input signal sampled at times `t_signal`.
        t_signal (np.ndarray): Original time vector for the input signal.
        t_new (np.ndarray): Desired time vector for the resampled signal.

    Returns:
        np.ndarray: Resampled signal at times `t_new`.
    """
    # Interpolation function based on the original time and signal
    interpolator = interp1d(t_signal, signal, kind='linear', fill_value='extrapolate')

    # Evaluate the interpolator at the new time points
    resampled_signal = interpolator(t_new)

    return resampled_signal

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


def pll_fm_demodulator_old(signal, fs, carrier_freq, loop_bandwidth=50):
    dt = 1 / fs
    n_samples = len(signal)
    vco_phase = 0.0
    vco_freq = carrier_freq
    loop_filter_output = 0.0

    # PLL loop parameters
    vco_sensitivity = 2 * np.pi * loop_bandwidth  # Gain of the VCO (rad/s)
    cutoff = loop_bandwidth  # Low-pass filter cutoff

    # Demodulated signal
    
    demodulated = np.zeros(n_samples)
    time = np.arange(n_samples) * dt  # Time vector

    for i in range(1, len(signal)):
        # Generate VCO signal
        vco_output = np.sin(vco_phase)

        # Multiply input with VCO to get phase error
        phase_error = signal[i] * vco_output

        # Apply low-pass filter to the phase error
        #loop_filter_output = butter_lowpass_filter([phase_error], cutoff, fs)[-1]
        loop_filter_output = lowpass_filter([phase_error], cutoff, fs, order=4)[-1]
        # Update VCO frequency
        vco_freq = carrier_freq + vco_sensitivity * loop_filter_output

        # Update VCO phase
        vco_phase += vco_freq * dt
        vco_phase %= 2 * np.pi

        # Demodulated signal is the loop filter output
        demodulated[i] = loop_filter_output

    return demodulated, time

from scipy.signal import butter, sosfilt, sosfiltfilt

def pll_fm_demodulator(signal, fs, carrier_freq, loop_bandwidth=50, order=4):
    """
    Phase-Locked Loop (PLL) FM demodulator using sosfilt (stateful filtering).

    Args:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency (Hz).
        carrier_freq (float): Carrier frequency (Hz).
        loop_bandwidth (float): PLL loop bandwidth (Hz, default: 50).
        order (int): Filter order for the loop filter (default: 4).

    Returns:
        demodulated (np.ndarray): Demodulated signal (instantaneous frequency deviation).
        time (np.ndarray): Time vector corresponding to the demodulated signal.
    """
    # Time step
    dt = 1 / fs
    n_samples = len(signal)

    # Initialize PLL variables
    vco_phase = 0.0
    vco_freq = carrier_freq
    vco_sensitivity = 2 * np.pi * loop_bandwidth  # VCO sensitivity (rad/s)

    # Design a low-pass filter (sos form)
    nyquist = 0.5 * fs
    normal_cutoff = loop_bandwidth / nyquist
    sos = butter(order, normal_cutoff, btype='low', output='sos')

    # Initialize filter state for sosfilt
    filter_state = np.zeros((sos.shape[0], 2))

    # Outputs
    demodulated = np.zeros(n_samples)
    time = np.arange(n_samples) * dt  # Time vector

    # PLL Loop
    for i in range(n_samples):
        # Generate VCO signal
        vco_output = np.sin(vco_phase)

        # Multiply input with VCO (phase detection)
        phase_error = signal[i] * vco_output

        # Apply low-pass filter to phase error (stateful filtering)
        phase_error_filtered, filter_state = sosfilt(sos, [phase_error], zi=filter_state)

        # Update VCO frequency based on filtered phase error
        loop_filter_output = phase_error_filtered[0]
        vco_freq = carrier_freq + vco_sensitivity * loop_filter_output

        # Update VCO phase
        vco_phase += vco_freq * dt
        vco_phase %= 2 * np.pi  # Keep phase within [0, 2π]

        # Store demodulated output
        demodulated[i] = loop_filter_output

    return demodulated, time

import numpy as np
from scipy.signal import find_peaks

def time_binned_fft_demodulation_with_band(signal, fs, carrier_freq, fft_window=1200, carrier_bandwidth=1000, threshold=0.5):
    """
    Demodulate FM signal using time-binned FFT to track a range of frequencies around the peak.

    Args:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency (Hz).
        carrier_freq (float): Carrier frequency (Hz).
        fft_window (int): FFT window size (samples, default: 1200).
        carrier_bandwidth (float): Bandwidth around carrier frequency (Hz).
        threshold (float): Minimum peak height relative to the max peak (default: 40%).

    Returns:
        demodulated_signal (np.ndarray): Demodulated signal based on weighted frequency center.
        time (np.ndarray): Time vector corresponding to the demodulated signal.
    """
    # FFT window step size (overlap)
    step_size = fft_window // 16  # Adjust overlap
    num_bins = (len(signal) - fft_window) // step_size + 1

    # Time vector for FFT windows
    time = np.arange(num_bins) * step_size / fs + (fft_window / (2 * fs))
    demodulated_signal = []
    zero_pad_on = 0

    for i in range(num_bins):
        start = i * step_size
        end = start + fft_window
        segment = signal[start:end]

        # FFT with optional zero padding
        if zero_pad_on == 0:
            spectrum = np.fft.fft(segment * np.kaiser(len(segment), beta=8))
            freqs = np.fft.fftfreq(len(segment), d=1/fs)
        else:
            zero_pad = 4 * fft_window  # Zero-pad to 4x the window size
            #spectrum = np.fft.fft(segment * np.hanning(len(segment)), n=zero_pad)
            spectrum = np.fft.fft(segment * np.kaiser(len(segment), beta=8))
            freqs = np.fft.fftfreq(zero_pad, d=1/fs)

        # Focus on the carrier band
        carrier_range = (freqs >= (carrier_freq - carrier_bandwidth / 2)) & (freqs <= (carrier_freq + carrier_bandwidth / 2))
        freqs_in_band = freqs[carrier_range]
        spectrum_in_band = np.abs(spectrum[carrier_range])

        # Find peaks in the spectrum
        peaks, properties = find_peaks(spectrum_in_band, height=threshold * np.max(spectrum_in_band))
        if len(peaks) == 0:
            # If no peaks are found, fall back to the single max peak
            peak_idx = np.argmax(spectrum_in_band)
            peak_freq = freqs_in_band[peak_idx]
            demodulated_signal.append(peak_freq - carrier_freq)
            continue

        # Extract the frequencies and magnitudes of the identified peaks
        peak_freqs = freqs_in_band[peaks]
        peak_mags = spectrum_in_band[peaks]

        # Fit a quadratic curve (parabola) to the selected peaks
        weights = peak_mags / np.sum(peak_mags)  # Normalize magnitudes as weights
        weighted_freq = np.sum(weights * peak_freqs)  # Weighted average frequency

        # Instantaneous frequency deviation
        demodulated_signal.append(weighted_freq - carrier_freq)

    # Ensure output is a NumPy array
    demodulated_signal = np.array(demodulated_signal)

    return demodulated_signal, time




def time_binned_fft_demodulation(signal, fs, carrier_freq, fft_window=1200, carrier_bandwidth=1000):
    """
    Demodulate FM signal using time-binned FFT to track the peak frequency.
    """
    step_size = fft_window // 32  # Overlap by 50%
    num_bins = (len(signal) - fft_window) // step_size + 1

    # Corrected time vector for center of each FFT window
    time = np.arange(num_bins) * step_size / fs + (fft_window / (2 * fs))
    demodulated_signal = []
    zero_pad_on = 0 
    for i in range(num_bins):
        start = i * step_size
        end = start + fft_window
        segment = signal[start:end]

        # Perform FFT
        # spectrum = np.fft.fft(segment * np.hanning(len(segment)))
        if zero_pad_on == 0:
            spectrum = np.fft.fft(segment * np.kaiser(len(segment), beta=8))
            freqs = np.fft.fftfreq(len(segment), d=1/fs)
        else:
            zero_pad = 4 * fft_window  # Zero-pad to 4x the window size
            spectrum = np.fft.fft(segment * np.hanning(len(segment)), n=zero_pad)
            freqs = np.fft.fftfreq(zero_pad, d=1/fs)

        # Focus on the carrier band
        carrier_range = (freqs >= (carrier_freq - carrier_bandwidth / 2)) & (freqs <= (carrier_freq + carrier_bandwidth / 2))
        freqs_in_band = freqs[carrier_range]
        spectrum_in_band = np.abs(spectrum[carrier_range])

        # Find the peak frequency
        peak_idx = np.argmax(spectrum_in_band)
        peak_freq = freqs_in_band[peak_idx]

        # Instantaneous frequency deviation
        demodulated_signal.append(peak_freq - carrier_freq)

    # Ensure output is a NumPy array
    demodulated_signal = np.array(demodulated_signal)

    return demodulated_signal, time



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

import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

from scipy.signal import stft


def multi_resolution_stft(signal, fs, windows=[128, 512, 2048], overlaps=[64, 256, 1024]):
    """
    Compute multi-resolution STFT with multiple window sizes and overlaps.

    Args:
        signal (np.ndarray): Input signal (1D array).
        fs (float): Sampling frequency (Hz).
        windows (list): List of window sizes for STFT (default: [128, 512, 2048]).
        overlaps (list): List of overlap sizes corresponding to each window (default: [64, 256, 1024]).

    Returns:
        freqs (list): List of frequency arrays for each resolution.
        times (list): List of time arrays for each resolution.
        Sxx (list): List of magnitude arrays (|STFT|) for each resolution.
    """
    # Validate input dimensions
    if len(windows) != len(overlaps):
        raise ValueError("The number of windows must match the number of overlaps.")

    # Initialize outputs
    freqs = []
    times = []
    Sxx = []

    # Compute STFT for each window/overlap configuration
    for nperseg, noverlap in zip(windows, overlaps):
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs.append(f)
        times.append(t)
        Sxx.append(np.abs(Zxx))  # Use magnitude of STFT

    return freqs, times, Sxx

from scipy.signal import stft
from scipy.interpolate import interp2d

def weighted_multi_resolution_stft(signal, fs, windows=[64, 128, 512, 2048, 4096], overlaps=[32, 64, 256, 1024, 2048], weights=None):
    """
    Compute a multi-resolution STFT with weighted averaging across resolutions, using interpolation to align grids.

    Args:
        signal (np.ndarray): Input signal (1D array).
        fs (float): Sampling frequency (Hz).
        windows (list): List of window sizes for STFT (default: [128, 512, 2048]).
        overlaps (list): List of overlap sizes corresponding to each window.
        weights (list): Optional weights for each resolution (default: None, equal weights).

    Returns:
        freqs (np.ndarray): Common frequency bins.
        times (np.ndarray): Common time bins.
        Sxx (np.ndarray): Weighted average spectrogram.
    """
    # Validate input
    if len(windows) != len(overlaps):
        raise ValueError("Number of windows must match the number of overlaps.")
    if weights is not None and len(weights) != len(windows):
        raise ValueError("Number of weights must match number of windows.")
    
    # Default weights: equal weighting
    if weights is None:
        weights = np.ones(len(windows)) / len(windows)
    else:
        weights = np.array(weights) / np.sum(weights)  # Normalize weights to sum to 1

    # Initialize variables for interpolation
    all_freqs = []
    all_times = []
    all_Sxx = []

    # Compute STFT for each window/overlap pair
    max_freqs = 0
    max_times = 0
    for nperseg, noverlap in zip(windows, overlaps):
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        max_times = np.max([max_times,len(t)])
        max_freqs = np.max([max_freqs,len(f)])
        Sxx = np.abs(Zxx)  # Magnitude spectrum
        all_freqs.append(f)
        all_times.append(t)
        all_Sxx.append(Sxx)

    # Define common frequency and time grid
    max_freq = min([np.max(f) for f in all_freqs])  # Use the smallest max frequency
    min_freq = max([np.min(f) for f in all_freqs])  # Use the largest min frequency
    common_freqs = np.linspace(min_freq, max_freq, max_freqs)  # Uniform frequency bins

    max_time = max([np.max(t) for t in all_times])
    min_time = min([np.min(t) for t in all_times])
    common_times = np.linspace(min_time, max_time, max_times)  # Uniform time bins

    # Interpolate each spectrogram onto the common grid
    interpolated_Sxx = []
    for i in range(len(all_Sxx)):
        interp_func = interp2d(all_times[i], all_freqs[i], all_Sxx[i], kind='linear')
        Sxx_interp = interp_func(common_times, common_freqs)
        interpolated_Sxx.append(Sxx_interp)

    # Weighted average of interpolated spectrograms
    avg_Sxx = np.zeros_like(interpolated_Sxx[0])
    for i in range(len(interpolated_Sxx)):
        avg_Sxx += weights[i] * interpolated_Sxx[i]

    return common_freqs, common_times, avg_Sxx


def spectrogram_demodulation_with_particles(signal, fs, carrier_freq, carrier_bandwidth=1000,
                                            nperseg=4096, noverlap=512, 
                                            n_particles=500, smoothness_weight=0.1, 
                                            dynamics_std=5, plot_results=True):
    """
    Demodulate FM signal using a spectrogram and particle filtering to find the smoothest path.

    Args:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency of the signal.
        carrier_freq (float): Carrier frequency in Hz.
        carrier_bandwidth (float): Bandwidth around carrier to search for peak (default: 1000 Hz).
        nperseg (int): Number of samples per spectrogram segment (default: 1024).
        noverlap (int): Number of overlapping samples (default: 512).
        n_particles (int): Number of particles for dynamic smoothing.
        smoothness_weight (float): Weight for penalizing changes in frequency.
        dynamics_std (float): Standard deviation of particle dynamics (random motion).
        plot_results (bool): Whether to visualize the spectrogram and results.

    Returns:
        np.ndarray: Demodulated signal (instantaneous frequency).
        np.ndarray: Time vector corresponding to the demodulated signal.
    """
    # Compute the spectrogram
    print(signal.shape)
    
    #freqs, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freqs, times, Sxx = weighted_multi_resolution_stft(signal, fs, windows=[128, 512, 2048], overlaps=[64, 256, 1024])
    # Focus on the carrier band
    carrier_range = (freqs >= (carrier_freq - carrier_bandwidth / 2)) & (freqs <= (carrier_freq + carrier_bandwidth / 2))
    freqs_in_band = freqs[carrier_range]
    Sxx_in_band = Sxx[carrier_range, :]

    # Initialize particles
    n_freqs = len(freqs_in_band)
    particles = np.random.choice(n_freqs, size=(n_particles, len(times)))
    weights = np.ones((n_particles, len(times))) / n_particles  # Uniform weights initially

    # Particle filtering
    for t in range(1, len(times)):
        # Predict particle positions based on random dynamics
        particles[:, t] = particles[:, t - 1] + np.random.normal(0, dynamics_std, size=n_particles)
        particles[:, t] = np.clip(particles[:, t], 0, n_freqs - 1)  # Keep particles in bounds

        # Evaluate weights based on spectrogram intensity and smoothness
        for i in range(n_particles):
            freq_idx = int(particles[i, t])
            prev_idx = int(particles[i, t - 1])
            power = Sxx_in_band[freq_idx, t]
            smoothness_penalty = smoothness_weight * abs(freq_idx - prev_idx)
            weights[i, t] = power - smoothness_penalty

        # Normalize weights
        weights[:, t] = np.exp(weights[:, t] - np.max(weights[:, t]))  # Avoid numerical instability
        weights[:, t] /= np.sum(weights[:, t])  # Normalize

        # Resample particles
        resample_idx = np.random.choice(n_particles, size=n_particles, p=weights[:, t])
        particles[:, t] = particles[resample_idx, t]

    # Compute the smoothest path
    best_path = np.zeros(len(times))
    for t in range(len(times)):
        best_path[t] = np.mean(freqs_in_band[particles[:, t].astype(int)])  # Weighted mean

    # Demodulated signal
    demodulated_signal = best_path - carrier_freq

    # Plot results if required
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs_in_band, 10 * np.log10(Sxx_in_band), shading='gouraud')
        plt.colorbar(label='Power (dB)')
        #plt.plot(times, best_path, color='red', label='Tracked Path')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram and Particle Filter Tracking')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs_in_band, 10 * np.log10(Sxx_in_band), shading='gouraud')
        plt.colorbar(label='Power (dB)')
        plt.plot(times, best_path, color='red', label='Tracked Path')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectrogram and Particle Filter Tracking')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return demodulated_signal, times

### --- Step 1: Spectrogram Calculation --- ###
def compute_spectrogram(signal, fs, fft_window=1200, overlap=600, nfft=2048*4):
    """
    Computes the spectrogram of the input signal.

    Args:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency (Hz).
        fft_window (int): FFT window size in samples.
        overlap (int): Overlap size in samples.
        nfft (int): Number of FFT points (zero-padding).

    Returns:
        Sxx (np.ndarray): Magnitude spectrogram (frequency x time).
        freqs (np.ndarray): Frequency axis.
        times (np.ndarray): Time axis.
    """
    freqs, times, Sxx = spectrogram(
        signal, fs=fs, nperseg=fft_window, noverlap=overlap, nfft=nfft, mode='magnitude'
    )

    plt.figure(figsize=(10, 6))
    #plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
    plt.pcolormesh(times, freqs, Sxx, shading='gouraud')
    plt.plot([times[0], times[-1]],[18e3,18e3],'r:')
    plt.plot([times[0], times[-1]],[20e3,20e3],'r:')
    plt.ylim([17.5e3, 20.5e3])
    plt.colorbar(label='Power (dB)')
    #plt.plot(times, best_path, color='red', label='Tracked Path')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram and Particle Filter Tracking')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Sxx, freqs, times


from scipy.ndimage import gaussian_filter
def optimize_demodulation(Sxx, freqs, alpha=1e-0, beta=1.0e4, max_iter=500, tol=1e-6):
    """
    Optimizes the demodulation curve using smoothness and spectrogram intensity.

    Args:
        Sxx (np.ndarray): Spectrogram data (frequency x time).
        freqs (np.ndarray): Frequency axis.
        alpha (float): Regularization strength for smoothness (TV penalty).
        beta (float): Weight for spectrogram intensity.
        max_iter (int): Maximum iterations for optimization.
        tol (float): Convergence tolerance.

    Returns:
        opt_freqs (np.ndarray): Optimized frequency curve.
    """


    Sxx = gaussian_filter(Sxx, sigma=[2, 1])
    n_time = Sxx.shape[1]
    n_freqs = len(freqs)
    step_size = 0.0001
    gamma = (1e1)*0

    # Normalize spectrogram to treat it as probabilities
    Sxx = Sxx / np.max(Sxx)

    # Initial frequency estimates (use max magnitude frequencies as a starting guess)
    initial_freqs = freqs[np.argmax(Sxx, axis=0)]

    # Optimization variable
    x = initial_freqs.copy()

    # Define TV regularizer
    def tv_regularizer(x):
        return np.sum(np.abs(np.diff(x)))
    
    def tv_regularizer2(x):
        return np.sum(np.abs(np.diff(np.diff(x))))

    # Cost function
    def cost_function(x):
        # Spectrogram intensity term (maximize magnitude)
        intensity_term = -np.sum(Sxx[np.searchsorted(freqs, x), np.arange(n_time)])

        # Smoothness term (TV regularizer)
        smoothness_term = alpha * tv_regularizer(x)
        # Smoothness term (TV regularizer)
        smoothness_term2 = alpha * tv_regularizer2(x)*1e6

        # Total cost
        return intensity_term + smoothness_term + smoothness_term2

    # Gradient computation
    def gradient(x):
        grad = np.zeros_like(x)

        # Spectrogram gradient
        for t in range(n_time):
            idx = np.searchsorted(freqs, x[t])
            grad[t] -= beta * Sxx[idx, t]

        # Smoothness gradient (finite differences)
        grad[1:-1] += alpha * np.sign(x[1:-1] - x[0:-2]) - alpha * np.sign(x[2:] - x[1:-1])

        # Second-order TV gradient
        grad[2:-2] += gamma * (
            np.sign(x[1:-3] - 2 * x[2:-2] + x[3:-1]) -
            np.sign(x[2:-2] - 2 * x[3:-1] + x[4:])
        )

        return grad

    # FISTA Optimization
    t = 1.0
    z = x.copy()
    for iteration in range(max_iter):
        # Compute gradient
        grad = gradient(z)

        # Gradient step
        x_new = z - step_size * grad  # Fixed step size (can be tuned)

        # Projection step: Ensure x_new is within valid frequencies
        x_new = np.clip(x_new, freqs[0], freqs[-1])

        # Update momentum (FISTA)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)

        # Projection step: Ensure x_new is within valid frequencies
        z = np.clip(z, freqs[0], freqs[-1])

        # Check for convergence
        if np.linalg.norm(x_new - x) / np.linalg.norm(x) < tol:
            break

        # Update variables
        x, t = x_new, t_new

    return x
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

def do_demodulation(mixed_signal, method='fft',  t_rs=None, fs=48000, resample_rate=600):
    method = method.lower()
    if method == 0 or method == 'hilbert':
        print('Hilbert demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        demodulated_signal = fm_demodulate(filtered_data, fs, CARRIER_FREQ)
        ecg_signal = demodulated_signal / CALIBRATION
        # ecg_signal = frequency_domain_filter(ecg_signal, fs, method='fixed')
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)
        recovered_signal = resample_signal(recovered_signal, t_rs2, t_rs)
        t_rs2 = t_rs
        print(len(recovered_signal))
        print(len(t_rs2))
        #recovered_signal, t_rs2 = resample_signal(ecg_signal, t_fft, t_rs)
        #t_rs2 = t_rs

    elif method == 1 or method == 'superhetrodyne' or method == 'hetrodyne':
        print('Superhetrodyne demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        demodulated_signal = superheterodyne_demodulate(filtered_data, fs, carrier_freq=19000, if_freq=2000)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)
        
    elif method == 2 or method == 'cs' or method == 'sparse':
        print('Sparse demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        recovered_signal, _ = sparse_fm_demodulation(filtered_data, fs, carrier_freq=19000, if_freq=2000, resample_rate=resample_rate)
        recovered_signal = bandpass_filter(recovered_signal, FILTER_LOW, FILTER_HIGH, fs)
        _, t_rs2 = resample_signal_rate(recovered_signal, resample_rate, new_sample_rate=resample_rate, get_t=1, plusone=0)
        recovered_signal = resample_signal(recovered_signal, t_rs2, t_rs)
        t_rs2 = t_rs

    elif method == 3 or method == 'quadrature':
        print('Quadrature demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        # print("I don't know why it is upsidedown")
        demodulated_signal = -quadrature_demodulation(filtered_data, fs, CARRIER_FREQ)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)
        recovered_signal = resample_signal(recovered_signal, t_rs2, t_rs)
        t_rs2 = t_rs
    elif method == 4 or method == 'zero crossing':
        print('Zero crossing demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        demodulated_signal = zero_crossing_demodulation(filtered_data, fs)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)

    elif method == 5 or method == 'fft' or method == 'fourier':
        print('Fourier window demodulation')
        #mixed_signal = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs, order=8)

        demodulated_signal, t_fft = time_binned_fft_demodulation(mixed_signal, fs, CARRIER_FREQ, fft_window=1200, carrier_bandwidth=1000)
        fs_fft = len(t_fft)/(t_fft[-1]-t_fft[0])
        print(demodulated_signal.shape)
        print(t_fft.shape)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_fft, t_rs)
        t_rs2 = t_rs
        #recovered_signal, t_rs2 = resample_signal(ecg_signal, fs_fft, new_sample_rate=resample_rate, get_t=1, plusone=0)
    elif method == 6 or method == 'fft_band' or method == 'fourier_band':
        print('Fourier window demodulation')
        #mixed_signal = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs, order=8)

        demodulated_signal, t_fft = time_binned_fft_demodulation_with_band(mixed_signal, fs, CARRIER_FREQ, fft_window=1200, carrier_bandwidth=1000*2)
        fs_fft = len(t_fft)/(t_fft[-1]-t_fft[0])
        print(demodulated_signal.shape)
        print(t_fft.shape)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_fft, t_rs)
        t_rs2 = t_rs
        #recovered_signal, t_rs2 = resample_signal(ecg_signal, fs_fft, new_sample_rate=resample_rate, get_t=1, plusone=0)
    elif method == 699999 or method == 'pll' or method == 'phase locked loop':
        print('This method doesnt work')
        print('Phase locked loop demodulation')
        # TODO finish this
        demodulated_signal, t_pll = pll_fm_demodulator(mixed_signal, fs, CARRIER_FREQ, loop_bandwidth=50)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_pll, t_rs)
        t_rs2 = t_rs
    elif method == 8 or method == 'fourier follower' or method == 'follower' or method == 'follow':
        # demodulated_signal, t_fft, SNR = fft_demodulation_with_prediction(mixed_signal, fs, CARRIER_FREQ, fft_window=1200, 
        #                              carrier_bandwidth=2000, step_size_factor=32, 
        #                              prior_sigma=500, momentum=0.1)
        demodulated_signal, t_fft, SNR = fft_demodulation_with_prediction(mixed_signal, fs, CARRIER_FREQ)#, fft_window=1200, 
                                     #carrier_bandwidth=2000, step_size_factor=32, 
                                     #prior_sigma=500, momentum=0.1)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_fft, t_rs)
        t_rs2 = t_rs
    elif method == 9 or method == 'spectrogram':
        pass
    elif method == 10 or method == 'spectrogram with particles' or method == 'particles':
        demodulated_signal, t_spec = spectrogram_demodulation_with_particles(mixed_signal, fs, CARRIER_FREQ)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_spec, t_rs)
        t_rs2 = t_rs
    elif method == 11 or method == 'spectrogram optimisation':
        print(method)
        Sxx, freqs, t_spec = compute_spectrogram(mixed_signal, fs, fft_window=1200, overlap=600, nfft=2048)
        demodulated_signal = optimize_demodulation(Sxx, freqs)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_spec, t_rs)
        t_rs2 = t_rs
    else:
        assert "warning demodulation method not correctly specified"
    return recovered_signal, t_rs2


# --- Simulation and Testing ---
def run_simulation(fs, duration, signal_freq, mic_locs, signal_loc, noise_sources=None, resample_rate=600):
    """Run a full simulation and test signal recovery."""
    # Generate the main signal
    signal_type = 1
    if signal_type == 0:
        main_signal = generate_signal(fs, duration, freq=signal_freq)
    elif signal_type == 1:
        main_signal,_ = generate_ecg_signal(duration, signal_freq, fs)
        main_signal = bandpass_filter(main_signal, 0.1, 40, fs, order=1)


    main_signal = main_signal/np.max(main_signal)*5
    # Filter to remove spikes
    #main_signal = bandpass_filter(main_signal, FILTER_LOW, FILTER_HIGH, fs)

    main_signal_rs, t_rs = resample_signal_rate(main_signal, fs, new_sample_rate=resample_rate, get_t=1)
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

    method = 'quadrature'
    recovered_signal, t_rs2 = do_demodulation(mixed_signal, method=method,  t_rs=t_rs, fs=fs, resample_rate=resample_rate)

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
    for i in np.arange(-5,6):
        plt.plot([t_rs[0], t_rs[-1]], [i, i],':r', alpha=0.5)
    plt.plot(t_rs, main_signal_rs,'b')
    #plt.ylim([-np.max(main_signal_rs),np.max(main_signal_rs)])
    plt.ylim([-3,6])
    plt.xlim([15,20])
    #plt.subplot(3, 1, 2)
    #plt.title("Mixed Signal at Microphones")
    #plt.plot(mixed_signal[:fs])
    plt.subplot(2, 1, 2)
    plt.title("Recovered Signal")
    for i in np.arange(-5,6):
        plt.plot([t_rs[0], t_rs[-1]], [i, i],':r', alpha=0.5)
    plt.plot(t_rs2, recovered_signal, 'b')
    #plt.ylim([-np.max(main_signal_rs),np.max(main_signal_rs)])
    plt.ylim([-3,6])
    plt.xlim([15,20])
    plt.xlabel("cross corr total: "+str(np.round(cross_corr,decimals=4))+ " Cross corr end: "+str(np.round(cross_corr_end,decimals=4)))
    plt.tight_layout()
    plt.show()

    print(f"Cross-Correlation: {cross_corr:.3f}")
    print(f"Normalized MSE: {nmse:.3f}")

# Stereo Channel Selection
def select_best_channel(data, fs):
    if data.ndim == 2:  # Stereo signal
        channel_energies = []
        for channel in range(data.shape[1]):
            channel_data = data[:, channel]
            bandpassed = bandpass_filter(channel_data, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
            snr = np.sum(np.abs(bandpassed)**2) / np.sum(np.abs(channel_data - bandpassed)**2)
            channel_energies.append(snr)
        print('Channel energies')
        print(channel_energies)
        best_channel = np.argmax(channel_energies)
        print(f"Selected Channel: {best_channel}")
        return data[:, best_channel]
    return data  # Already mono
# Mains Frequency Filtering
def detect_and_filter_mains(signal, fs, harmonics=3):
    sd_signal = np.std(signal)
    print(sd_signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    fft_magnitude = np.abs(np.fft.rfft(signal))
    plt.figure()
    plt.plot(freqs[:int(len(freqs)*0.1)], fft_magnitude[:int(len(freqs)*0.1)])
    plt.show()
    mains_freq = freqs[np.argmax(fft_magnitude[:int(len(freqs)*0.1)])]  # Detect around 50 Hz
    print(f"Detected Mains Frequency: {mains_freq} Hz")
    notch_width = 0.01
    for harmonic in range(1, harmonics + 1):
        min_freq = mains_freq * harmonic *(1-notch_width)
        max_freq = mains_freq * harmonic *(1+notch_width)
        print(min_freq, max_freq)
        signal = bandpass_filter(signal, min_freq, max_freq, fs, order=4)
    print(np.std(signal))
    signal = signal/np.std(signal)*sd_signal
    print(np.std(signal))
    return signal


def frequency_domain_filter_targeted(signal, fs, harmonics=5, notch_width=0.05, target_low =40, target_high = 100, method='fixed', fwhm_factor=2):
    """
    Filters the mains frequency and harmonics in the frequency domain.

    Parameters:
        signal (np.ndarray): Input time-domain signal.
        fs (float): Sampling frequency of the signal.
        harmonics (int): Number of harmonics to filter (default is 5).
        method (str): Filtering method ('fixed' or 'fwhm').
        fwhm_factor (float): Multiplier for FWHM width (default is 2).

    Returns:
        np.ndarray: Filtered time-domain signal.
    """

    fixed_notch_width = notch_width
    # Compute the FFT of the signal
    freqs = fftfreq(len(signal), d=1/fs)
    fft_values = fft(signal)

    # Compute magnitude spectrum
    spectrum = np.abs(fft_values)
    positive_freqs = freqs[freqs >= 0]
    positive_freqs40 = freqs[(freqs >= target_low) & (freqs <= target_high)]
    positive_spectrum = spectrum[freqs >= 0]
    positive_spectrum40 = spectrum[(freqs >= target_low) & (freqs <= target_high)]

    # Detect the mains frequency
    #mains_idx = np.argmax(positive_spectrum[:int(len(positive_freqs) * 0.1)])  # Search below 100 Hz
    mains_idx = np.argmax(positive_spectrum40[:int(len(positive_freqs40))])  # Search below 100 Hz
    #mains_freq = positive_freqs[mains_idx]
    mains_freq = positive_freqs40[mains_idx]
    print(f"Detected Mains Frequency: {mains_freq:.2f} Hz")

    # Create a mask for filtering
    mask = np.ones_like(fft_values, dtype=bool)  # True means keep the frequency

    for harmonic in range(1, harmonics + 1):
        freq = harmonic * mains_freq

        if method == 'fixed':
            # Option 1: Fixed range
            lower_bound = (1-fixed_notch_width) * freq
            upper_bound = (1+fixed_notch_width) * freq
        elif method == 'fwhm':
            # TODO fix this bit which is wrong
            # Option 2: Compute FWHM
            freq_ind = np.nanargmin(np.abs(positive_freqs-freq))
            print('freq_ind', freq_ind)
            #magnitude_of_peak = positive_spectrum[freq_ind]

            #peak_idx = np.argmax(positive_spectrum)
            half_max = positive_spectrum[freq_ind] / 2
            fwhm_range = np.where(positive_spectrum > half_max)[0]
            print('fwhm_range: ', str(fwhm_range))
            fwhm_width = (positive_freqs[fwhm_range[-1]] - positive_freqs[fwhm_range[0]]) * fwhm_factor
            lower_bound = freq - fwhm_width / 2
            upper_bound = freq + fwhm_width / 2
        else:
            raise ValueError("Invalid method. Use 'fixed' or 'fwhm'.")

        # Zero out frequencies in the range
        mask &= ~((freqs >= lower_bound) & (freqs <= upper_bound))
        mask &= ~((freqs <= -lower_bound) & (freqs >= -upper_bound))  # Handle negative frequencies
        print(f"Filtering {freq:.2f} Hz: Range [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Apply the mask
    fft_values[~mask] = 0

    # Transform back to the time domain
    filtered_signal = np.real(ifft(fft_values))
    #plt.figure()
    #plt.plot(freqs[:int(len(freqs)*0.1)], np.absolute(fft_values[:int(len(freqs)*0.1)]))
    #plt.show()
    return filtered_signal

def total_variation_denoise_old(signal, weight=0.1):
    """
    Total Variation (TV) denoising using L1 regularization.
    
    Args:
        signal (np.ndarray): Input signal.
        weight (float): Regularization weight.
    
    Returns:
        np.ndarray: TV denoised signal.
    """
    n = len(signal)

    # Minimize the TV norm using L1 regularization
    def tv_objective(x):
        return np.sum(np.abs(np.diff(x))) + weight * np.sum((x - signal) ** 2)

    # Initial guess
    x0 = signal.copy()

    # Optimization
    result = minimize(tv_objective, x0, method='L-BFGS-B')

    return result.x




def total_variation_denoise_lowfissue(signal, weight=0.1, l1_weight=0.1, max_iter=100, tol=1e-6):
    """
    Total Variation (TV) denoising using Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Args:
        signal (np.ndarray): Input noisy signal.
        weight (float): TV regularization weight (higher -> more smoothing).
        l1_weight (float): L1 sparsity weight (higher -> enforces zeros).
        max_iter (int): Maximum iterations for optimization.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Denoised signal.
    """
    n = len(signal)
    x = np.copy(signal)  # Initialization
    z = np.copy(signal)  # Intermediate variable
    t = 1.0  # Momentum term
    step_size = 1.0  # Step size for gradient descent

    def gradient(y):
        """Gradient of data fidelity term."""
        return y - signal

    def prox_tv(y, step):
        """Proximal operator for Total Variation (TV)."""
        diff = np.diff(y)
        # Apply soft thresholding to the gradient
        shrink = np.sign(diff) * np.maximum(np.abs(diff) - step * weight, 0)
        # Compute cumulative sum to reconstruct denoised signal
        return np.cumsum(np.concatenate(([y[0]], shrink)))

    def prox_l1(y, step):
        """Proximal operator for L1 regularization."""
        return np.sign(y) * np.maximum(np.abs(y) - step * l1_weight, 0)

    # FISTA iterations
    for i in range(max_iter):
        x_old = np.copy(x)

        # Gradient step
        grad = gradient(z)
        x = prox_tv(z - step_size * grad, step_size)

        # Apply sparsity constraint (L1 regularization)
        x = prox_l1(x, step_size)

        # FISTA momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x






def total_variation_denoise(signal, weight=0.001, l1_weight=0.0, low_freq_weight=10, low_freq_cutoff=0.01, max_iter=100, tol=1e-6):
    """
    Total Variation (TV) denoising with FISTA and low-frequency anchoring.

    Args:
        signal (np.ndarray): Input noisy signal.
        weight (float): TV regularization weight.
        l1_weight (float): L1 sparsity weight for zero enforcement.
        low_freq_weight (float): Low-frequency anchoring weight.
        low_freq_cutoff (float): Low-frequency cutoff fraction (default: 0.01).
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Denoised signal.
    """
    # Initialization
    n = len(signal)
    x = np.copy(signal)
    z = np.copy(signal)
    t = 1.0
    step_size = 0.1  # Gradient step size

    # Precompute low-frequency components
    def low_freq_filter(sig, cutoff):
        """Smooth low-pass filtering in DCT domain."""
        dct_sig = dct(sig, norm='ortho')
        freqs = np.linspace(0, 1, len(dct_sig))  # Normalized frequency axis
        low_pass = np.exp(-0.5 * (freqs / cutoff) ** 2)  # Gaussian filter
        dct_sig *= low_pass  # Apply Gaussian filter
        return idct(dct_sig, norm='ortho')

    # Precomputed low-frequency target
    low_freq_signal = low_freq_filter(signal, low_freq_cutoff)

    def gradient(y):
        """Gradient computation for data fidelity and low-frequency anchoring."""
        # Fidelity gradient (data term)
        grad = y - signal

        # Low-frequency anchoring term
        if low_freq_weight > 0:
            # Apply low-frequency filter and enforce similarity
            grad += low_freq_weight * (low_freq_filter(y, low_freq_cutoff) - low_freq_signal)

        return grad

    def prox_tv(y, step):
        """Proximal operator for TV regularization."""
        diff = np.diff(y)
        shrink = np.sign(diff) * np.maximum(np.abs(diff) - step * weight, 0)
        return np.cumsum(np.concatenate(([y[0]], shrink)))

    def prox_l1(y, step):
        """Proximal operator for L1 regularization."""
        return np.sign(y) * np.maximum(np.abs(y) - step * l1_weight, 0)

    # FISTA iterations
    for i in range(max_iter):
        x_old = np.copy(x)

        # Compute gradient
        grad = gradient(z)

        # Apply TV regularization
        x = prox_tv(z - step_size * grad, step_size)

        # Apply sparsity constraint
        x = prox_l1(x, step_size)

        # Momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x


def total_variation_denoise_grad(signal, weight=0.001, l1_weight=0.0, low_freq_weight=10, low_freq_cutoff=0.01, max_iter=100, tol=1e-6):
    """
    Total Variation (TV) denoising with FISTA and low-frequency anchoring.

    Args:
        signal (np.ndarray): Input noisy signal.
        weight (float): TV regularization weight.
        l1_weight (float): L1 sparsity weight for zero enforcement.
        low_freq_weight (float): Low-frequency anchoring weight.
        low_freq_cutoff (float): Low-frequency cutoff fraction (default: 0.01).
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Denoised signal.
    """
    # Initialization
    n = len(signal)
    x = np.copy(signal)
    z = np.copy(signal)
    t = 1.0
    step_size = 0.1  # Gradient step size

    # Precompute low-frequency components
    def low_freq_filter(sig, cutoff):
        """Smooth low-pass filtering in DCT domain."""
        dct_sig = dct(sig, norm='ortho')
        freqs = np.linspace(0, 1, len(dct_sig))  # Normalized frequency axis
        low_pass = np.exp(-0.5 * (freqs / cutoff) ** 2)  # Gaussian filter
        dct_sig *= low_pass  # Apply Gaussian filter
        return idct(dct_sig, norm='ortho')

    # Precomputed low-frequency target
    low_freq_signal = low_freq_filter(signal, low_freq_cutoff)

    def gradient(y):
        """Gradient computation for data fidelity and low-frequency anchoring."""
        # Fidelity gradient (data term)
        grad = y - signal

        # Low-frequency anchoring term
        if low_freq_weight > 0:
            # Apply low-frequency filter and enforce similarity
            grad += low_freq_weight * (low_freq_filter(y, low_freq_cutoff) - low_freq_signal)

        return grad

    def prox_tv(y, step):
        """Proximal operator for TV regularization."""
        diff = np.diff(np.diff(y))
        shrink = np.sign(diff) * np.maximum(np.abs(diff) - step * weight, 0)
        return np.cumsum(np.cumsum(np.concatenate(([y[0]], [y[1]], shrink))))

    def prox_l1(y, step):
        """Proximal operator for L1 regularization."""
        return np.sign(y) * np.maximum(np.abs(y) - step * l1_weight, 0)

    # FISTA iterations
    for i in range(max_iter):
        x_old = np.copy(x)

        # Compute gradient
        grad = gradient(z)

        # Apply TV regularization
        x = prox_tv(z - step_size * grad, step_size)

        # Apply sparsity constraint
        x = prox_l1(x, step_size)

        # Momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x



def fista_smooth_second_derivative(signal, alpha, max_iter=500, tol=1e-6):
    """
    Smooth a signal by penalizing large second derivatives using FISTA optimization.

    Args:
        signal (np.ndarray): Input signal to be smoothed.
        alpha (float): Regularization strength (higher = smoother).
        max_iter (int): Maximum number of iterations for FISTA.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Smoothed signal.
    """
    # Normalize the input signal to avoid scaling issues
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    signal_normalized = (signal - mean_signal) / std_signal  # Normalize

    # Initialization
    n = len(signal_normalized)
    x = np.copy(signal_normalized)  # Initial guess
    z = np.copy(signal_normalized)  # Momentum variable
    t = 1.0

    # Lipschitz constant based step size (stable for any alpha)
    L = 4 + alpha  # Derived from the second derivative operator
    step_size = 1 / L  # Ensure stability
    print(step_size)

    # Gradient computation
    def gradient(x):
        """Compute the gradient including the second derivative term."""
        # Compute second derivative (D^2 x) with boundary handling
        d2x = np.zeros_like(x)
        d2x[1:-1] = x[:-2] - 2 * x[1:-1] + x[2:]  # Central difference
        return x - signal_normalized + alpha * d2x

    # Iterative FISTA updates
    for i in range(max_iter):
        # Compute gradient at z
        grad = gradient(z)

        # Update x and t (FISTA step)
        x_new = z - step_size * grad
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)

        # Convergence check
        if np.linalg.norm(x_new - x) / np.linalg.norm(x) < tol:
            break

        # Update variables
        x, t = x_new, t_new

    # Rescale the smoothed signal to original range
    smoothed_signal = x * std_signal + mean_signal  # De-normalize
    return smoothed_signal

from scipy.ndimage import gaussian_filter1d
def fft_demodulation_with_prediction(signal, fs, carrier_freq, fft_window=1200, 
                                     carrier_bandwidth=2000, step_size_factor=32, 
                                     prior_sigma=500, momentum=0.8):
    """
    FM Demodulation using FFT with Gaussian prediction-based priors and SNR estimation.

    Args:
        signal (np.ndarray): Input FM signal.
        fs (float): Sampling frequency (Hz).
        carrier_freq (float): Carrier frequency (Hz).
        fft_window (int): FFT window size (samples, default: 1200).
        carrier_bandwidth (float): Frequency range around carrier (Hz).
        step_size_factor (int): Factor controlling FFT step size relative to window.
        prior_sigma (float): Standard deviation for Gaussian prior (Hz).
        momentum (float): Weight of prior prediction (0–1).

    Returns:
        demodulated_signal (np.ndarray): Instantaneous frequency deviations.
        time (np.ndarray): Time vector for demodulated signal.
        snr_values (np.ndarray): SNR values for each detected frequency.
    """
    # FFT step size (overlap)
    step_size = fft_window // step_size_factor
    num_bins = (len(signal) - fft_window) // step_size + 1

    # Time vector
    time = np.arange(num_bins) * step_size / fs + (fft_window / (2 * fs))

    # Outputs
    demodulated_signal = []
    snr_values = []

    # Initialize prior estimate
    prior_freq = carrier_freq

    # FFT Demodulation Loop
    plt.figure()
    weight_function = None
    for i in range(num_bins):
        # Extract segment
        start = i * step_size
        end = start + fft_window
        segment = signal[start:end]
        #print(segment.shape)
        # FFT
        spectrum = np.fft.fft(segment * np.kaiser(len(segment), beta=8))
        freqs = np.fft.fftfreq(len(segment), d=1/fs)
        
        #print(freqs.shape)
        # Focus on carrier band
        carrier_range = (freqs >= (carrier_freq - carrier_bandwidth / 2)) & \
                        (freqs <= (carrier_freq + carrier_bandwidth / 2))
        freqs_in_band = freqs[carrier_range]

        basic_weight = (np.abs((freqs_in_band - 19e3)/1000))**0.1
        basic_weight = basic_weight/np.max(basic_weight)
        basic_weight = 1-basic_weight

        spectrum_in_band = np.abs(spectrum[carrier_range])
        sigma = 2
        spectrum_smoothed = gaussian_filter1d(np.abs(spectrum), sigma=sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0,  radius=None)
        spectrum_in_band_smoothed = np.abs(spectrum_smoothed[carrier_range])
        if weight_function is None:
            weight_function = np.zeros_like(freqs_in_band) + 1
        #print(spectrum_in_band.shape)
        # Apply Gaussian prior around predicted frequency
        
        if weight_function is None:
            #prior_gaussian = np.exp(-0.5 * ((freqs_in_band - prior_freq) / prior_sigma) ** 2)
            prior_gaussian = basic_weight
        else:
            #prior_gaussian = weight_function
            prior_gaussian = basic_weight
        weighted_spectrum = spectrum_in_band * prior_gaussian

        # Find peak
        peak_idx = np.argmax(weighted_spectrum)
        peak_freq = freqs_in_band[peak_idx]
        ntop = 30
        peak_idxtop10 = (-weighted_spectrum).argsort()[:ntop]
        peak_freqtop10 = freqs_in_band[peak_idxtop10]
        spec_peakstop10 = spectrum_in_band[peak_idxtop10]
        spec_peakstop10 = spec_peakstop10/np.sum(spec_peakstop10)
        weight_function = np.zeros_like(freqs_in_band)
        
        #plt.figure()
        for ii in range(ntop):
            #plt.plot(spec_peakstop10[ii] * np.exp(-0.5 * ((freqs_in_band - peak_freqtop10[ii]) / prior_sigma) ** 2))
            #plt.show()
            weight_function += spec_peakstop10[ii] * np.exp(-0.5 * ((freqs_in_band - peak_freqtop10[ii]) / 50) ** 2)
        weight_function = weight_function/np.max(weight_function)
        # print(weight_function)
        # print(freqs_in_band.shape)
        # print(spectrum_in_band.shape)
        # print(prior_gaussian.shape)
        plt.cla()
        plt.subplot(2,1,1)
        plt.cla()
        plt.semilogy(freqs_in_band, spectrum_in_band/np.max(spectrum_in_band),'r')
        plt.semilogy(freqs_in_band, spectrum_in_band_smoothed/np.max(spectrum_in_band_smoothed),'m')
        plt.semilogy(freqs_in_band, prior_gaussian/np.max(prior_gaussian),':k')
        plt.semilogy(freqs_in_band, weighted_spectrum/np.max(weighted_spectrum),'b')
        plt.semilogy(peak_freq, 1, '*b')

        plt.legend(['original data','weighting function','weighted spectrum'], loc='lower left')
        plt.subplot(2,1,2)
        plt.cla()
        plt.plot(freqs_in_band, weighted_spectrum,'b')

        #plt.ylim([0.1,100000])
        
        #plt.pause(0.01)
        plt.draw()
        plt.pause(0.01)

        

        # Compute SNR
        peak_power = np.max(weighted_spectrum)
        noise_power = np.mean(weighted_spectrum)  # Average background noise
        snr = 10 * np.log10(peak_power / noise_power) if noise_power > 0 else 0  # dB

        # Update prior using momentum
        prior_freq = momentum * prior_freq + (1 - momentum) * peak_freq

        # Store results
        demodulated_signal.append(peak_freq - carrier_freq)
        snr_values.append(snr)

    # Ensure output arrays are NumPy arrays
    demodulated_signal = np.array(demodulated_signal)
    snr_values = np.array(snr_values)

    plt.figure()
    plt.plot(time, np.clip(demodulated_signal,-6,6))
    plt.plot(time, snr_values)
    return demodulated_signal, time, snr_values

# decode ecg
def decode_ecg(fs, audio, resample_rate=600):
    # Select best channel if stereo

    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(SAMPLE_RATE)

    # Extract raw audio data
    data = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(-1, audio.channels)
    fs = audio.frame_rate

    # Select best channel if stereo
    data = select_best_channel(data, fs)

    # Remove DC offset
    data -= np.mean(data)

    # Extract raw audio data
    
    _, t_rs = resample_signal_rate(data, fs, new_sample_rate=resample_rate, get_t=1)

    method = 'spectrogram optimisation'
    method = 'fourier follower'
    recovered_signal, t_rs2 = do_demodulation(data, method=method,  t_rs=t_rs, fs=fs, resample_rate=resample_rate)
    recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=5, notch_width=0.01, target_low =45, target_high = 55, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=1, notch_width=0.02, target_low =35, target_high = 100, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=1, notch_width=0.02, target_low =48, target_high = 150, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=1, notch_width=0.05, target_low =4, target_high = 7, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter(recovered_signal, resample_rate, harmonics=5, method='fixed')
    #recovered_signal_db4 = denoise(recovered_signal, wavelet='db4', threshold=10, sorh='h', n_shifts=10)
    #recovered_signal_sym8 = denoise(recovered_signal, wavelet='sym8',threshold=10, sorh='h', n_shifts=10)
    #recovered_signal_coif5 = denoise(recovered_signal, wavelet='coif5',threshold=10, sorh='h', n_shifts=10)
    #recovered_signal = (recovered_signal_db4 + recovered_signal_sym8 + recovered_signal_coif5)/3
    # recovered_signalTV = total_variation_denoise_grad(recovered_signal, weight=0.003, max_iter = 200)
    #recovered_signalTV = fista_smooth_second_derivative(recovered_signal, alpha=0.25, max_iter=500, tol=1e-6)
    recovered_signal = bandpass_filter(recovered_signal, 0.5, 40, resample_rate)
    #recovered_signalTV = bandpass_filter(recovered_signalTV, 0.5, 40, resample_rate)

    
    #recovered_signal = np.clip(recovered_signal, -3, 6)
    plt.figure(figsize=(10, 6))
    #plt.subplot(2, 1, 1)
    plt.title("Original Signal")
    plt.plot([t_rs[0], t_rs[-1]], [0, 0],'-r', alpha=1)
    for i in np.arange(-3,6):
        plt.plot([t_rs[0], t_rs[-1]], [i, i],':r', alpha=0.5)
    plt.plot(t_rs2, recovered_signal)
    #plt.plot(t_rs2, recovered_signalTV, 'g')
    plt.ylim([-3,6])

    plt.show()

    # plt.figure(figsize=(10, 6))
    # #plt.subplot(2, 1, 1)
    # plt.title("Original Signal")
    
    # for i in np.arange(-5,6):
    #     plt.plot([t_rs[0], t_rs[-1]], [i, i],':r', alpha=0.5)
    # plt.plot(t_rs2[(40<t_rs2)& (t_rs2<60)], recovered_signal[(40<t_rs2)&(t_rs2<60)])
    # #plt.ylim([-3,6])

    # plt.show()


# Example Simulation



file_path = r"C:\\Users\\joshu\\Downloads\\Clean ecg kitchen.m4a"
audio = AudioSegment.from_file(file_path)


decode_ecg(SAMPLE_RATE, audio, resample_rate=600)
