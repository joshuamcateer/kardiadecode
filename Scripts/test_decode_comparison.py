import numpy as np
from scipy.signal import correlate, butter, lfilter, hilbert, resample
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    
    if method == 0 or method.lower() == 'hilbert':
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

    elif method == 1 or method.lower() == 'superhetrodyne' or method.lower() == 'hetrodyne':
        print('Superhetrodyne demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs)
        demodulated_signal = superheterodyne_demodulate(filtered_data, fs, carrier_freq=19000, if_freq=2000)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)
        
    elif method == 2 or method.lower() == 'cs' or method.lower() == 'sparse':
        print('Sparse demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        recovered_signal, _ = sparse_fm_demodulation(filtered_data, fs, carrier_freq=19000, if_freq=2000, resample_rate=resample_rate)
        recovered_signal = bandpass_filter(recovered_signal, FILTER_LOW, FILTER_HIGH, fs)
        _, t_rs2 = resample_signal_rate(recovered_signal, resample_rate, new_sample_rate=resample_rate, get_t=1, plusone=0)
        recovered_signal = resample_signal(recovered_signal, t_rs2, t_rs)
        t_rs2 = t_rs

    elif method == 3 or method.lower() == 'quadrature':
        print('Quadrature demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        print("I don't know why it is upsidedown")
        demodulated_signal = -quadrature_demodulation(filtered_data, fs, CARRIER_FREQ)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)
        recovered_signal = resample_signal(recovered_signal, t_rs2, t_rs)
        t_rs2 = t_rs
    elif method == 4 or method.lower() == 'zero crossing':
        print('Zero crossing demodulation')
        filtered_data = bandpass_filter(mixed_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)
        demodulated_signal = zero_crossing_demodulation(filtered_data, fs)
        ecg_signal = demodulated_signal / CALIBRATION
        ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
        recovered_signal, t_rs2 = resample_signal_rate(ecg_filtered, fs, new_sample_rate=resample_rate, get_t=1, plusone=0)

    elif method == 5 or method.lower() == 'fft' or method.lower() == 'fourier':
        print('Fourier window demodulation')
        #mixed_signal = bandpass_filter(mixed_signal, CARRIER_FREQ - 500, CARRIER_FREQ + 500, fs, order=8)

        demodulated_signal, t_fft = time_binned_fft_demodulation(mixed_signal, fs, CARRIER_FREQ, fft_window=1200, carrier_bandwidth=1000*2)
        fs_fft = len(t_fft)/(t_fft[-1]-t_fft[0])
        print(demodulated_signal.shape)
        print(t_fft.shape)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_fft, t_rs)
        t_rs2 = t_rs
        #recovered_signal, t_rs2 = resample_signal(ecg_signal, fs_fft, new_sample_rate=resample_rate, get_t=1, plusone=0)
    elif method == 6 or method.lower() == 'fft_band' or method.lower() == 'fourier_band':
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
    elif method == 699999 or method.lower() == 'pll' or method.lower() == 'phase locked loop':
        print('This method doesnt work')
        print('Phase locked loop demodulation')
        # TODO finish this
        demodulated_signal, t_pll = pll_fm_demodulator(mixed_signal, fs, CARRIER_FREQ, loop_bandwidth=50)
        ecg_signal = demodulated_signal / CALIBRATION
        recovered_signal = resample_signal(ecg_signal, t_pll, t_rs)
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

# Example Simulation
fs = 48000
duration = 20  # seconds
signal_freq = 75
#mic_locs = np.array([[0, 0], [0, 0.1]])  # 4 microphones
mic_locs = np.array([[0, 0]])  # 4 microphones
signal_loc = np.array([0, -0.01])  # Signal source location
noise_sources = [
    (generate_mains_hum, np.array([1, 1.5]), 0.5),
    (generate_gwn, np.array([0.2, 0.8]), 0.8),
    (generate_pink_noise, np.array([0.7, 0.3]), 0.6),
    (generate_broad_spectrum_noise, np.array([1.2, 0.7]), 0.7)
]

run_simulation(fs, duration, signal_freq, mic_locs, signal_loc, noise_sources=noise_sources)
