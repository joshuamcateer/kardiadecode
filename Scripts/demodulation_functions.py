
from scipy.signal import butter, hilbert, resample
import numpy as np
from filtering_functions import bandpass_filter, lowpass_filter
from denoise_functions import inverse_wavelet_transform, wavelet_transform, soft_thresholding
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks, spectrogram, stft
from scipy.interpolate import interp2d, interp1d
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import matplotlib.pyplot as plt


# Helper function to do demodulation
def do_demodulation(mixed_signal, method='fft',  t_rs=None, fs=48000, resample_rate=600, CARRIER_FREQ = 19000, CALIBRATION=200, FILTER_LOW=0.52, FILTER_HIGH=40, estimate_quality=True):
    method = method.lower()
    SNR = None
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
        assert "warning demodulation method notcorrectly specified"


    if estimate_quality == True:
        if SNR is not None:
            return recovered_signal, t_rs2, SNR
        else:
            #mixed_signal_rs = 
            return recovered_signal, t_rs2, local_SNR(resample_signal_rate(mixed_signal, fs, new_sample_rate=resample_rate, get_t=0, plusone=0), fs, CARRIER_FREQ, 1000, 2000)

    return recovered_signal, t_rs2

def local_SNR(data, fs, freqOfInterst, inBand, outBand):
    bandpassedIn = bandpass_filter(data, freqOfInterst - inBand/2, freqOfInterst + inBand/2, fs)
    bandpassedOut = bandpass_filter(data, freqOfInterst - outBand/2, freqOfInterst + outBand/2, fs)
    return np.abs(bandpassedIn)**2 / (np.abs(bandpassedOut)**2)

#def quality_estimate(data)
# demodulation functions

# works okay
def fm_demodulate(signal, fs, carrier_freq):
    # FM Demodulation hilbert transform based
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0 * np.pi)
    return instantaneous_frequency - carrier_freq
# Works pretty well
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
# Simplest demodulation
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

# FFT methods
# Kind of works but gets stuck at the ends due to high energy in lower freqs
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
# Works well
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


# Spectrogram methods
# Works find
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
# Helper function to compute spectrogram
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
# Clever optimised demodulation method
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


# works, doesn't really make sense
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
# Kind of works. I think
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
# kind of works. Helper function
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


# Not working. Doesn't really make sense
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
    # plt.figure()
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

        basic_weight = (np.abs(((freqs_in_band - 19e3)**2.0)/1000))**0.1
        basic_weight = basic_weight/np.max(basic_weight)
        basic_weight = 1-basic_weight

        spectrum_in_band = np.abs(spectrum[carrier_range])
        sigma = 2
        #spectrum_smoothed = gaussian_filter1d(np.abs(spectrum), sigma=sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0,  radius=None)
        #spectrum_in_band_smoothed = np.abs(spectrum_smoothed[carrier_range])
        #if weight_function is None:
        #    weight_function = np.zeros_like(freqs_in_band) + 1
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
        # plt.cla()
        # plt.subplot(2,1,1)
        # plt.cla()
        # plt.semilogy(freqs_in_band, spectrum_in_band/np.max(spectrum_in_band),'r')
        # plt.semilogy(freqs_in_band, spectrum_in_band_smoothed/np.max(spectrum_in_band_smoothed),'m')
        # plt.semilogy(freqs_in_band, prior_gaussian/np.max(prior_gaussian),':k')
        # plt.semilogy(freqs_in_band, weighted_spectrum/np.max(weighted_spectrum),'b')
        # plt.semilogy(peak_freq, 1, '*b')

        # plt.legend(['original data','weighting function','weighted spectrum'], loc='lower left')
        # plt.subplot(2,1,2)
        # plt.cla()
        # plt.plot(freqs_in_band, weighted_spectrum,'b')

        #plt.ylim([0.1,100000])
        
        #plt.pause(0.01)
        # plt.draw()
        # plt.pause(0.01)

        

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

    # plt.figure()
    # plt.plot(time, np.clip(demodulated_signal,-6,6))
    # plt.plot(time, snr_values)
    return demodulated_signal, time, snr_values

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