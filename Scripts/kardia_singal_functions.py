
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


import numpy as np

def rolling_sd(signal, window_size):
    """
    Calculates the rolling standard deviation of a 1D NumPy array.

    Parameters:
    -----------
    signal : np.ndarray
        One-dimensional input array (e.g. your signal).
    window_size : int
        Size of the rolling window.

    Returns:
    --------
    np.ndarray
        Array of the same length as 'signal' containing the rolling
        standard deviations. The first (window_size - 1) entries are set
        to NaN because a full window is not available for those positions.
    """
    if len(signal) < window_size:
        raise ValueError("Signal length must be at least 'window_size'.")

    # For each valid window, compute the standard deviation
    rolling_values = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        rolling_values.append(window.std())

    # Pad the beginning of the result with NaNs so output matches the length of 'signal'
    padding = [np.nan] * (window_size - 1)
    return np.array(padding + rolling_values)

def mask_signal_above_sd_threshold(signal, threshold, window_size):
    """
    Masks (sets to NaN) values in 'signal' where the rolling standard deviation
    exceeds a given threshold.

    Parameters:
    -----------
    signal : np.ndarray
        One-dimensional input array (e.g. your signal).
    threshold : float
        Threshold for the rolling standard deviation.
    window_size : int
        Size of the rolling window.

    Returns:
    --------
    np.ndarray
        A copy of 'signal' where any position whose rolling SD is above
        'threshold' has been replaced with NaN.
    """
    sds = rolling_sd(signal, window_size)
    # Wherever the SD exceeds the threshold, replace the original signal with NaN
    masked_signal = np.where(sds > threshold, np.nan, signal)
    return masked_signal

def rolling_sd_sparse(signal, window_size, step=1):
    """
    Computes a 'sparse' rolling standard deviation of the input signal. 
    - It only calculates the rolling SD at intervals defined by 'step' 
      to reduce computation.
    - It returns an upsampled array of the same length as 'signal' via interpolation.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array (the input signal).
    window_size : int
        Size of the rolling window.
    step : int, optional
        How frequently to compute the rolling standard deviation. For example, 
        if step=10, the rolling SD is computed for i=0, 10, 20, ...
        Default is 1 (compute for every sample).
    
    Returns
    -------
    sds_full : np.ndarray
        Rolling SD estimates at each sample (same length as 'signal'), 
        obtained by interpolating the sparse calculations.
    """
    n = len(signal)
    if n < window_size:
        raise ValueError("Signal length must be at least 'window_size'.")

    # Indices where we'll compute the rolling SD
    sparse_indices = range(0, n - window_size + 1, step)
    
    # Calculate the 'centre' of each window so that when we interpolate 
    # back to the full index, it's aligned sensibly.
    # For a window [i : i + window_size], we use the centre ~ i + (window_size // 2).
    centers = []
    sds_sparse = []

    for i in sparse_indices:
        window = signal[i : i + window_size]
        sds_sparse.append(window.std())
        centers.append(i + (window_size // 2))
        
    # If the last sparse index doesn't reach the final window, 
    # we can compute one more point at the very end. This helps 
    # avoid an extrapolation gap at the end.
    last_possible_start = n - window_size
    if (last_possible_start not in sparse_indices) and (last_possible_start > sparse_indices[-1]):
        # Compute the SD for the final valid window
        window = signal[last_possible_start : last_possible_start + window_size]
        sds_sparse.append(window.std())
        centers.append(last_possible_start + (window_size // 2))

    # Interpolate the sparse SD values onto the full set of sample indices [0..n-1]
    x_full = np.arange(n)
    sds_full = np.interp(x_full, centers, sds_sparse)
    
    return sds_full

def remove_short_valid_sections_below_threshold(valid_mask, min_valid_length):
    """
    Given a boolean array 'valid_mask' (True indicates below threshold, 
    False indicates above threshold), remove any short contiguous stretches 
    of True values whose length is < min_valid_length.

    Parameters
    ----------
    valid_mask : np.ndarray (dtype=bool)
        Boolean array indicating which samples are valid (below threshold).
    min_valid_length : int
        Minimum length of a valid contiguous region to keep.

    Returns
    -------
    np.ndarray (dtype=bool)
        A copy of 'valid_mask' with any short True segments turned to False.
    """
    valid_mask = valid_mask.copy()
    n = len(valid_mask)
    i = 0
    
    while i < n:
        if valid_mask[i]:
            # Found the start of a contiguous valid region
            start = i
            while i < n and valid_mask[i]:
                i += 1
            end = i - 1
            length = end - start + 1
            
            if length < min_valid_length:
                # Mask out (set to False) this entire short valid region
                valid_mask[start : end + 1] = False
        else:
            i += 1
    
    return valid_mask

def mask_short_below_threshold(signal, window_size, threshold, min_valid_length, step=1):
    """
    1) Computes a sparse rolling SD (faster than computing for every sample).
    2) Identifies samples where the SD is below threshold (== 'valid').
    3) Removes any short valid stretches whose length is below 'min_valid_length'.
    4) Returns a masked (NaN) version of the signal for invalid (above-threshold or short) regions.
    
    Parameters
    ----------
    signal : np.ndarray
        The 1D input signal.
    window_size : int
        Size of the rolling window for computing standard deviation.
    threshold : float
        Threshold applied to the rolling SD. Samples where SD < threshold
        are initially considered 'valid'.
    min_valid_length : int
        Minimum number of consecutive valid samples required to keep them.
        Shorter stretches of valid samples are masked out.
    step : int, optional
        Step size for the sparse SD calculation (default=1 means no sparsity).
    
    Returns
    -------
    masked_signal : np.ndarray
        Copy of 'signal' with invalid samples replaced by NaN.
        - Invalid means either (SD >= threshold) or in a short valid stretch.
    sds_full : np.ndarray
        The upsampled rolling SD array.
    """
    # 1) Compute sparse rolling SD & upsample
    sds_full = rolling_sd_sparse(signal, window_size, step=step)
    
    # 2) Create a boolean mask for which samples are below the threshold
    below_thresh_mask = (sds_full < threshold)
    
    # 3) Remove any contiguous valid region that is too short
    final_valid_mask = remove_short_valid_sections_below_threshold(
        below_thresh_mask, 
        min_valid_length
    )
    
    # 4) Create a masked version of the signal
    masked_signal = np.where(final_valid_mask, signal, np.nan)
    return masked_signal, sds_full