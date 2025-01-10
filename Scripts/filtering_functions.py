# Filtering functions
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.fft import dct, idct
import pywt
from scipy.optimize import minimize
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks, spectrogram, stft

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

    return sosfiltfilt(sos, data)


def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    sos = butter(order, low, btype='low', output='sos')
    return sosfiltfilt(sos, data)


def frequency_domain_filter(signal, fs, harmonics=5, method='fixed', fwhm_factor=2):
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

    fixed_notch_width = 0.1
    # Compute the FFT of the signal
    freqs = fftfreq(len(signal), d=1/fs)
    fft_values = fft(signal)

    # Compute magnitude spectrum
    spectrum = np.abs(fft_values)
    positive_freqs = freqs[freqs >= 0]
    positive_spectrum = spectrum[freqs >= 0]


    # plt.figure()
    # plt.plot(freqs[:int(len(freqs)*0.1)], np.absolute(fft_values[:int(len(freqs)*0.1)]))
    # plt.show()
    # Detect the mains frequency
    mains_idx = np.argmax(positive_spectrum[:int(len(positive_freqs) * 0.1)])  # Search below 100 Hz
    mains_freq = positive_freqs[mains_idx]
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
        # print(f"Filtering {freq:.2f} Hz: Range [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Apply the mask
    fft_values[~mask] = 0

    # Transform back to the time domain
    filtered_signal = np.real(ifft(fft_values))
    # plt.figure()
    # plt.plot(freqs[:int(len(freqs)*0.1)], np.absolute(fft_values[:int(len(freqs)*0.1)]))
    # plt.show()
    return filtered_signal

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