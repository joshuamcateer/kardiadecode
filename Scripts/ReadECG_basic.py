from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert, sosfilt
from scipy.fftpack import fft
import os


from scipy.fft import fft, ifft, fftfreq

# Constants
CARRIER_FREQ = 19000  # Carrier frequency in Hz
CALIBRATION = 200     # 200 Hz/mV
SAMPLE_RATE = 48000   # Audio sampling rate
SAMPLE_RATE = 88200
FILTER_LOW = 0.5      # Low cutoff for cardiac signal (Hz)
FILTER_HIGH = 40      # High cutoff for cardiac signal (Hz)
MIN_RECORD_GAP_LENGTH = 3  # Minimum gap length in seconds
MIN_RECORD_LENGTH = 10     # Minimum signal length in seconds
NOISE_THRESHOLD = 0.05     # Noise threshold (adjust as needed)


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


# FM Demodulation
def fm_demodulate(signal, fs, carrier_freq):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0 * np.pi)
    return instantaneous_frequency - carrier_freq


def post_proces_signal(signal, clip=10):
    signal = np.nan_to_num(signal, 0)
    signal = np.clip(signal, -clip, clip)
    return signal

def power_in_band(signal, low, high, fs, normalised=0):
    bandpassed = bandpass_filter(signal, low, high, fs)
    if normalised==1:
        return np.sum(np.abs(bandpassed)**2) / np.sum(np.abs(signal - bandpassed)**2)
    else:
        return np.sum(np.abs(bandpassed)**2)


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


# Detect Gaps and Noisy Regions
def detect_regions(signal, fs):
    power = np.abs(signal)
    noise_mask = power > NOISE_THRESHOLD
    regions = []
    start_idx, in_gap = 0, False

    for i in range(len(noise_mask)):
        if not noise_mask[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif noise_mask[i] and in_gap:
            in_gap = False
            if (i - gap_start) / fs > MIN_RECORD_GAP_LENGTH:
                regions.append((start_idx, gap_start))
                start_idx = i
    # Final region
    if (len(signal) - start_idx) / fs > MIN_RECORD_LENGTH:
        regions.append((start_idx, len(signal)))
    return regions, noise_mask


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
        print(f"Filtering {freq:.2f} Hz: Range [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Apply the mask
    fft_values[~mask] = 0

    # Transform back to the time domain
    filtered_signal = np.real(ifft(fft_values))
    plt.figure()
    plt.plot(freqs[:int(len(freqs)*0.1)], np.absolute(fft_values[:int(len(freqs)*0.1)]))
    plt.show()
    return filtered_signal



# Plotting Functions
def plot_ecg(signal, regions, fs, output_dir=None):
    time = np.arange(len(signal)) / fs
    rows = int(np.ceil(len(signal) / (fs * 8)))
    offset = -20  # mV offset between rows

    for row in range(rows):
        start = row * 8 * fs
        end = min((row + 1) * 8 * fs, len(signal))
        t_row = time[start:end]
        s_row = signal[start:end] #+ row * offset

        # Plot current row
        plt.figure(figsize=(12, 4))
        plt.plot(t_row, s_row, color='black', linewidth=0.1)
        plt.grid(which='both', linestyle='-', linewidth=0.1, alpha=0.5)
        plt.title(f"ECG Signal: Row {row + 1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.ylim([-10, 10])

        # Save or display
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"ECG_Row_{row + 1}.png"), dpi=300)
        else:
            plt.show()
        plt.close()


# Process File
def process_file(file_path):
    # Load audio using pydub
    audio = AudioSegment.from_file(file_path)
    # audio = audio.set_frame_rate(SAMPLE_RATE)

    # Extract raw audio data
    data = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(-1, audio.channels)
    fs = audio.frame_rate
    SAMPLE_RATE = audio.frame_rate
    # Select best channel if stereo
    data = select_best_channel(data, fs)

    # Remove DC offset
    data -= np.mean(data)
    print('number of nans in original data is:')
    print(np.count_nonzero(np.isnan(data)))
    # Bandpass filter to isolate carrier signal
    filtered_data = bandpass_filter(data, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, fs)

    print('number of nans in filtered_data is:')
    print(np.count_nonzero(np.isnan(filtered_data)))
    # FM demodulation
    demodulated_signal = fm_demodulate(filtered_data, fs, CARRIER_FREQ)
    
    
    print('number of nans in demodulated_signal is:')
    print(np.count_nonzero(np.isnan(demodulated_signal)))
    # plt.figure()
    # plt.plot(demodulated_signal)
    # plt.show()
    # Scale to ECG voltage (mV)
    ecg_signal = demodulated_signal / CALIBRATION
    
    print(ecg_signal)
    # Filter mains frequency and harmonics
    # ecg_signal = detect_and_filter_mains(ecg_signal, fs)

    print('power_in_band' + str(power_in_band(ecg_signal, 2, 40, fs, normalised=0)))
    ecg_signal = frequency_domain_filter(ecg_signal, fs, method='fwhm')
    print('power_in_band' + str(power_in_band(ecg_signal, 2, 40, fs, normalised=0)))
    #frequency_domain_filter
    print('post mains filter')

    plt.figure()
    plt.plot(ecg_signal)
    plt.title('ecg_signal post mains filter')
    plt.show()
    

    # ecg_signal = post_proces_signal(ecg_signal, clip=1000)

    # Further filtering to isolate ECG range (0.5 Hz to 40 Hz)
    ecg_filtered = bandpass_filter(ecg_signal, FILTER_LOW, FILTER_HIGH, fs)
    print('power_in_band: ' + str(power_in_band(ecg_filtered, 2, 40, fs, normalised=0)))
    print('power_in_band50: ' + str(power_in_band(ecg_filtered, 49, 51, fs, normalised=0)))
    print('post HB filter')
    print(ecg_filtered)
    
    # Remove nans and clip to range
    ecg_filtered = post_proces_signal(ecg_filtered, clip=10)

    # plt.figure()
    # plt.plot(ecg_filtered)
    # plt.show()

    # Detect regions
    regions, noise_mask = detect_regions(ecg_filtered, fs)

    # Plot results
    output_dir = os.path.splitext(os.path.basename(file_path))[0]
    print(output_dir)
    plot_ecg(ecg_filtered, regions, fs, output_dir=output_dir)
    return ecg_filtered




# Main
file_path = r"C:\\Users\\joshu\\Downloads\\Similtaneous ECG 1 2024-12-11 140800.m4a"
file_path = r"C:\\Users\\joshu\\Downloads\\ECG test1.m4a"
file_path = r"C:\Users\joshu\OneDrive\Documents\Projects\KaridaDecode\piezo_pickup\recordings\secondtestprojhighersamplerate88200_16PCM_louder0dB.wav"
print(os.path.exists(file_path))

process_file(file_path)
