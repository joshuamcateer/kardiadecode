from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert, sosfilt
from scipy.fftpack import fft
import os


from scipy.fft import fft, ifft, fftfreq


import numpy as np
from pydub import AudioSegment
from scipy.signal import correlate, butter, lfilter, hilbert, resample
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import pywt
from scipy.optimize import minimize

from kardia_singal_functions import  select_best_channel, mask_short_below_threshold
from signal_generation import *
from scipy.fft import fft, ifft, fftfreq
from scipy.fft import dct, idct

from demodulation_functions import do_demodulation, resample_signal_rate, resample_signal
from filtering_functions import frequency_domain_filter_targeted, bandpass_filter
from scipy.ndimage import gaussian_filter1d


from denoise_functions import denoise
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






def power_in_band(signal, low, high, fs, normalised=0):
    bandpassed = bandpass_filter(signal, low, high, fs)
    if normalised==1:
        return np.sum(np.abs(bandpassed)**2) / np.sum(np.abs(signal - bandpassed)**2)
    else:
        return np.sum(np.abs(bandpassed)**2)










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

# decode ecg
def decode_ecg(audio, resample_rate=600, plotOn=1, freq_min=2, freq_max=150, mask_threshold=0.5):
    # Select best channel if stereo

    audio = AudioSegment.from_file(file_path)
    
    SAMPLE_RATE = audio.frame_rate
    

    # Extract raw audio data
    data = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(-1, audio.channels)
    

    # Select best channel if stereo
    data = select_best_channel(data, SAMPLE_RATE, CARRIER_FREQ=CARRIER_FREQ)

    # Remove DC offset
    data -= np.mean(data)

    # Extract raw audio data
    _, t_rs = resample_signal_rate(data, SAMPLE_RATE, new_sample_rate=resample_rate, get_t=1)

    method = 'quadrature'
    # perform the demodulation
    recovered_signal, t_rs2, SNR = do_demodulation(data, method=method,  t_rs=t_rs, fs=SAMPLE_RATE, resample_rate=resample_rate)

    # filter out mains hum include just above and just below US/UK/EU mains frequency regions
    recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=5, notch_width=0.01, target_low =45, target_high = 65, method='fixed', fwhm_factor=2)
    recovered_signal_db4 = denoise(recovered_signal, wavelet='db4', threshold=100, sorh='h', n_shifts=10)

    
    recovered_signal = bandpass_filter(recovered_signal_db4, freq_min, freq_max, resample_rate)

    if len(recovered_signal) > len(t_rs2):
        
        recovered_signal = recovered_signal[(len(recovered_signal)-len(t_rs2)):]
    
    if mask_threshold != 0:
        masked_signal,_ = mask_short_below_threshold(recovered_signal, int(resample_rate/2), mask_threshold, 300, step=100)
    # clip signal to maximum range
    recovered_signal = np.clip(recovered_signal, -6, 6)
    if plotOn == 1:
        plt.figure(figsize=(10, 6))
        #plt.subplot(2, 1, 1)
        plt.title("Original Signal")
        plt.plot([t_rs[0], t_rs[-1]], [0, 0],'-r', alpha=1)
        for i in np.arange(-3,6):
            plt.plot([t_rs[0], t_rs[-1]], [i, i],':r', alpha=0.75)
        for i in np.arange(-3,6,0.5):
            plt.plot([t_rs[0], t_rs[-1]], [i, i],'-r', alpha=0.25)
        for i in np.arange(-3,6,0.1):
            plt.plot([t_rs[0], t_rs[-1]], [i, i],'-r', alpha=0.1)
        plt.plot(t_rs2, recovered_signal,'b', label='Recovered signal')
        if mask_threshold != 0:
            plt.plot(t_rs2, masked_signal, 'g', label='Masked signal')
        plt.title('Recovered signal estimate')
        plt.legend(loc="upper left")

        
        plt.ylim([-3,6])
        plt.ylabel('ECG signal/mV')
        plt.xlabel('time/sec')

        plt.show()
    if mask_threshold != 0:
        return t_rs2, masked_signal
    return t_rs2, recovered_signal





# Main
file_path = r"C:\\Users\\joshu\\Downloads\\Similtaneous ECG 1 2024-12-11 140800.m4a"
file_path = r"C:\\Users\\joshu\\Downloads\\ECG test1.m4a"
#file_path = r"C:\Users\joshu\OneDrive\Documents\Projects\KaridaDecode\piezo_pickup\recordings\secondtestprojhighersamplerate88200_16PCM_louder0dB.wav"
print(os.path.exists(file_path))

decode_ecg(file_path)
