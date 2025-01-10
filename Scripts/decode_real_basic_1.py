import numpy as np
from pydub import AudioSegment
from scipy.signal import correlate, butter, lfilter, hilbert, resample
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import pywt
from scipy.optimize import minimize

from kardia_singal_functions import  select_best_channel
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
FILTER_LOW = 0.5      # Low cutoff for cardiac signal (Hz)
FILTER_LOW = 0.1      # Low cutoff for cardiac signal (Hz)
FILTER_HIGH = 40      # High cutoff for cardiac signal (Hz)
FILTER_HIGH = 150      # High cutoff for cardiac signal (Hz)
MIN_RECORD_GAP_LENGTH = 3  # Minimum gap length in seconds
MIN_RECORD_LENGTH = 10     # Minimum signal length in seconds
NOISE_THRESHOLD = 0.05     # Noise threshold (adjust as needed)

#def load_data_from_file

# decode ecg
def decode_ecg(fs, audio, resample_rate=600):
    # Select best channel if stereo

    audio = AudioSegment.from_file(file_path)
    
    SAMPLE_RATE = audio.frame_rate
    #audio = audio.set_frame_rate(SAMPLE_RATE)

    # Extract raw audio data
    data = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(-1, audio.channels)
    

    # Select best channel if stereo
    data = select_best_channel(data, SAMPLE_RATE, CARRIER_FREQ=CARRIER_FREQ)

    # Remove DC offset
    data -= np.mean(data)

    # Extract raw audio data
    
    _, t_rs = resample_signal_rate(data, SAMPLE_RATE, new_sample_rate=resample_rate, get_t=1)

    method = 'spectrogram optimisation'
    #method = 'fourier follower'
    method = 'quadrature'
    recovered_signal, t_rs2, SNR = do_demodulation(data, method=method,  t_rs=t_rs, fs=SAMPLE_RATE, resample_rate=resample_rate)
    recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=5, notch_width=0.01, target_low =45, target_high = 55, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=1, notch_width=0.02, target_low =35, target_high = 100, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=1, notch_width=0.02, target_low =48, target_high = 150, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter_targeted(recovered_signal, resample_rate, harmonics=1, notch_width=0.05, target_low =4, target_high = 7, method='fixed', fwhm_factor=2)
    #recovered_signal = frequency_domain_filter(recovered_signal, resample_rate, harmonics=5, method='fixed')
    recovered_signal_db4 = denoise(recovered_signal, wavelet='db4', threshold=10, sorh='h', n_shifts=10)
    recovered_signal_sym8 = denoise(recovered_signal, wavelet='sym8',threshold=10, sorh='h', n_shifts=10)
    recovered_signal_coif5 = denoise(recovered_signal, wavelet='coif5',threshold=10, sorh='h', n_shifts=10)
    recovered_signal = (recovered_signal_db4 + recovered_signal_sym8 + recovered_signal_coif5)/3
    # recovered_signalTV = total_variation_denoise_grad(recovered_signal, weight=0.003, max_iter = 200)
    #recovered_signalTV = fista_smooth_second_derivative(recovered_signal, alpha=0.25, max_iter=500, tol=1e-6)
    recovered_signal = bandpass_filter(recovered_signal, 2, 40, resample_rate)
    #recovered_signalTV = bandpass_filter(recovered_signalTV, 0.5, 40, resample_rate)

    if len(recovered_signal) > len(t_rs2):
        print(len(recovered_signal)-len(t_rs2))
        print(len(recovered_signal))
        print(len(t_rs2))
        print(len(t_rs2[(len(recovered_signal)-len(t_rs2)):]))
        #recovered_signal = resample_signal(recovered_signal[(len(recovered_signal)-len(t_rs2)):], t_rs2, t_rs2)
        recovered_signal = recovered_signal[(len(recovered_signal)-len(t_rs2)):]
        print(len(recovered_signal)-len(t_rs2))

    SNR = np.clip(SNR, -1, 20)
    mSNR = np.mean(SNR)
    SNRblur = bandpass_filter(SNR, 0.0001, 1, resample_rate) + mSNR
    plt.figure()
    plt.plot(t_rs2, SNR)
    plt.plot(t_rs2, SNRblur)
    plt.title('SNR estimate')
    plt.show()

    #recovered_signal = np.clip(recovered_signal, -3, 6)
    plt.figure(figsize=(10, 6))
    #plt.subplot(2, 1, 1)
    plt.title("Original Signal")
    plt.plot([t_rs[0], t_rs[-1]], [0, 0],'-r', alpha=1)
    for i in np.arange(-3,6):
        plt.plot([t_rs[0], t_rs[-1]], [i, i],':r', alpha=0.5)
    plt.plot(t_rs2, recovered_signal,'b')
    plt.title('Recovered signal estimate')
    
    #plt.plot(t_rs2, recovered_signalTV, 'g')
    plt.ylim([-3,6])

    plt.show()

    plt.figure()
    plt.hist(np.log(SNR),1000)
    plt.title('SNR estimate histogram')
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
file_path = r"C:\Users\joshu\OneDrive\Documents\Projects\KaridaDecode\piezo_pickup\recordings\secondtestprojhighersamplerate88200_16PCM_louder0dB.wav"
file_path = r"C:\Users\joshu\OneDrive\Documents\Projects\KaridaDecode\piezo_pickup\recordings\acutal_stereo_newpickups.wav"

audio = AudioSegment.from_file(file_path)


decode_ecg(SAMPLE_RATE, audio, resample_rate=600)
