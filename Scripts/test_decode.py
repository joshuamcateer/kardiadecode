import numpy as np
from scipy.signal import correlate, butter, lfilter, hilbert, resample
from scipy.stats import norm
import matplotlib.pyplot as plt

import pywt
from scipy.optimize import minimize


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def wavelet_transform(signal, wavelet='db4', level=None):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_flat, slices = pywt.coeffs_to_array(coeffs)
    return coeffs_flat, slices


def inverse_wavelet_transform(coeffs_flat, slices, wavelet='db4'):
    coeffs = pywt.array_to_coeffs(coeffs_flat, slices, wavelet)
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


def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype='low')
    return lfilter(b, a, data)


def fm_demodulate(signal, fs):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) * fs / (2.0 * np.pi)
    return instantaneous_frequency


def superheterodyne_demodulate(signal, fs, carrier_freq, if_freq):
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

def sparse_fm_demodulation(signal, fs, carrier_freq, if_freq):
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

    # Step 3: Compressed sensing reconstruction
    recovered_coeffs = compressive_sensing_reconstruction(baseband_signal)
    sparse_signal = inverse_wavelet_transform(recovered_coeffs)
    return sparse_signal

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
    num_samples = int(len(baseband_signal) * resample_rate / fs)
    resampled_signal = resample(baseband_signal, num_samples)

    # Step 4: Iterative compressed sensing reconstruction
    recovered_coeffs, slices = iterative_compressed_sensing(resampled_signal)
    sparse_signal = inverse_wavelet_transform(recovered_coeffs, slices)
    return sparse_signal

# --- Sound Sources ---
def generate_signal(fs, duration, freq=10, amplitude=1.0):
    """Generate a sine wave signal."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_gwn(fs, duration, amplitude=1.0):
    """Generate Gaussian White Noise."""
    return amplitude * np.random.normal(0, 1, int(fs * duration))

def generate_pink_noise(fs, duration, amplitude=1.0):
    """Generate Pink Noise using a 1/f distribution."""
    samples = int(fs * duration)
    uneven = samples % 2
    X = np.random.randn(samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
    S = np.arange(len(X)) + 1
    y = (np.fft.irfft(X / np.sqrt(S))).real
    return amplitude * y[:samples]

def generate_mains_hum(fs, duration, freq=50, amplitude=1.0):
    """Generate Mains Frequency Hum."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_broad_spectrum_noise(fs, duration, amplitude=1.0):
    """Generate Intermittent Broad Spectrum Noise."""
    noise = generate_gwn(fs, duration, amplitude)
    intervals = np.random.choice([0, 1], size=(int(duration * fs),), p=[0.8, 0.2])
    return noise * intervals

# --- Microphone Array ---
def propagate_sound(source, mic_locs, source_loc, fs, sound_speed=343, gain=1.0):
    """Propagate sound from a source to multiple microphones."""
    distances = np.linalg.norm(mic_locs - source_loc, axis=1)
    delays = distances / sound_speed
    attenuations = 1 / (1 + distances)
    propagated_signals = []
    for delay, attenuation in zip(delays, attenuations):
        delay_samples = int(delay * fs)
        signal = np.roll(source, delay_samples) * attenuation * gain
        propagated_signals.append(signal)
    return np.array(propagated_signals)

def mix_signals(signals):
    """Mix multiple signals."""
    return np.sum(signals, axis=0)

# --- Signal Recovery Quality ---
def cross_correlation_metric(original, recovered):
    """Compute cross-correlation to handle time delays."""
    correlation = correlate(recovered, original, mode='full')
    max_corr = np.max(correlation)
    return max_corr / (np.linalg.norm(original) * np.linalg.norm(recovered))

def normalized_mean_squared_error(original, recovered):
    """Compute normalized mean squared error (NMSE)."""
    original = original[:len(recovered)]
    mse = np.mean((original - recovered) ** 2)
    return mse / np.mean(original ** 2)

# --- Simulation and Testing ---
def run_simulation(fs, duration, signal_freq, mic_locs, signal_loc, noise_sources):
    """Run a full simulation and test signal recovery."""
    # Generate the main signal
    main_signal = generate_signal(fs, duration, freq=signal_freq)

    # Propagate the main signal to the microphones
    propagated_signals = propagate_sound(main_signal, mic_locs, signal_loc, fs)

    # Generate and propagate noise signals
    for noise_func, noise_loc, noise_gain in noise_sources:
        noise = noise_func(fs, duration)
        noise_propagated = propagate_sound(noise, mic_locs, noise_loc, fs, gain=noise_gain)
        propagated_signals += noise_propagated

    # Mix signals from all microphones
    mixed_signal = mix_signals(propagated_signals)

    # Apply demodulation and sparse recovery
    recovered_signal = sparse_fm_demodulation(mixed_signal, fs, carrier_freq=19000, if_freq=2000)

    # Evaluate signal recovery quality
    cross_corr = cross_correlation_metric(main_signal, recovered_signal)
    nmse = normalized_mean_squared_error(main_signal, recovered_signal)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.title("Original Signal")
    plt.plot(main_signal[:fs])
    plt.subplot(3, 1, 2)
    plt.title("Mixed Signal at Microphones")
    plt.plot(mixed_signal[:fs])
    plt.subplot(3, 1, 3)
    plt.title("Recovered Signal")
    plt.plot(recovered_signal[:fs])
    plt.tight_layout()
    plt.show()

    print(f"Cross-Correlation: {cross_corr:.3f}")
    print(f"Normalized MSE: {nmse:.3f}")

# Example Simulation
fs = 48000
duration = 2.0  # seconds
signal_freq = 10
mic_locs = np.array([[0, 0], [0, 0.1]])  # 4 microphones
signal_loc = np.array([0, -0.01])  # Signal source location
noise_sources = [
    (generate_mains_hum, np.array([1, 1.5]), 0.5),
    (generate_gwn, np.array([0.2, 0.8]), 0.8),
    (generate_pink_noise, np.array([0.7, 0.3]), 0.6),
    (generate_broad_spectrum_noise, np.array([1.2, 0.7]), 0.7)
]

run_simulation(fs, duration, signal_freq, mic_locs, signal_loc, noise_sources)
