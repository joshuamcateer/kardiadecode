import numpy as np


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


def generate_mains_hum(fs, duration, freq=50, amplitude=1.0, harmonics=[1, 2, 3]):
    """Generate Mains Frequency Hum with Harmonics."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    hum = np.zeros_like(t)
    for h in harmonics:
        hum += amplitude / h * np.sin(2 * np.pi * (freq * h) * t)
    return hum


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
