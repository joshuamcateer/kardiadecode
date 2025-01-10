import numpy as np
from scipy.signal import correlate


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