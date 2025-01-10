import numpy as np
import pywt
from scipy.optimize import minimize
from scipy.fft import fft, ifft, fftfreq
from scipy.fft import dct, idct



def total_variation_denoise_lowfissue(signal, weight=0.1, l1_weight=0.1, max_iter=100, tol=1e-6):
    """
    Total Variation (TV) denoising using Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Args:
        signal (np.ndarray): Input noisy signal.
        weight (float): TV regularization weight (higher -> more smoothing).
        l1_weight (float): L1 sparsity weight (higher -> enforces zeros).
        max_iter (int): Maximum iterations for optimization.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Denoised signal.
    """
    n = len(signal)
    x = np.copy(signal)  # Initialization
    z = np.copy(signal)  # Intermediate variable
    t = 1.0  # Momentum term
    step_size = 1.0  # Step size for gradient descent

    def gradient(y):
        """Gradient of data fidelity term."""
        return y - signal

    def prox_tv(y, step):
        """Proximal operator for Total Variation (TV)."""
        diff = np.diff(y)
        # Apply soft thresholding to the gradient
        shrink = np.sign(diff) * np.maximum(np.abs(diff) - step * weight, 0)
        # Compute cumulative sum to reconstruct denoised signal
        return np.cumsum(np.concatenate(([y[0]], shrink)))

    def prox_l1(y, step):
        """Proximal operator for L1 regularization."""
        return np.sign(y) * np.maximum(np.abs(y) - step * l1_weight, 0)

    # FISTA iterations
    for i in range(max_iter):
        x_old = np.copy(x)

        # Gradient step
        grad = gradient(z)
        x = prox_tv(z - step_size * grad, step_size)

        # Apply sparsity constraint (L1 regularization)
        x = prox_l1(x, step_size)

        # FISTA momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x






def total_variation_denoise(signal, weight=0.001, l1_weight=0.0, low_freq_weight=10, low_freq_cutoff=0.01, max_iter=100, tol=1e-6):
    """
    Total Variation (TV) denoising with FISTA and low-frequency anchoring.

    Args:
        signal (np.ndarray): Input noisy signal.
        weight (float): TV regularization weight.
        l1_weight (float): L1 sparsity weight for zero enforcement.
        low_freq_weight (float): Low-frequency anchoring weight.
        low_freq_cutoff (float): Low-frequency cutoff fraction (default: 0.01).
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Denoised signal.
    """
    # Initialization
    n = len(signal)
    x = np.copy(signal)
    z = np.copy(signal)
    t = 1.0
    step_size = 0.1  # Gradient step size

    # Precompute low-frequency components
    def low_freq_filter(sig, cutoff):
        """Smooth low-pass filtering in DCT domain."""
        dct_sig = dct(sig, norm='ortho')
        freqs = np.linspace(0, 1, len(dct_sig))  # Normalized frequency axis
        low_pass = np.exp(-0.5 * (freqs / cutoff) ** 2)  # Gaussian filter
        dct_sig *= low_pass  # Apply Gaussian filter
        return idct(dct_sig, norm='ortho')

    # Precomputed low-frequency target
    low_freq_signal = low_freq_filter(signal, low_freq_cutoff)

    def gradient(y):
        """Gradient computation for data fidelity and low-frequency anchoring."""
        # Fidelity gradient (data term)
        grad = y - signal

        # Low-frequency anchoring term
        if low_freq_weight > 0:
            # Apply low-frequency filter and enforce similarity
            grad += low_freq_weight * (low_freq_filter(y, low_freq_cutoff) - low_freq_signal)

        return grad

    def prox_tv(y, step):
        """Proximal operator for TV regularization."""
        diff = np.diff(y)
        shrink = np.sign(diff) * np.maximum(np.abs(diff) - step * weight, 0)
        return np.cumsum(np.concatenate(([y[0]], shrink)))

    def prox_l1(y, step):
        """Proximal operator for L1 regularization."""
        return np.sign(y) * np.maximum(np.abs(y) - step * l1_weight, 0)

    # FISTA iterations
    for i in range(max_iter):
        x_old = np.copy(x)

        # Compute gradient
        grad = gradient(z)

        # Apply TV regularization
        x = prox_tv(z - step_size * grad, step_size)

        # Apply sparsity constraint
        x = prox_l1(x, step_size)

        # Momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x


def total_variation_denoise_grad(signal, weight=0.001, l1_weight=0.0, low_freq_weight=10, low_freq_cutoff=0.01, max_iter=100, tol=1e-6):
    """
    Total Variation (TV) denoising with FISTA and low-frequency anchoring.

    Args:
        signal (np.ndarray): Input noisy signal.
        weight (float): TV regularization weight.
        l1_weight (float): L1 sparsity weight for zero enforcement.
        low_freq_weight (float): Low-frequency anchoring weight.
        low_freq_cutoff (float): Low-frequency cutoff fraction (default: 0.01).
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Denoised signal.
    """
    # Initialization
    n = len(signal)
    x = np.copy(signal)
    z = np.copy(signal)
    t = 1.0
    step_size = 0.1  # Gradient step size

    # Precompute low-frequency components
    def low_freq_filter(sig, cutoff):
        """Smooth low-pass filtering in DCT domain."""
        dct_sig = dct(sig, norm='ortho')
        freqs = np.linspace(0, 1, len(dct_sig))  # Normalized frequency axis
        low_pass = np.exp(-0.5 * (freqs / cutoff) ** 2)  # Gaussian filter
        dct_sig *= low_pass  # Apply Gaussian filter
        return idct(dct_sig, norm='ortho')

    # Precomputed low-frequency target
    low_freq_signal = low_freq_filter(signal, low_freq_cutoff)

    def gradient(y):
        """Gradient computation for data fidelity and low-frequency anchoring."""
        # Fidelity gradient (data term)
        grad = y - signal

        # Low-frequency anchoring term
        if low_freq_weight > 0:
            # Apply low-frequency filter and enforce similarity
            grad += low_freq_weight * (low_freq_filter(y, low_freq_cutoff) - low_freq_signal)

        return grad

    def prox_tv(y, step):
        """Proximal operator for TV regularization."""
        diff = np.diff(np.diff(y))
        shrink = np.sign(diff) * np.maximum(np.abs(diff) - step * weight, 0)
        return np.cumsum(np.cumsum(np.concatenate(([y[0]], [y[1]], shrink))))

    def prox_l1(y, step):
        """Proximal operator for L1 regularization."""
        return np.sign(y) * np.maximum(np.abs(y) - step * l1_weight, 0)

    # FISTA iterations
    for i in range(max_iter):
        x_old = np.copy(x)

        # Compute gradient
        grad = gradient(z)

        # Apply TV regularization
        x = prox_tv(z - step_size * grad, step_size)

        # Apply sparsity constraint
        x = prox_l1(x, step_size)

        # Momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next

        # Check convergence
        if np.linalg.norm(x - x_old) < tol:
            break

    return x



def fista_smooth_second_derivative(signal, alpha, max_iter=500, tol=1e-6):
    """
    Smooth a signal by penalizing large second derivatives using FISTA optimization.

    Args:
        signal (np.ndarray): Input signal to be smoothed.
        alpha (float): Regularization strength (higher = smoother).
        max_iter (int): Maximum number of iterations for FISTA.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Smoothed signal.
    """
    # Normalize the input signal to avoid scaling issues
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    signal_normalized = (signal - mean_signal) / std_signal  # Normalize

    # Initialization
    n = len(signal_normalized)
    x = np.copy(signal_normalized)  # Initial guess
    z = np.copy(signal_normalized)  # Momentum variable
    t = 1.0

    # Lipschitz constant based step size (stable for any alpha)
    L = 4 + alpha  # Derived from the second derivative operator
    step_size = 1 / L  # Ensure stability
    print(step_size)

    # Gradient computation
    def gradient(x):
        """Compute the gradient including the second derivative term."""
        # Compute second derivative (D^2 x) with boundary handling
        d2x = np.zeros_like(x)
        d2x[1:-1] = x[:-2] - 2 * x[1:-1] + x[2:]  # Central difference
        return x - signal_normalized + alpha * d2x

    # Iterative FISTA updates
    for i in range(max_iter):
        # Compute gradient at z
        grad = gradient(z)

        # Update x and t (FISTA step)
        x_new = z - step_size * grad
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)

        # Convergence check
        if np.linalg.norm(x_new - x) / np.linalg.norm(x) < tol:
            break

        # Update variables
        x, t = x_new, t_new

    # Rescale the smoothed signal to original range
    smoothed_signal = x * std_signal + mean_signal  # De-normalize
    return smoothed_signal


def denoise_old(signal, wavelet='db4', threshold=0.04, sorh='s'):
    sorh = sorh.lower()
    if sorh == 's':
        sorh = 'soft'
    else:
        sorh = 'hard'
    coeffs = pywt.wavedec(signal, wavelet)
    coeffs[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:]]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal



def denoise(signal, wavelet='db4', threshold=None, sorh='soft', n_shifts=10):
    """
    Denoise a signal using wavelet thresholding with optional cycle spinning.
    
    Args:
        signal (np.ndarray): Input noisy signal.
        wavelet (str): Type of wavelet to use (default: 'db4').
        threshold (float or None): Threshold for wavelet coefficients (default: None, adaptive).
        sorh (str): 'soft' or 'hard' thresholding (default: 'soft').
        cycle_spinning (bool): Apply cycle spinning to improve shift invariance (default: False).
        n_shifts (int): Number of shifts for cycle spinning (default: 10).

    Returns:
        np.ndarray: Denoised signal.
    """
    # Ensure valid thresholding mode
    sorh = sorh.lower()

    if sorh == 's':
        sorh = 'soft'
    elif sorh == 'h':
        sorh = 'hard'

    if sorh not in ['soft', 'hard']:
        raise ValueError("Thresholding mode must be 's', 'soft' or 'h', 'hard'.")

    # Estimate noise level if threshold is not provided
    def estimate_noise(coeffs):
        # Use the detail coefficients at the first level to estimate noise
        median = np.median(np.abs(coeffs[-1]))
        sigma = median / 0.6745  # Robust estimate of sigma (MAD)
        return sigma

    def wavelet_denoise(sig, thr):
        """Performs standard wavelet denoising."""
        coeffs = pywt.wavedec(sig, wavelet)

        # Estimate threshold adaptively if not provided
        
        sigma = estimate_noise(coeffs)
        thr = sigma * np.sqrt(2 * np.log(len(signal)))*thr  # Universal threshold

        # Apply thresholding to detail coefficients
        coeffs[1:] = [pywt.threshold(c, value=thr, mode=sorh) for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)

    if n_shifts == 0:
        # No cycle spinning: Apply wavelet denoising directly
        return wavelet_denoise(signal, threshold)

    # With cycle spinning: Average denoised outputs across shifts
    denoised_signals = []
    n = len(signal)

    shift_factor = 600

    for shift in range(n_shifts):
        # Circularly shift the signal
        shifted_signal = np.roll(signal, shift*shift_factor)
        # Denoise the shifted signal
        denoised_shifted = wavelet_denoise(shifted_signal, threshold)
        # Reverse the shift
        denoised_signals.append(np.roll(denoised_shifted, -shift*shift_factor))

    # Average across all shifted versions
    denoised_signal = np.mean(denoised_signals, axis=0)
    return denoised_signal



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



def wavelet_transform(signal, wavelet='db4', level=None):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs_flat, slices = pywt.coeffs_to_array(coeffs)
    return coeffs_flat, slices


def inverse_wavelet_transform(coeffs_flat, slices, wavelet='db4'):
    coeffs = pywt.array_to_coeffs(coeffs_flat, slices, output_format='wavedec')
    return pywt.waverec(coeffs, wavelet)


def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


