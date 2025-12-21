import numpy as np
from scipy.fftpack import dct, idct
from typing import Optional


def generate_measurement_matrix(
    m: int, n: int, matrix_type: str = "gaussian", seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a measurement matrix for compressed sensing.

    The measurement matrix Φ should be incoherent with the sparsity basis.
    Random matrices (Gaussian, Bernoulli) are incoherent with any fixed basis
    with high probability.

    Args:
        m: Number of measurements (compressed size), M << N
        n: Signal dimension (original size)
        matrix_type: Type of measurement matrix ('gaussian', 'bernoulli', 'random')
        seed: Random seed for reproducibility (important for decoder!)

    Returns:
        Measurement matrix of shape (m, n), normalized columns
    """
    if seed is not None:
        np.random.seed(seed)

    if matrix_type == "gaussian":
        # Gaussian random matrix - elements from N(0, 1/m)
        # This provides good RIP properties with high probability
        matrix = np.random.randn(m, n) / np.sqrt(m)

    elif matrix_type == "bernoulli":
        # Bernoulli random matrix - elements ±1/√m with equal probability
        # Also provides good RIP properties
        matrix = np.random.choice([-1, 1], size=(m, n)) / np.sqrt(m)

    elif matrix_type == "random":
        # Uniform random matrix (less common, but still works)
        matrix = (np.random.rand(m, n) - 0.5) * 2 / np.sqrt(m)

    else:
        raise ValueError(
            f"Unknown matrix type: {matrix_type}. Use 'gaussian', 'bernoulli', or 'random'"
        )

    return matrix


def compute_sparsity(signal: np.ndarray, threshold: float = 1e-10) -> int:
    """
    Compute the sparsity of a signal (number of non-zero elements).

    In compressed sensing, sparsity K is the number of significant (non-zero)
    coefficients. A signal is K-sparse if it has at most K non-zero elements.

    Args:
        signal: Input signal (can be 1D or 2D)
        threshold: Threshold below which values are considered zero

    Returns:
        Number of significant coefficients (sparsity level K)
    """
    return np.sum(np.abs(signal) > threshold)


def validate_imcs_format(data: bytes) -> bool:
    if len(data) < 4:
        return False
    return data[:4] == b"IMCS"


def dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def omp(
    y: np.ndarray,
    A: np.ndarray,
    sparsity: int,
    tolerance: float = 1e-6,
    return_history: bool = False,
):
    m, n = A.shape
    y = y.flatten()

    residual = y.copy()  # r_0 = y
    support: list[int] = []  # Λ - indices of selected atoms
    s = np.zeros(n)  # sparse coefficients

    if return_history:
        history = [s.copy()]
        residuals = [float(np.linalg.norm(residual))]

    for _ in range(sparsity):
        # Step 1: Find index with maximum correlation
        # λ = argmax_j |⟨r, a_j⟩|
        correlations = np.abs(A.T @ residual)
        correlations[support] = -np.inf  # Don't select already chosen indices
        new_idx = int(np.argmax(correlations))
        support.append(new_idx)

        # Step 2: Solve least squares on selected support
        # s_Λ = argmin ||y - A_Λ s_Λ||_2
        A_support = A[:, support]
        s_support, _, _, _ = np.linalg.lstsq(A_support, y, rcond=None)

        # Step 3: Update residual
        # r = y - A_Λ s_Λ
        residual = y - A_support @ s_support

        # Save history
        if return_history:
            s_temp = np.zeros(n)
            s_temp[support] = s_support
            history.append(s_temp.copy())
            residuals.append(float(np.linalg.norm(residual)))

        # Check convergence
        if np.linalg.norm(residual) < tolerance:
            break

    # Construct full sparse vector
    s[support] = s_support

    if return_history:
        return s, history, residuals
    return s


def ista(
    y: np.ndarray,
    A: np.ndarray,
    lambda_param: float = 0.1,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    return_history: bool = False,
):
    m, n = A.shape
    y = y.flatten()

    # Lipschitz constant (for step size)
    # L = ||A^T A||_2 ≈ largest eigenvalue
    L = np.linalg.norm(A, ord=2) ** 2

    step = 1.0 / L

    # Soft thresholding operator
    def soft_threshold(x, thresh):
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

    s = np.zeros(n)
    threshold = lambda_param * step

    if return_history:
        history = [s.copy()]
        residuals = [float(np.linalg.norm(y - A @ s))]

    for _ in range(max_iter):
        s_old = s.copy()

        # Gradient step
        gradient = A.T @ (A @ s - y)
        s = s - step * gradient

        # Proximal step (soft thresholding)
        s = soft_threshold(s, threshold)

        # Save history
        if return_history:
            history.append(s.copy())
            residuals.append(float(np.linalg.norm(y - A @ s)))

        # Check convergence
        if np.linalg.norm(s - s_old) < tolerance:
            break

    if return_history:
        return s, history, residuals
    return s


def simulated_annealing(
    y: np.ndarray,
    A: np.ndarray,
    lambda_param: float = 0.1,
    max_iter: int = 10000,
    T_init: float = 10.0,
    cooling_rate: float = 0.89,
    step_size: float = 10.0,
    bounds: tuple = (-1000, 1000),
    adaptive_step: bool = True,
    return_history: bool = False,
):
    m, n = A.shape
    y = y.flatten()
    blo, bup = bounds

    def objective(s):
        """Вычисляет ||y - A·s||₂² + λ·||s||₁"""
        fit_error = np.sum((y - A @ s) ** 2)
        l1_penalty = lambda_param * np.sum(np.abs(s))
        return fit_error + l1_penalty

    # Используем ISTA как начальную точку (warm start)
    # Это намного лучше, чем стартовать с нулей
    s_current = ista(y, A, lambda_param, max_iter=100)
    f_current = objective(s_current)

    s_best = s_current.copy()
    f_best = f_current

    if return_history:
        history = [s_current.copy()]
        residuals = [float(np.linalg.norm(y - A @ s_current))]

    T = T_init
    current_step = step_size

    accept_count = 0
    total_count = 0

    for iteration in range(max_iter):
        perturbation = np.zeros(n)
        n_perturb = max(1, int(0.1 * n))
        indices = np.random.choice(n, n_perturb, replace=False)
        perturbation[indices] = np.random.uniform(-1, 1, n_perturb)

        s_candidate = s_current + current_step * perturbation

        s_candidate = np.clip(s_candidate, blo, bup)

        f_candidate = objective(s_candidate)

        delta_f = f_candidate - f_current

        accepted = False
        if delta_f < 0:
            s_current = s_candidate.copy()
            f_current = f_candidate
            accepted = True

            if f_candidate < f_best:
                s_best = s_candidate.copy()
                f_best = f_candidate
        else:
            acceptance_probability = np.exp(-delta_f / T)
            if np.random.uniform(0, 1) < acceptance_probability:
                s_current = s_candidate.copy()
                f_current = f_candidate
                accepted = True

        accept_count += int(accepted)
        total_count += 1

        # Save history every 50 iterations
        if return_history and iteration % 50 == 0:
            history.append(s_current.copy())
            residuals.append(float(np.linalg.norm(y - A @ s_current)))

        if adaptive_step and total_count > 0 and total_count % 50 == 0:
            acceptance_rate = accept_count / total_count
            if acceptance_rate > 0.6:
                current_step *= 1.1
            elif acceptance_rate < 0.3:
                current_step *= 0.9
            accept_count = 0
            total_count = 0

        T = cooling_rate * T

    threshold = 0.01 * np.max(np.abs(s_best))
    s_best = np.sign(s_best) * np.maximum(np.abs(s_best) - threshold, 0)

    if return_history:
        return s_best, history, residuals
    return s_best


def calculate_compression_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")

    orig_flat: np.ndarray = original.flatten().astype(np.float64)
    recon_flat: np.ndarray = reconstructed.flatten().astype(np.float64)

    # Mean Squared Error
    mse = np.mean((orig_flat - recon_flat) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(orig_flat - recon_flat))

    # Relative Error (L2 norm)
    orig_norm = np.linalg.norm(orig_flat)
    if orig_norm > 0:
        relative_error = np.linalg.norm(orig_flat - recon_flat) / orig_norm
    else:
        relative_error = 0.0 if np.allclose(orig_flat, recon_flat) else np.inf

    # Peak Signal-to-Noise Ratio
    # For normalized signals [0, 1], MAX = 1
    # For 8-bit images [0, 255], MAX = 255
    max_val = max(np.max(np.abs(orig_flat)), 1.0)
    if mse > 0:
        psnr = 10 * np.log10((max_val**2) / mse)
    else:
        psnr = np.inf  # Perfect reconstruction

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "psnr": float(psnr),
        "relative_error": float(relative_error),
    }
