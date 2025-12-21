import numpy as np
from typing import Union, Optional
from pathlib import Path
import struct
from scipy.fftpack import dct, idct
from PIL import Image

from .utils import (
    generate_measurement_matrix,
    omp,
    ista,
    simulated_annealing,
)


class IMCSDecoder:
    def __init__(
        self,
        reconstruction_algorithm: str = "omp",
        max_iter: int = 1000,
        lambda_param: float = 10.0,
    ):
        valid_algorithms = ["omp", "basis_pursuit", "iterative_threshold", "simulated_annealing"]
        if reconstruction_algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")

        self.reconstruction_algorithm = reconstruction_algorithm
        self.max_iter = max_iter
        self.lambda_param = lambda_param

        self.last_history = None
        self.last_residuals = None
        self.return_history = False

    def decode(self, compressed_data: bytes, return_history: bool = False) -> np.ndarray:
        self.return_history = return_history
        self.last_history = None
        self.last_residuals = None

        metadata, measurements = self._deserialize(compressed_data)

        if metadata["is_2d"]:
            return self._decode_2d(measurements, metadata)
        else:
            return self._decode_1d(measurements, metadata)

    def _run_reconstruction(
        self,
        y: np.ndarray,
        A: np.ndarray,
        sparsity: Optional[int] = None,
        adaptive_lambda: Optional[float] = None,
    ) -> np.ndarray:
        """
        Запускает алгоритм реконструкции и сохраняет историю если нужно.

        Args:
            y: Вектор измерений
            A: Sensing matrix
            sparsity: Параметр sparsity для OMP (опционально)
            adaptive_lambda: Параметр lambda для ISTA/SA (опционально)

        Returns:
            Восстановленный разреженный вектор s
        """
        if self.reconstruction_algorithm == "omp":
            if sparsity is None:
                raise ValueError("sparsity is required for OMP algorithm")
            result = omp(y, A, sparsity, return_history=self.return_history)
        elif self.reconstruction_algorithm in ["basis_pursuit", "iterative_threshold"]:
            if adaptive_lambda is None:
                raise ValueError("adaptive_lambda is required for ISTA algorithm")
            result = ista(y, A, adaptive_lambda, self.max_iter, return_history=self.return_history)
        elif self.reconstruction_algorithm == "simulated_annealing":
            if adaptive_lambda is None:
                raise ValueError("adaptive_lambda is required for SA algorithm")
            result = simulated_annealing(
                y, A, adaptive_lambda, max_iter=self.max_iter, return_history=self.return_history
            )
        else:
            if sparsity is None:
                raise ValueError("sparsity is required for OMP algorithm")
            result = omp(y, A, sparsity, return_history=self.return_history)

        if self.return_history:
            s, self.last_history, self.last_residuals = result
        else:
            s = result

        return s

    def _deserialize(self, data: bytes) -> tuple[dict, np.ndarray]:
        if len(data) < 28:
            raise ValueError("Invalid IMCS data: too short")
        if data[:4] != b"IMCS":
            raise ValueError("Invalid IMCS data: wrong magic number")

        # Unpack header
        header_format = "<4s B B B B I I I I I I"
        header_size = struct.calcsize(header_format)
        header = struct.unpack(header_format, data[:header_size])

        (
            magic,
            version,
            flags,
            basis_id,
            matrix_id,
            seed,
            orig_height,
            orig_width,
            m_row,
            m_col,
            data_length,
        ) = header

        # Map IDs back to strings
        basis_map = {0: "dct", 1: "wavelet"}
        matrix_map = {0: "gaussian", 1: "bernoulli", 2: "random"}

        is_2d = bool(flags & 1)

        metadata = {
            "version": version,
            "is_2d": is_2d,
            "sparsity_basis": basis_map.get(basis_id, "dct"),
            "matrix_type": matrix_map.get(matrix_id, "gaussian"),
            "seed": seed,
            "original_shape": (orig_height, orig_width) if is_2d else (orig_height,),
            "m_row": m_row,
            "m_col": m_col,
        }

        # Extract measurements
        measurements_bytes = data[header_size : header_size + data_length]
        measurements = np.frombuffer(measurements_bytes, dtype=np.float64)

        # Reshape if 2D AND separable (m_col > 0)
        # If m_col == 0, it's full random (keep as 1D vector)
        if is_2d and m_col > 0:
            measurements = measurements.reshape(m_row, m_col)

        return metadata, measurements

    def _decode_1d(self, y: np.ndarray, metadata: dict) -> np.ndarray:
        n = metadata["original_shape"][0]
        m = metadata["m_row"]

        # Regenerate the same measurement matrix using the stored seed
        Phi = generate_measurement_matrix(m, n, metadata["matrix_type"], metadata["seed"])

        # Create explicit DCT matrix for small signals
        # For 1D, we use DCT as the sparsity basis: A = Φ·Ψ^T
        Psi = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            Psi[:, i] = dct(e_i, norm="ortho")

        # Sensing matrix in DCT domain: A = Φ·Ψ^T
        A = Phi @ Psi.T

        # Target sparsity (rough estimate)
        sparsity = max(int(0.2 * n), 5)

        # Adaptive lambda: scale with signal size
        adaptive_lambda = self.lambda_param * (n / 1000.0)

        # Reconstruct sparse coefficients
        s = self._run_reconstruction(y, A, sparsity=sparsity, adaptive_lambda=adaptive_lambda)

        # Transform back from DCT domain: x = Ψ^T · s = IDCT(s)
        x = idct(s, norm="ortho")

        return x

    def _decode_2d(self, y: np.ndarray, metadata: dict) -> np.ndarray:
        n_row, n_col = metadata["original_shape"]
        m_total = metadata["m_row"]  # Total measurements
        n_total = n_row * n_col

        # Regenerate FULL measurement matrix
        Phi = generate_measurement_matrix(
            m_total, n_total, metadata["matrix_type"], metadata["seed"]
        )

        # Create 2D DCT matrix as Kronecker product
        def create_dct_matrix(n):
            """Create explicit DCT matrix."""
            Psi = np.zeros((n, n))
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                Psi[:, i] = dct(e_i, norm="ortho")
            return Psi

        Psi_row = create_dct_matrix(n_row)
        Psi_col = create_dct_matrix(n_col)
        Psi_2d = np.kron(Psi_col, Psi_row)

        # Sensing matrix in DCT domain: A = Φ · Ψ^T
        A = Phi @ Psi_2d.T

        # Adaptive sparsity based on measurements
        sparsity = max(min(m_total // 2, n_total // 4), 1)

        # Adaptive lambda: scale with image size
        # Smaller images need smaller lambda
        adaptive_lambda = self.lambda_param * (n_total / 1000.0)

        # Solve: y = A · s
        s = self._run_reconstruction(y, A, sparsity=sparsity, adaptive_lambda=adaptive_lambda)

        # Transform back: x = Ψ^T · s
        x_vec = Psi_2d.T @ s

        # Reshape to 2D (column-major)
        X = x_vec.reshape(n_row, n_col, order="F")

        return X

    def decode_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load compressed data
        with open(input_path, "rb") as f:
            compressed_data = f.read()

        # Decode
        reconstructed = self.decode(compressed_data)

        # Save based on output file type
        suffix = output_path.suffix.lower()
        if suffix == ".npy":
            np.save(output_path, reconstructed)
        elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            # Clip to valid range and convert to uint8
            img_data = np.clip(reconstructed, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_data, mode="L")
            img.save(output_path)
        else:
            # Default: save as numpy
            np.save(output_path, reconstructed)

    def set_reconstruction_algorithm(self, algorithm: str) -> None:
        valid_algorithms = ["omp", "basis_pursuit", "iterative_threshold", "simulated_annealing"]
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        self.reconstruction_algorithm = algorithm

    def get_info(self, compressed_data: bytes) -> dict:
        metadata, measurements = self._deserialize(compressed_data)

        # Calculate compression achieved
        original_size = np.prod(metadata["original_shape"]) * 8  # float64
        compressed_size = measurements.size * 8

        return {
            **metadata,
            "compressed_shape": measurements.shape,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "actual_compression_ratio": compressed_size / original_size,
        }
