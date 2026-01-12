import numpy as np
from typing import Union, Optional
from pathlib import Path
import struct
from PIL import Image

from .utils import generate_measurement_matrix


class IMCSEncoder:
    # IMCS file format version
    FORMAT_VERSION = 1

    def __init__(
        self,
        compression_ratio: float = 0.5,
        sparsity_basis: str = "dct",
        matrix_type: str = "gaussian",
        seed: Optional[int] = None,
    ):
        if not 0 < compression_ratio < 1:
            raise ValueError("Compression ratio must be between 0 and 1")

        self.compression_ratio = compression_ratio
        self.sparsity_basis = sparsity_basis
        self.matrix_type = matrix_type
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)

    def encode(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)

        if data.ndim == 1:
            return self._encode_1d(data)
        elif data.ndim == 2:
            return self._encode_2d(data)
        else:
            raise ValueError(f"Only 1D and 2D arrays supported, got {data.ndim}D")

    def _encode_1d(self, x: np.ndarray) -> bytes:
        n = len(x)
        m = max(1, int(n * self.compression_ratio))

        # Generate measurement matrix Φ (M × N)
        Phi = generate_measurement_matrix(m, n, self.matrix_type, self.seed)

        # Compute measurements: y = Φ · x
        y = Phi @ x

        # Serialize to IMCS format
        return self._serialize(
            measurements=y,
            original_shape=(n,),
            m_row=m,
            m_col=0,  # 0 indicates 1D data
        )

    def _encode_2d(self, X: np.ndarray) -> bytes:
        n_row, n_col = X.shape
        n_total = n_row * n_col

        # Total measurements (not separable!)
        m_total = max(1, int(n_total * self.compression_ratio))

        # Generate FULL random matrix Φ ∈ ℝ^(M×N_total)
        Phi = generate_measurement_matrix(m_total, n_total, self.matrix_type, self.seed)

        # Vectorize image (column-major for DCT compatibility)
        x_vec = X.reshape(-1, order="F")

        # Compute measurements: y = Φ · x
        y = Phi @ x_vec

        # Serialize to IMCS format
        return self._serialize(
            measurements=y,
            original_shape=(n_row, n_col),
            m_row=m_total,
            m_col=0,
        )

    def _serialize(
        self,
        measurements: np.ndarray,
        original_shape: tuple,
        m_row: int,
        m_col: int,
    ) -> bytes:
        # Map sparsity basis to ID
        basis_map = {"dct": 0, "wavelet": 1}
        basis_id = basis_map.get(self.sparsity_basis, 0)

        # Map matrix type to ID
        matrix_map = {"gaussian": 0, "bernoulli": 1, "random": 2}
        matrix_id = matrix_map.get(self.matrix_type, 0)

        # Flags
        is_2d = len(original_shape) == 2
        flags = int(is_2d)

        # Convert measurements to bytes
        measurements_flat: np.ndarray = measurements.flatten().astype(np.float64)
        measurements_bytes = measurements_flat.tobytes()

        # Build header
        header = struct.pack(
            "<4s B B B B I I I I I I",
            b"IMCS",  # Magic number
            self.FORMAT_VERSION,  # Version
            flags,  # Flags
            basis_id,  # Sparsity basis
            matrix_id,  # Matrix type
            self.seed,  # Seed for reproducibility
            original_shape[0],  # Original height/length
            original_shape[1] if is_2d else 0,  # Original width (0 for 1D)
            m_row,  # M_row
            m_col,  # M_col
            len(measurements_bytes),  # Data length
        )

        return header + measurements_bytes

    def encode_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load data based on file type
        suffix = input_path.suffix.lower()
        if suffix == ".npy":
            data = np.load(input_path)
        elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            img = Image.open(input_path).convert("L")  # Convert to grayscale
            data = np.array(img, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Encode and save
        compressed = self.encode(data)
        with open(output_path, "wb") as f:
            f.write(compressed)

    def set_compression_ratio(self, ratio: float) -> None:
        if not 0 < ratio < 1:
            raise ValueError("Compression ratio must be between 0 and 1")
        self.compression_ratio = ratio

    def get_info(self) -> dict:
        return {
            "compression_ratio": self.compression_ratio,
            "sparsity_basis": self.sparsity_basis,
            "matrix_type": self.matrix_type,
            "seed": self.seed,
            "format_version": self.FORMAT_VERSION,
        }
