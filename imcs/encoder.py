import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
import struct
from PIL import Image

from .utils import (
    dct2,
    idct2,
    generate_measurement_matrix,
    IMCS_HEADER_V1_FMT,
    IMCS_HEADER_V2_EXT_FMT,
    IMCS_HEADER_V3_EXT_FMT,
    IMCS_FLAG_BLOCK_MEAN,
    IMCS_FLAG_IS_2D,
    IMCS_FLAG_LOW_FREQUENCY,
    IMCS_FLAG_PER_BLOCK,
    IMCS_QUANT_FLOAT64,
    IMCS_QUANT_INT16,
    IMCS_QUANT_INT8,
    zigzag_indices,
)


class IMCSEncoder:
    # IMCS file format: v3 adds quantized measurement payloads.
    FORMAT_VERSION = 3

    def __init__(
        self,
        compression_ratio: float = 0.5,
        sparsity_basis: str = "dct",
        matrix_type: str = "gaussian",
        seed: Optional[int] = None,
        block_size: Optional[Tuple[int, int]] = None,
        measurement_mode: str = "shared",
        measurement_dtype: str = "float64",
        block_mean_residual: bool = True,
        low_frequency_coeffs: int = 0,
    ):
        if not 0 < compression_ratio < 1:
            raise ValueError("Compression ratio must be between 0 and 1")
        if sparsity_basis not in {"dct", "wavelet"}:
            raise ValueError("sparsity_basis must be 'dct' or 'wavelet'")
        if measurement_mode not in {"shared", "per_block"}:
            raise ValueError("measurement_mode must be 'shared' or 'per_block'")
        if measurement_dtype not in {"float64", "int16", "int8"}:
            raise ValueError("measurement_dtype must be 'float64', 'int16', or 'int8'")
        if low_frequency_coeffs < 0:
            raise ValueError("low_frequency_coeffs must be non-negative")

        self.compression_ratio = compression_ratio
        self.sparsity_basis = sparsity_basis
        self.matrix_type = matrix_type
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        self.block_size = block_size
        self.measurement_mode = measurement_mode
        self.measurement_dtype = measurement_dtype
        self.block_mean_residual = block_mean_residual
        self.low_frequency_coeffs = low_frequency_coeffs

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
            block_h=0,
            block_w=0,
            content_shape=(n,),
        )

    def _encode_2d(self, X: np.ndarray) -> bytes:
        if self.block_size is not None:
            return self._encode_2d_blocked(X)
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

        # Serialize to IMCS format (v2 with zero block fields = legacy full-image)
        return self._serialize(
            measurements=y,
            original_shape=(n_row, n_col),
            m_row=m_total,
            m_col=0,
            block_h=0,
            block_w=0,
            content_shape=(n_row, n_col),
        )

    def _encode_2d_blocked(self, X: np.ndarray) -> bytes:
        bh, bw = self.block_size  # type: ignore[misc]
        if bh < 1 or bw < 1:
            raise ValueError("block_size must be positive")

        content_h, content_w = X.shape
        pad_h = int(np.ceil(content_h / bh) * bh)
        pad_w = int(np.ceil(content_w / bw) * bw)
        if pad_h != content_h or pad_w != content_w:
            X_pad = np.zeros((pad_h, pad_w), dtype=np.float64)
            X_pad[:content_h, :content_w] = X
        else:
            X_pad = np.asarray(X, dtype=np.float64)

        n_blocks_h = pad_h // bh
        n_blocks_w = pad_w // bw
        n_blocks = n_blocks_h * n_blocks_w
        b_pixels = bh * bw
        m_per_block = max(1, int(b_pixels * self.compression_ratio))

        measurements_list: list[np.ndarray] = []
        block_means: list[float] = []
        low_frequency_list: list[np.ndarray] = []
        low_frequency_positions = zigzag_indices(bh, bw)[: self.low_frequency_coeffs]
        idx = 0
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                block = X_pad[bi * bh : (bi + 1) * bh, bj * bw : (bj + 1) * bw]
                block_mean = float(np.mean(block)) if self.block_mean_residual else 0.0
                if self.block_mean_residual:
                    block_means.append(block_mean)
                residual_block = block - block_mean
                if low_frequency_positions:
                    coeffs = dct2(residual_block)
                    side_coeffs = np.array(
                        [coeffs[pos] for pos in low_frequency_positions], dtype=np.float64
                    )
                    low_frequency_list.append(side_coeffs)
                    side_coeff_matrix = np.zeros_like(coeffs)
                    for pos, value in zip(low_frequency_positions, side_coeffs):
                        side_coeff_matrix[pos] = value
                    residual_block = residual_block - idct2(side_coeff_matrix)
                x_vec = residual_block.reshape(-1, order="F")
                phi_seed = self.seed if self.measurement_mode == "shared" else self.seed + idx
                Phi = generate_measurement_matrix(m_per_block, b_pixels, self.matrix_type, phi_seed)
                measurements_list.append(Phi @ x_vec)
                idx += 1

        y = np.concatenate(measurements_list)

        return self._serialize(
            measurements=y,
            original_shape=(pad_h, pad_w),
            m_row=m_per_block,
            m_col=n_blocks,
            block_h=bh,
            block_w=bw,
            content_shape=(content_h, content_w),
            block_means=np.asarray(block_means, dtype=np.float64)
            if self.block_mean_residual
            else None,
            low_frequency_coefficients=np.vstack(low_frequency_list)
            if low_frequency_list
            else None,
        )

    def _quantize_low_frequency_coefficients(
        self, coefficients: Optional[np.ndarray]
    ) -> tuple[int, float, bytes]:
        if coefficients is None or coefficients.size == 0:
            return 0, 1.0, b""
        clipped_count = int(coefficients.shape[1])
        max_abs = float(np.max(np.abs(coefficients)))
        scale = max_abs / float(np.iinfo(np.int8).max) if max_abs > 0.0 else 1.0
        info = np.iinfo(np.int8)
        quantized = np.clip(np.rint(coefficients / scale), info.min, info.max).astype(np.int8)
        return clipped_count, scale, quantized.tobytes()

    def _quantize_measurements(self, measurements: np.ndarray) -> tuple[int, float, float, bytes]:
        measurements_flat = measurements.flatten().astype(np.float64)
        if self.measurement_dtype == "float64":
            return IMCS_QUANT_FLOAT64, 1.0, 0.0, measurements_flat.tobytes()

        max_abs = float(np.max(np.abs(measurements_flat))) if measurements_flat.size else 0.0
        if max_abs <= 0.0:
            scale = 1.0
        elif self.measurement_dtype == "int8":
            scale = max_abs / float(np.iinfo(np.int8).max)
        else:
            scale = max_abs / float(np.iinfo(np.int16).max)

        if self.measurement_dtype == "int8":
            info = np.iinfo(np.int8)
            quantized = np.clip(np.rint(measurements_flat / scale), info.min, info.max).astype(
                np.int8
            )
            return IMCS_QUANT_INT8, scale, 0.0, quantized.tobytes()

        info = np.iinfo(np.int16)
        quantized = np.clip(np.rint(measurements_flat / scale), info.min, info.max).astype(
            np.int16
        )
        return IMCS_QUANT_INT16, scale, 0.0, quantized.tobytes()

    def _serialize(
        self,
        measurements: np.ndarray,
        original_shape: tuple,
        m_row: int,
        m_col: int,
        block_h: int = 0,
        block_w: int = 0,
        content_shape: Optional[tuple] = None,
        block_means: Optional[np.ndarray] = None,
        low_frequency_coefficients: Optional[np.ndarray] = None,
    ) -> bytes:
        # Map sparsity basis to ID
        basis_map = {"dct": 0, "wavelet": 1}
        basis_id = basis_map.get(self.sparsity_basis, 0)

        # Map matrix type to ID
        matrix_map = {"gaussian": 0, "bernoulli": 1, "random": 2}
        matrix_id = matrix_map.get(self.matrix_type, 0)

        # Flags
        is_2d = len(original_shape) == 2
        flags = IMCS_FLAG_IS_2D if is_2d else 0
        if self.measurement_mode == "per_block":
            flags |= IMCS_FLAG_PER_BLOCK
        block_mean_bytes = b""
        if block_means is not None and block_means.size:
            flags |= IMCS_FLAG_BLOCK_MEAN
            means_uint8 = np.clip(np.rint(block_means), 0, 255).astype(np.uint8)
            block_mean_bytes = means_uint8.tobytes()

        low_frequency_count, low_frequency_scale, low_frequency_bytes = (
            self._quantize_low_frequency_coefficients(low_frequency_coefficients)
        )
        if low_frequency_count:
            flags |= IMCS_FLAG_LOW_FREQUENCY

        quantization_id, quantization_scale, quantization_offset, measurements_bytes = (
            self._quantize_measurements(measurements)
        )

        if content_shape is None:
            content_shape = original_shape

        ch = content_shape[0] if len(content_shape) > 0 else 0
        cw = content_shape[1] if len(content_shape) > 1 else 0

        # Build header (v2: padded shape in original_*; block/content meta in extension)
        header = struct.pack(
            IMCS_HEADER_V1_FMT,
            b"IMCS",  # Magic number
            self.FORMAT_VERSION,  # Version
            flags,  # Flags
            basis_id,  # Sparsity basis
            matrix_id,  # Matrix type
            self.seed,  # Seed for reproducibility
            original_shape[0],  # Height / length (padded for block 2D)
            original_shape[1] if is_2d else 0,  # Width (0 for 1D)
            m_row,  # M_row (per-block M when block mode)
            m_col,  # M_col (0: 1D or legacy 2D; >0: number of blocks)
            len(low_frequency_bytes)
            + (8 if low_frequency_count else 0)
            + len(block_mean_bytes)
            + len(measurements_bytes),  # Data length
        )
        header += struct.pack(IMCS_HEADER_V2_EXT_FMT, block_h, block_w, ch, cw)
        header += struct.pack(
            IMCS_HEADER_V3_EXT_FMT,
            quantization_id,
            low_frequency_count,
            0,
            float(quantization_scale),
            float(quantization_offset),
        )

        low_frequency_scale_bytes = (
            struct.pack("<d", float(low_frequency_scale)) if low_frequency_count else b""
        )
        return header + low_frequency_scale_bytes + low_frequency_bytes + block_mean_bytes + measurements_bytes

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
            "block_size": self.block_size,
            "measurement_mode": self.measurement_mode,
            "measurement_dtype": self.measurement_dtype,
            "block_mean_residual": self.block_mean_residual,
            "low_frequency_coeffs": self.low_frequency_coeffs,
        }
