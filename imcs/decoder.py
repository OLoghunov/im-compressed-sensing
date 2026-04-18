from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import time
import numpy as np
from typing import Callable, Optional, Union
from pathlib import Path
import struct
from PIL import Image

from .utils import (
    IMCS_HEADER_V1_FMT,
    IMCS_HEADER_V1_SIZE,
    IMCS_HEADER_V2_EXT_FMT,
    IMCS_HEADER_V2_SIZE,
    generate_measurement_matrix,
    omp,
    ista,
    fista,
    simulated_annealing,
    create_basis_matrix,
    create_2d_basis_matrix,
)


DEFAULT_MIN_PARALLEL_BLOCKS = 16


def _run_reconstruction_algorithm(
    reconstruction_algorithm: str,
    y: np.ndarray,
    A: np.ndarray,
    *,
    max_iter: int,
    return_history: bool,
    sparsity: Optional[int] = None,
    adaptive_lambda: Optional[float] = None,
):
    if reconstruction_algorithm == "omp":
        if sparsity is None:
            raise ValueError("sparsity is required for OMP algorithm")
        return omp(y, A, sparsity, return_history=return_history)
    if reconstruction_algorithm in ["basis_pursuit", "iterative_threshold"]:
        if adaptive_lambda is None:
            raise ValueError("adaptive_lambda is required for ISTA algorithm")
        return ista(y, A, adaptive_lambda, max_iter, return_history=return_history)
    if reconstruction_algorithm == "fista":
        if adaptive_lambda is None:
            raise ValueError("adaptive_lambda is required for FISTA algorithm")
        return fista(y, A, adaptive_lambda, max_iter, return_history=return_history)
    if reconstruction_algorithm == "simulated_annealing":
        if adaptive_lambda is None:
            raise ValueError("adaptive_lambda is required for SA algorithm")
        return simulated_annealing(
            y, A, adaptive_lambda, max_iter=max_iter, return_history=return_history
        )
    if sparsity is None:
        raise ValueError("sparsity is required for OMP algorithm")
    return omp(y, A, sparsity, return_history=return_history)


def _empty_decode_profile() -> dict[str, float | int]:
    return {
        "deserialize_s": 0.0,
        "basis_build_s": 0.0,
        "phi_build_s": 0.0,
        "sensing_matrix_build_s": 0.0,
        "reconstruction_s": 0.0,
        "inverse_transform_s": 0.0,
        "assemble_blocks_s": 0.0,
        "total_decode_s": 0.0,
        "blocks_total": 0,
        "avg_block_reconstruction_s": 0.0,
        "max_block_reconstruction_s": 0.0,
        "parallel_workers": 1,
    }


def _merge_decode_profiles(
    base: dict[str, float | int], update: dict[str, float | int]
) -> dict[str, float | int]:
    for key, value in update.items():
        if key == "max_block_reconstruction_s":
            base[key] = max(float(base.get(key, 0.0)), float(value))
        elif key == "parallel_workers":
            base[key] = max(int(base.get(key, 1)), int(value))
        elif key == "blocks_total":
            base[key] = int(base.get(key, 0)) + int(value)
        elif key in base:
            base[key] = float(base.get(key, 0.0)) + float(value)
        else:
            base[key] = value
    return base


def _finalize_decode_profile(profile: Optional[dict[str, float | int]]) -> Optional[dict]:
    if profile is None:
        return None
    blocks_total = int(profile.get("blocks_total", 0))
    reconstruction_total = float(profile.get("reconstruction_s", 0.0))
    profile["avg_block_reconstruction_s"] = (
        reconstruction_total / blocks_total if blocks_total > 0 else 0.0
    )
    return dict(profile)


def _decode_block_chunk(task: dict) -> tuple[int, np.ndarray, dict[str, float | int]]:
    start_idx = int(task["start_idx"])
    y_chunk = np.asarray(task["y_chunk"], dtype=np.float64)
    bh = int(task["bh"])
    bw = int(task["bw"])
    b_pixels = bh * bw
    m_per_block = int(task["m_per_block"])
    basis = str(task["basis"])
    matrix_type = str(task["matrix_type"])
    measurement_mode = str(task["measurement_mode"])
    seed = int(task["seed"])
    reconstruction_algorithm = str(task["reconstruction_algorithm"])
    max_iter = int(task["max_iter"])
    adaptive_lambda = float(task["adaptive_lambda"])
    sparsity = int(task["sparsity"])

    profile = _empty_decode_profile()

    t0 = time.perf_counter()
    Psi_2d = create_2d_basis_matrix(bh, bw, basis)
    profile["basis_build_s"] = time.perf_counter() - t0

    shared_A: Optional[np.ndarray] = None
    if measurement_mode == "shared" and len(y_chunk) > 0:
        t0 = time.perf_counter()
        shared_phi = generate_measurement_matrix(m_per_block, b_pixels, matrix_type, seed)
        profile["phi_build_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        shared_A = shared_phi @ Psi_2d.T
        profile["sensing_matrix_build_s"] = time.perf_counter() - t0

    blocks = np.zeros((len(y_chunk), bh, bw), dtype=np.float64)
    for offset, y_b in enumerate(y_chunk):
        global_idx = start_idx + offset
        A = shared_A
        if A is None:
            phi_seed = seed if measurement_mode == "shared" else seed + global_idx
            t0 = time.perf_counter()
            Phi = generate_measurement_matrix(m_per_block, b_pixels, matrix_type, phi_seed)
            profile["phi_build_s"] = float(profile["phi_build_s"]) + (time.perf_counter() - t0)
            t0 = time.perf_counter()
            A = Phi @ Psi_2d.T
            profile["sensing_matrix_build_s"] = float(
                profile["sensing_matrix_build_s"]
            ) + (time.perf_counter() - t0)

        t0 = time.perf_counter()
        s = _run_reconstruction_algorithm(
            reconstruction_algorithm,
            y_b,
            A,
            max_iter=max_iter,
            return_history=False,
            sparsity=sparsity,
            adaptive_lambda=adaptive_lambda,
        )
        reconstruction_s = time.perf_counter() - t0
        profile["reconstruction_s"] = float(profile["reconstruction_s"]) + reconstruction_s
        profile["max_block_reconstruction_s"] = max(
            float(profile["max_block_reconstruction_s"]), reconstruction_s
        )

        t0 = time.perf_counter()
        x_vec = Psi_2d.T @ s
        blocks[offset] = x_vec.reshape(bh, bw, order="F")
        profile["inverse_transform_s"] = float(profile["inverse_transform_s"]) + (
            time.perf_counter() - t0
        )

    profile["blocks_total"] = len(y_chunk)
    return start_idx, blocks, profile


class IMCSDecoder:
    def __init__(
        self,
        reconstruction_algorithm: str = "omp",
        max_iter: int = 1000,
        lambda_param: float = 10.0,
        parallel_block_workers: Optional[int] = None,
        min_parallel_blocks: int = DEFAULT_MIN_PARALLEL_BLOCKS,
        log_fn: Optional[Callable[[str], None]] = None,
        progress_interval_s: float = 0.5,
    ):
        valid_algorithms = [
            "omp",
            "basis_pursuit",
            "iterative_threshold",
            "fista",
            "simulated_annealing",
        ]
        if reconstruction_algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")

        self.reconstruction_algorithm = reconstruction_algorithm
        self.max_iter = max_iter
        self.lambda_param = lambda_param
        self.parallel_block_workers = parallel_block_workers
        self.min_parallel_blocks = max(1, min_parallel_blocks)
        self.log_fn = log_fn
        self.progress_interval_s = max(0.1, float(progress_interval_s))

        self.last_history = None
        self.last_residuals = None
        self.last_profile: Optional[dict] = None
        self.return_history = False
        self.profile_enabled = False
        self._last_progress_log_t = 0.0

    def _record_profile(self, key: str, value: float) -> None:
        if self.last_profile is None:
            return
        self.last_profile[key] = float(self.last_profile.get(key, 0.0)) + float(value)

    def _set_profile_value(self, key: str, value: int | float) -> None:
        if self.last_profile is None:
            return
        self.last_profile[key] = value

    def _log(self, message: str) -> None:
        if self.log_fn is not None:
            self.log_fn(message)

    def _log_progress(
        self,
        label: str,
        completed: int,
        total: int,
        started_t: float,
        *,
        force: bool = False,
    ) -> None:
        if self.log_fn is None or total <= 0:
            return
        now = time.perf_counter()
        if not force and (now - self._last_progress_log_t) < self.progress_interval_s:
            return
        elapsed = max(now - started_t, 1e-9)
        rate = completed / elapsed
        eta = max(total - completed, 0) / rate if rate > 1e-9 else 0.0
        self._last_progress_log_t = now
        self._log(
            f"{label}: {completed}/{total} ({(100.0 * completed / total):.1f}%), "
            f"elapsed={elapsed:.2f} c, rate={rate:.1f} blk/s, eta={eta:.2f} c"
        )

    def _resolve_parallel_workers(self, n_blocks: int) -> int:
        if self.return_history or n_blocks < self.min_parallel_blocks:
            return 1
        if self.parallel_block_workers is not None:
            return max(1, min(int(self.parallel_block_workers), n_blocks))
        if mp.current_process().name != "MainProcess":
            return 1
        return 1

    def decode(
        self, compressed_data: bytes, return_history: bool = False, profile: bool = False
    ) -> np.ndarray:
        self.return_history = return_history
        self.profile_enabled = profile
        self.last_history = None
        self.last_residuals = None
        self.last_profile = _empty_decode_profile() if profile else None
        self._last_progress_log_t = 0.0

        total_t0 = time.perf_counter()
        self._log(
            f"[decode] start algorithm={self.reconstruction_algorithm.upper()}, "
            f"return_history={'yes' if return_history else 'no'}, profile={'yes' if profile else 'no'}"
        )

        t0 = time.perf_counter()
        metadata, measurements = self._deserialize(compressed_data)
        self._record_profile("deserialize_s", time.perf_counter() - t0)
        self._log(
            f"[decode] deserialized format={'2d' if metadata['is_2d'] else '1d'}, "
            f"basis={metadata['sparsity_basis']}, matrix={metadata['matrix_type']}, "
            f"measurement_mode={metadata.get('measurement_mode', 'n/a')}, "
            f"measurements={measurements.size}"
        )

        if metadata["is_2d"]:
            reconstructed = self._decode_2d(measurements, metadata)
        else:
            reconstructed = self._decode_1d(measurements, metadata)

        self._set_profile_value("total_decode_s", time.perf_counter() - total_t0)
        self.last_profile = _finalize_decode_profile(self.last_profile)
        self._log(f"[decode] done total={time.perf_counter() - total_t0:.3f} c")
        return reconstructed

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
        result = _run_reconstruction_algorithm(
            self.reconstruction_algorithm,
            y,
            A,
            max_iter=self.max_iter,
            return_history=self.return_history,
            sparsity=sparsity,
            adaptive_lambda=adaptive_lambda,
        )

        if self.return_history:
            s, self.last_history, self.last_residuals = result
        else:
            s = result

        return s

    def _deserialize(self, data: bytes) -> tuple[dict, np.ndarray]:
        if len(data) < IMCS_HEADER_V1_SIZE:
            raise ValueError("Invalid IMCS data: too short")
        if data[:4] != b"IMCS":
            raise ValueError("Invalid IMCS data: wrong magic number")

        header = struct.unpack(IMCS_HEADER_V1_FMT, data[:IMCS_HEADER_V1_SIZE])
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

        header_total = IMCS_HEADER_V1_SIZE
        is_2d = bool(flags & 1)

        if version >= 2 and len(data) >= IMCS_HEADER_V2_SIZE:
            block_h, block_w, content_h, content_w = struct.unpack(
                IMCS_HEADER_V2_EXT_FMT, data[IMCS_HEADER_V1_SIZE:IMCS_HEADER_V2_SIZE]
            )
            header_total = IMCS_HEADER_V2_SIZE
        else:
            block_h, block_w = 0, 0
            if is_2d:
                content_h, content_w = orig_height, orig_width
            else:
                content_h, content_w = orig_height, 0

        # Map IDs back to strings
        basis_map = {0: "dct", 1: "wavelet"}
        matrix_map = {0: "gaussian", 1: "bernoulli", 2: "random"}

        metadata = {
            "version": version,
            "is_2d": is_2d,
            "sparsity_basis": basis_map.get(basis_id, "dct"),
            "matrix_type": matrix_map.get(matrix_id, "gaussian"),
            "measurement_mode": "per_block" if (flags & 0b10) else "shared",
            "seed": seed,
            "original_shape": (orig_height, orig_width) if is_2d else (orig_height,),
            "m_row": m_row,
            "m_col": m_col,
            "block_h": block_h,
            "block_w": block_w,
            "content_shape": (content_h, content_w) if is_2d else (content_h,),
        }

        # Extract measurements
        measurements_bytes = data[header_total : header_total + data_length]
        measurements = np.frombuffer(measurements_bytes, dtype=np.float64)

        return metadata, measurements

    def _decode_1d(self, y: np.ndarray, metadata: dict) -> np.ndarray:
        n = metadata["original_shape"][0]
        m = metadata["m_row"]
        self._log(f"[decode-1d] n={n}, m={m}")

        # Regenerate the same measurement matrix using the stored seed
        self._log("[decode-1d] building Phi...")
        t0 = time.perf_counter()
        Phi = generate_measurement_matrix(m, n, metadata["matrix_type"], metadata["seed"])
        self._record_profile("phi_build_s", time.perf_counter() - t0)

        # For 1D, we use the selected orthonormal basis: A = Φ·Ψ^T
        self._log("[decode-1d] building basis...")
        t0 = time.perf_counter()
        Psi = create_basis_matrix(n, metadata["sparsity_basis"])
        self._record_profile("basis_build_s", time.perf_counter() - t0)

        # Sensing matrix in DCT domain: A = Φ·Ψ^T
        self._log("[decode-1d] building sensing matrix A...")
        t0 = time.perf_counter()
        A = Phi @ Psi.T
        self._record_profile("sensing_matrix_build_s", time.perf_counter() - t0)

        # Target sparsity (rough estimate)
        sparsity = max(int(0.2 * n), 5)

        # Adaptive lambda: scale with signal size
        adaptive_lambda = self.lambda_param * (n / 1000.0)

        # Reconstruct sparse coefficients
        self._log("[decode-1d] running reconstruction...")
        t0 = time.perf_counter()
        s = self._run_reconstruction(y, A, sparsity=sparsity, adaptive_lambda=adaptive_lambda)
        self._record_profile("reconstruction_s", time.perf_counter() - t0)

        # Transform back from sparse domain: x = Ψ^T · s
        self._log("[decode-1d] inverse transform...")
        t0 = time.perf_counter()
        x = Psi.T @ s
        self._record_profile("inverse_transform_s", time.perf_counter() - t0)

        return x

    def _decode_2d(self, y: np.ndarray, metadata: dict) -> np.ndarray:
        if metadata.get("block_h", 0) and metadata.get("block_w", 0):
            return self._decode_2d_blocked(y, metadata)
        n_row, n_col = metadata["original_shape"]
        m_total = metadata["m_row"]  # Total measurements
        n_total = n_row * n_col
        self._log(
            f"[decode-2d] full-frame shape={n_row}x{n_col}, "
            f"pixels={n_total}, measurements={m_total}"
        )

        # Regenerate FULL measurement matrix
        self._log("[decode-2d] building full Phi...")
        t0 = time.perf_counter()
        Phi = generate_measurement_matrix(
            m_total, n_total, metadata["matrix_type"], metadata["seed"]
        )
        self._record_profile("phi_build_s", time.perf_counter() - t0)

        self._log("[decode-2d] building 2D basis...")
        t0 = time.perf_counter()
        Psi_2d = create_2d_basis_matrix(n_row, n_col, metadata["sparsity_basis"])
        self._record_profile("basis_build_s", time.perf_counter() - t0)

        # Sensing matrix in DCT domain: A = Φ · Ψ^T
        self._log("[decode-2d] building sensing matrix A...")
        t0 = time.perf_counter()
        A = Phi @ Psi_2d.T
        self._record_profile("sensing_matrix_build_s", time.perf_counter() - t0)

        # Adaptive sparsity based on measurements
        sparsity = max(min(m_total // 2, n_total // 4), 1)

        # Adaptive lambda: scale with image size
        # Smaller images need smaller lambda
        adaptive_lambda = self.lambda_param * (n_total / 1000.0)

        # Solve: y = A · s
        self._log("[decode-2d] running reconstruction...")
        t0 = time.perf_counter()
        s = self._run_reconstruction(y, A, sparsity=sparsity, adaptive_lambda=adaptive_lambda)
        self._record_profile("reconstruction_s", time.perf_counter() - t0)

        # Transform back: x = Ψ^T · s
        self._log("[decode-2d] inverse transform...")
        t0 = time.perf_counter()
        x_vec = Psi_2d.T @ s
        self._record_profile("inverse_transform_s", time.perf_counter() - t0)

        # Reshape to 2D (column-major)
        X = x_vec.reshape(n_row, n_col, order="F")

        return X

    def _decode_2d_blocked(self, y: np.ndarray, metadata: dict) -> np.ndarray:
        """Decode 2D image encoded as independent CS blocks (JPEG-style tiling)."""
        pad_h, pad_w = metadata["original_shape"]
        bh = metadata["block_h"]
        bw = metadata["block_w"]
        content_h, content_w = metadata["content_shape"]
        m_per_block = metadata["m_row"]
        n_blocks = metadata["m_col"]
        b_pixels = bh * bw
        measurement_mode = metadata.get("measurement_mode", "shared")

        if y.size != m_per_block * n_blocks:
            raise ValueError(
                f"Measurement count mismatch: expected {m_per_block * n_blocks}, got {y.size}"
            )

        n_blocks_h = pad_h // bh
        n_blocks_w = pad_w // bw
        if n_blocks_h * n_blocks_w != n_blocks:
            raise ValueError("Block grid does not match padded image shape")

        sparsity = max(min(m_per_block // 2, b_pixels // 4), 1)
        adaptive_lambda = self.lambda_param * (b_pixels / 1000.0)

        X_pad = np.zeros((pad_h, pad_w), dtype=np.float64)
        y_blocks = y.reshape(n_blocks, m_per_block)

        workers = self._resolve_parallel_workers(n_blocks)
        self._set_profile_value("parallel_workers", workers)
        self._log(
            f"[decode-2d-blocked] content={content_h}x{content_w}, padded={pad_h}x{pad_w}, "
            f"block={bh}x{bw}, blocks={n_blocks_h}x{n_blocks_w}={n_blocks}, "
            f"m_per_block={m_per_block}, mode={measurement_mode}, workers={workers}"
        )

        if workers > 1:
            chunk_size = max(1, (n_blocks + workers * 4 - 1) // (workers * 4))
            tasks = []
            for start_idx in range(0, n_blocks, chunk_size):
                stop_idx = min(start_idx + chunk_size, n_blocks)
                tasks.append(
                    {
                        "start_idx": start_idx,
                        "y_chunk": y_blocks[start_idx:stop_idx],
                        "bh": bh,
                        "bw": bw,
                        "m_per_block": m_per_block,
                        "basis": metadata["sparsity_basis"],
                        "matrix_type": metadata["matrix_type"],
                        "measurement_mode": measurement_mode,
                        "seed": metadata["seed"],
                        "reconstruction_algorithm": self.reconstruction_algorithm,
                        "max_iter": self.max_iter,
                        "adaptive_lambda": adaptive_lambda,
                        "sparsity": sparsity,
                    }
                )
            self._log(
                f"[decode-2d-blocked] parallel reconstruction started, "
                f"chunk_size={chunk_size}, chunks={len(tasks)}"
            )
            ctx = mp.get_context("spawn")
            progress_t0 = time.perf_counter()
            completed_blocks = 0
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = [executor.submit(_decode_block_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    start_idx, blocks, profile = future.result()
                    _merge_decode_profiles(self.last_profile or {}, profile)
                    t0 = time.perf_counter()
                    for offset, block in enumerate(blocks):
                        idx = start_idx + offset
                        bi = idx // n_blocks_w
                        bj = idx % n_blocks_w
                        X_pad[bi * bh : (bi + 1) * bh, bj * bw : (bj + 1) * bw] = block
                    self._record_profile("assemble_blocks_s", time.perf_counter() - t0)
                    completed_blocks += len(blocks)
                    self._log_progress(
                        "[decode-2d-blocked] progress",
                        completed_blocks,
                        n_blocks,
                        progress_t0,
                        force=(completed_blocks == n_blocks),
                    )
        else:
            self._log("[decode-2d-blocked] serial reconstruction started")
            t0 = time.perf_counter()
            Psi_2d = create_2d_basis_matrix(bh, bw, metadata["sparsity_basis"])
            self._record_profile("basis_build_s", time.perf_counter() - t0)
            shared_A: Optional[np.ndarray] = None
            if measurement_mode == "shared":
                self._log("[decode-2d-blocked] building shared Phi...")
                t0 = time.perf_counter()
                shared_phi = generate_measurement_matrix(
                    m_per_block, b_pixels, metadata["matrix_type"], metadata["seed"]
                )
                self._record_profile("phi_build_s", time.perf_counter() - t0)
                self._log("[decode-2d-blocked] building shared sensing matrix A...")
                t0 = time.perf_counter()
                shared_A = shared_phi @ Psi_2d.T
                self._record_profile("sensing_matrix_build_s", time.perf_counter() - t0)

            idx = 0
            progress_t0 = time.perf_counter()
            for bi in range(n_blocks_h):
                for bj in range(n_blocks_w):
                    y_b = y_blocks[idx]
                    A = shared_A
                    if A is None:
                        phi_seed = (
                            metadata["seed"]
                            if measurement_mode == "shared"
                            else metadata["seed"] + idx
                        )
                        t0 = time.perf_counter()
                        Phi = generate_measurement_matrix(
                            m_per_block, b_pixels, metadata["matrix_type"], phi_seed
                        )
                        self._record_profile("phi_build_s", time.perf_counter() - t0)
                        t0 = time.perf_counter()
                        A = Phi @ Psi_2d.T
                        self._record_profile("sensing_matrix_build_s", time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    s = self._run_reconstruction(
                        y_b, A, sparsity=sparsity, adaptive_lambda=adaptive_lambda
                    )
                    reconstruction_s = time.perf_counter() - t0
                    self._record_profile("reconstruction_s", reconstruction_s)
                    if self.last_profile is not None:
                        self.last_profile["max_block_reconstruction_s"] = max(
                            float(self.last_profile["max_block_reconstruction_s"]),
                            reconstruction_s,
                        )
                        self.last_profile["blocks_total"] = int(self.last_profile["blocks_total"]) + 1
                    t0 = time.perf_counter()
                    x_vec = Psi_2d.T @ s
                    block = x_vec.reshape(bh, bw, order="F")
                    self._record_profile("inverse_transform_s", time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    X_pad[bi * bh : (bi + 1) * bh, bj * bw : (bj + 1) * bw] = block
                    self._record_profile("assemble_blocks_s", time.perf_counter() - t0)
                    idx += 1
                    self._log_progress(
                        "[decode-2d-blocked] progress",
                        idx,
                        n_blocks,
                        progress_t0,
                        force=(idx == n_blocks),
                    )

        return X_pad[:content_h, :content_w]

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
        valid_algorithms = [
            "omp",
            "basis_pursuit",
            "iterative_threshold",
            "fista",
            "simulated_annealing",
        ]
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        self.reconstruction_algorithm = algorithm

    def get_info(self, compressed_data: bytes) -> dict:
        metadata, measurements = self._deserialize(compressed_data)

        if metadata.get("is_2d") and metadata.get("block_h", 0):
            ch, cw = metadata["content_shape"]
            original_size = ch * cw * 8
        else:
            original_size = np.prod(metadata["original_shape"]) * 8  # float64

        compressed_size = measurements.size * 8

        return {
            **metadata,
            "compressed_shape": measurements.shape,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "actual_compression_ratio": compressed_size / original_size,
        }
