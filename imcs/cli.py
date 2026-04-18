from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_image(
    image_path: Path, verbose: bool = True, preserve_color: bool = False
) -> Optional[np.ndarray]:
    if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img = Image.open(image_path)

        if preserve_color:
            if img.mode not in {"RGB", "RGBA"}:
                img = img.convert("RGB")  # type: ignore
            elif img.mode == "RGBA":
                img = img.convert("RGB")  # type: ignore
        elif img.mode != "L":
            if verbose:
                print("  Конвертация RGB → Grayscale")
            img = img.convert("L")  # type: ignore

        data = np.array(img, dtype=np.float64)
        if verbose:
            mode_note = f", режим: {img.mode}" if preserve_color else ""
            print(f"  Загружено: {img.size[0]}×{img.size[1]} пикселей{mode_note}")
        return data
    else:
        if verbose:
            print(f"❌ Неподдерживаемый формат: {image_path.suffix}")
        return None


def save_image(data: np.ndarray, output_path: Path):
    img_data = np.clip(data, 0, 255).astype(np.uint8)
    if img_data.ndim == 2:
        img = Image.fromarray(img_data, mode="L")
    elif img_data.ndim == 3 and img_data.shape[2] == 3:
        img = Image.fromarray(img_data, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape for save_image: {img_data.shape}")
    img.save(output_path)


def create_output_directory(output_dir: Path, image_name: str) -> Path:
    output_subdir = output_dir / image_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    return output_subdir


def save_report(
    report_path: Path,
    image_path: Path,
    original_shape: tuple,
    compression_ratio: float,
    algorithm: str,
    t_encode: float,
    t_decode: float,
    original_size: int,
    compressed_size: int,
    metrics: dict,
    quality: str,
    extra_fields: Optional[dict] = None,
):
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЁТ О СЖАТИИ IMCS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Файл: {image_path.name}\n")
        f.write(f"Размер: {original_shape}\n")
        f.write(f"Compression ratio: {compression_ratio}\n")
        f.write(f"Алгоритм: {algorithm.upper()}\n\n")
        if extra_fields:
            for key, value in extra_fields.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        f.write(f"Время кодирования: {t_encode:.3f} сек\n")
        f.write(f"Время декодирования: {t_decode:.3f} сек\n\n")
        f.write(f"Исходный размер: {original_size} байт\n")
        f.write(f"Сжатый размер: {compressed_size} байт\n")
        f.write(f"Степень сжатия: {original_size / compressed_size:.2f}x\n\n")
        f.write("=" * 80 + "\n")
        f.write("МЕТРИКИ КАЧЕСТВА\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"MSE:  {metrics['mse']:.2f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"MAE:  {metrics['mae']:.2f}\n\n")
        if "ssim" in metrics:
            f.write(f"SSIM: {metrics['ssim']:.4f}\n\n")
        f.write(f"Качество: {quality}\n")


def interpret_psnr(psnr: float) -> str:
    if psnr > 30:
        return "отличное ✓"
    elif psnr > 20:
        return "хорошее"
    elif psnr > 15:
        return "удовлетворительное"
    else:
        return "плохое ⚠️"


def load_signal(signal_path: Path, verbose: bool = True) -> Optional[np.ndarray]:
    """Загружает 1D сигнал из .npy или .txt файла."""
    suffix = signal_path.suffix.lower()

    if suffix == ".npy":
        signal = np.load(signal_path)
        if signal.ndim != 1:
            if verbose:
                print(f"❌ Ожидается 1D массив, получен {signal.ndim}D")
            return None
        if verbose:
            print(f"  Загружено: {len(signal)} отсчётов")
        return signal
    elif suffix == ".txt":
        signal = np.loadtxt(signal_path)
        if signal.ndim != 1:
            if verbose:
                print(f"❌ Ожидается 1D массив, получен {signal.ndim}D")
            return None
        if verbose:
            print(f"  Загружено: {len(signal)} отсчётов")
        return signal
    else:
        if verbose:
            print(f"❌ Неподдерживаемый формат для 1D сигнала: {suffix}")
            print("  Поддерживаются: .npy, .txt")
        return None


def save_signal(signal: np.ndarray, output_path: Path):
    """Сохраняет 1D сигнал в .npy файл."""
    np.save(output_path, signal)


def visualize_signal_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
    verbose: bool = True,
):
    """Создаёт график сравнения исходного и восстановленного сигнала.

    Без pyplot.subplots: при уже открытом окне GUI второй pyplot-figure на macOS
    часто даёт segmentation fault.
    """
    from matplotlib.figure import Figure

    o = np.asarray(original, dtype=np.float64).ravel()
    r = np.asarray(reconstructed, dtype=np.float64).ravel()

    fig = Figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(o, "b-", label="Оригинал", linewidth=1.5, alpha=0.7)
    ax1.plot(r, "r--", label="Восстановлено", linewidth=1.5, alpha=0.7)
    ax1.set_title("Сравнение сигналов", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Отсчёт")
    ax1.set_ylabel("Амплитуда")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    error = o - r
    ax2.plot(error, "g-", linewidth=1.5)
    ax2.fill_between(range(len(error)), error, alpha=0.3, color="green")
    ax2.set_title(
        f"Ошибка восстановления (MAE: {np.abs(error).mean():.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Отсчёт")
    ax2.set_ylabel("Ошибка")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"   Сохранено: {output_path.name}")
