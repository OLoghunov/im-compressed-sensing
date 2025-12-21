import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image


def setup_argument_parser() -> argparse.ArgumentParser:
    """Создает и настраивает ArgumentParser для CLI."""
    parser = argparse.ArgumentParser(
        description="IMCS - Сжатие изображений с помощью Compressed Sensing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py                                # Обработать все изображения (с визуализацией)
  python main.py --input test_gradient.png      # Конкретный файл
  python main.py --ratio 0.8 --algorithm ista   # С параметрами
  python main.py --no-visualize                 # Без визуализации

Алгоритмы:
  omp   - Orthogonal Matching Pursuit (быстрый, но менее точный)
  ista  - Iterative Shrinkage-Thresholding (медленный, но точный)
  sa    - Simulated Annealing (гибкий)
        """,
    )

    parser.add_argument(
        "--input", type=str, help="Имя файла в examples/input/ (если не указан, обрабатываются все)"
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Compression ratio (0.1-0.9, рекомендуется ≥0.5 для разреженных изображений)",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="omp",
        choices=["omp", "ista", "sa"],
        help="Алгоритм восстановления (по умолчанию: omp)",
    )

    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        default=True,
        help="Отключить визуализацию сходимости (по умолчанию: включена)",
    )

    return parser


def find_input_files(input_arg: Optional[str], input_dir: Path) -> List[Path]:
    if input_arg:
        exact_file = input_dir / input_arg
        if exact_file.exists():
            return [exact_file]
        else:
            matching = (
                list(input_dir.glob(f"*{input_arg}*.png"))
                + list(input_dir.glob(f"*{input_arg}*.jpg"))
                + list(input_dir.glob(f"*{input_arg}*.npy"))
                + list(input_dir.glob(f"*{input_arg}*.txt"))
            )
            if len(matching) == 0:
                print(f"❌ Файл не найден: {input_arg}")
                print(f"Искал в: {input_dir}/")
                return []
            else:
                if len(matching) > 1:
                    print(f"✓ Найдено {len(matching)} файлов для '{input_arg}':")
                    for f in matching:
                        print(f"  - {f.name}")
                    print()
                else:
                    print(f"✓ Найдено: {matching[0].name}")
                return matching
    else:
        files = (
            list(input_dir.glob("*.png"))
            + list(input_dir.glob("*.jpg"))
            + list(input_dir.glob("*.npy"))
            + list(input_dir.glob("*.txt"))
        )
        return files


def load_image(image_path: Path) -> Optional[np.ndarray]:
    if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img = Image.open(image_path)

        if img.mode != "L":
            print("  Конвертация RGB → Grayscale")
            img = img.convert("L")  # type: ignore

        data = np.array(img, dtype=np.float64)
        print(f"  Загружено: {img.size[0]}×{img.size[1]} пикселей")
        return data
    else:
        print(f"❌ Неподдерживаемый формат: {image_path.suffix}")
        return None


def save_image(data: np.ndarray, output_path: Path):
    img_data = np.clip(data, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode="L")
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
):
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЁТ О СЖАТИИ IMCS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Файл: {image_path.name}\n")
        f.write(f"Размер: {original_shape}\n")
        f.write(f"Compression ratio: {compression_ratio}\n")
        f.write(f"Алгоритм: {algorithm.upper()}\n\n")
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


def load_signal(signal_path: Path) -> Optional[np.ndarray]:
    """Загружает 1D сигнал из .npy или .txt файла."""
    suffix = signal_path.suffix.lower()

    if suffix == ".npy":
        signal = np.load(signal_path)
        if signal.ndim != 1:
            print(f"❌ Ожидается 1D массив, получен {signal.ndim}D")
            return None
        print(f"  Загружено: {len(signal)} отсчётов")
        return signal
    elif suffix == ".txt":
        signal = np.loadtxt(signal_path)
        if signal.ndim != 1:
            print(f"❌ Ожидается 1D массив, получен {signal.ndim}D")
            return None
        print(f"  Загружено: {len(signal)} отсчётов")
        return signal
    else:
        print(f"❌ Неподдерживаемый формат для 1D сигнала: {suffix}")
        print("  Поддерживаются: .npy, .txt")
        return None


def save_signal(signal: np.ndarray, output_path: Path):
    """Сохраняет 1D сигнал в .npy файл."""
    np.save(output_path, signal)


def visualize_signal_comparison(original: np.ndarray, reconstructed: np.ndarray, output_path: Path):
    """Создаёт график сравнения исходного и восстановленного сигнала."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # График 1: Оба сигнала
    ax1.plot(original, "b-", label="Оригинал", linewidth=1.5, alpha=0.7)
    ax1.plot(reconstructed, "r--", label="Восстановлено", linewidth=1.5, alpha=0.7)
    ax1.set_title("Сравнение сигналов", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Отсчёт")
    ax1.set_ylabel("Амплитуда")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # График 2: Ошибка
    error = original - reconstructed
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Сохранено: {output_path.name}")
