"""
Главный файл для демонстрации IMCS кодека.

Использование:
    python main.py                                    # Обработать все изображения
    python main.py --input test_gradient.png          # Конкретный файл
    python main.py --ratio 0.8 --algorithm ista       # С параметрами
"""

import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image

from imcs import IMCSEncoder, IMCSDecoder
from imcs.utils import calculate_compression_metrics


def load_image(image_path: Path) -> np.ndarray:
    """Загружает изображение из файла."""
    if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img = Image.open(image_path)

        # Конвертируем в grayscale если цветное
        if img.mode != "L":
            print("  Конвертация RGB → Grayscale")
            img = img.convert("L")

        # Конвертируем в numpy array
        data = np.array(img, dtype=np.float64)
        print(f"  Загружено: {img.size[0]}×{img.size[1]} пикселей")
        return data
    else:
        print(f"❌ Неподдерживаемый формат: {image_path.suffix}")
        return None


def save_image(data: np.ndarray, output_path: Path):
    """Сохраняет изображение в PNG."""
    img_data = np.clip(data, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode="L")
    img.save(output_path)


def process_image(image_path: Path, output_dir: Path, compression_ratio: float, algorithm: str):
    """
    Обрабатывает одно изображение: кодирует и декодирует.

    Args:
        image_path: Путь к входному изображению
        output_dir: Директория для результатов
        compression_ratio: Степень сжатия (0-1)
        algorithm: Алгоритм восстановления ('omp', 'ista', 'sa')
    """
    print("\n" + "=" * 80)
    print(f"Обработка: {image_path.name}")
    print("=" * 80 + "\n")

    # 1. Загрузка
    original = load_image(image_path)
    if original is None:
        return

    print(f"  Размер: {original.shape}")
    print(
        f"  Значения: [{original.min():.0f}, {original.max():.0f}], среднее: {original.mean():.1f}"
    )
    print()

    # 2. Кодирование
    print(f"Кодирование (compression_ratio={compression_ratio})...")
    encoder = IMCSEncoder(compression_ratio=compression_ratio, seed=42)

    t_start = time.time()
    compressed = encoder.encode(original)
    t_encode = time.time() - t_start

    print(f"  ✓ Сжато за {t_encode:.3f} сек")
    print(f"  Размер: {len(compressed)} байт (было {original.nbytes} байт)")
    print(f"  Степень сжатия: {original.nbytes / len(compressed):.2f}x")
    print()

    # 3. Декодирование
    print(f"Декодирование (алгоритм: {algorithm.upper()})...")

    # Мапинг названий алгоритмов
    algorithm_map = {"omp": "omp", "ista": "iterative_threshold", "sa": "simulated_annealing"}

    decoder = IMCSDecoder(
        reconstruction_algorithm=algorithm_map[algorithm]
        # lambda_param и max_iter используют значения по умолчанию из декодера
    )

    t_start = time.time()
    reconstructed = decoder.decode(compressed)
    t_decode = time.time() - t_start

    print(f"  ✓ Восстановлено за {t_decode:.3f} сек")
    print()

    # 4. Метрики
    metrics = calculate_compression_metrics(original, reconstructed)

    print("Качество восстановления:")
    print(f"  MSE:   {metrics['mse']:.2f}")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  MAE:   {metrics['mae']:.2f}")
    print()

    # Интерпретация PSNR
    psnr = metrics["psnr"]
    if psnr > 30:
        quality = "отличное ✓"
    elif psnr > 20:
        quality = "хорошее"
    elif psnr > 15:
        quality = "удовлетворительное"
    else:
        quality = "плохое ⚠️"

    print(f"  Оценка качества: {quality}")
    print()

    # 5. Сохранение результатов
    output_subdir = output_dir / image_path.stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Сохраняем compressed
    compressed_file = output_subdir / "compressed.imcs"
    with open(compressed_file, "wb") as f:
        f.write(compressed)

    # Сохраняем изображения
    save_image(original, output_subdir / "original.png")
    save_image(reconstructed, output_subdir / f"reconstructed_{algorithm}.png")

    # Сохраняем отчёт
    report_file = output_subdir / "report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЁТ О СЖАТИИ IMCS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Файл: {image_path.name}\n")
        f.write(f"Размер: {original.shape}\n")
        f.write(f"Compression ratio: {compression_ratio}\n")
        f.write(f"Алгоритм: {algorithm.upper()}\n\n")
        f.write(f"Время кодирования: {t_encode:.3f} сек\n")
        f.write(f"Время декодирования: {t_decode:.3f} сек\n\n")
        f.write(f"Исходный размер: {original.nbytes} байт\n")
        f.write(f"Сжатый размер: {len(compressed)} байт\n")
        f.write(f"Степень сжатия: {original.nbytes / len(compressed):.2f}x\n\n")
        f.write("=" * 80 + "\n")
        f.write("МЕТРИКИ КАЧЕСТВА\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"MSE:  {metrics['mse']:.2f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"MAE:  {metrics['mae']:.2f}\n\n")
        f.write(f"Качество: {quality}\n")

    print(f"✓ Результаты сохранены в: {output_subdir}/")
    print("  - original.png")
    print(f"  - reconstructed_{algorithm}.png")
    print("  - compressed.imcs")
    print("  - report.txt")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="IMCS - Сжатие изображений с помощью Compressed Sensing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py                                # Обработать все изображения
  python main.py --input test_gradient.png      # Конкретный файл
  python main.py --ratio 0.8 --algorithm ista   # С параметрами

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

    args = parser.parse_args()

    # Директории
    input_dir = Path("examples/input")
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    # Выбираем файлы
    if args.input:
        files = [input_dir / args.input]
        if not files[0].exists():
            print(f"❌ Файл не найден: {files[0]}")
            return
    else:
        files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not files:
        print("⚠️  Не найдено изображений!")
        print(f"Положите PNG или JPG файлы в: {input_dir}/")
        return

    # Обрабатываем
    print("\n" + "=" * 80)
    print("IMCS - Compressed Sensing Image Codec")
    print("=" * 80 + "\n")
    print(f"Найдено изображений: {len(files)}")
    print(f"Compression ratio: {args.ratio}")
    print(f"Алгоритм: {args.algorithm.upper()}")

    for img_file in files:
        try:
            process_image(img_file, output_dir, args.ratio, args.algorithm)
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {img_file.name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✓ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80 + "\n")
    print(f"Результаты: {output_dir}/")


if __name__ == "__main__":
    main()
