import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fftpack import dct

from imcs import IMCSEncoder, IMCSDecoder
from imcs.utils import calculate_compression_metrics, generate_measurement_matrix
from visualization.plot_convergence import plot_convergence_path


def load_image(image_path: Path) -> np.ndarray:
    """Загружает изображение из файла."""
    if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img = Image.open(image_path)

        if img.mode != "L":
            print("  Конвертация RGB → Grayscale")
            img = img.convert("L")

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


def process_image(
    image_path: Path,
    output_dir: Path,
    compression_ratio: float,
    algorithm: str,
    visualize: bool = False,
):
    """
    Обрабатывает одно изображение: кодирует и декодирует.

    Args:
        image_path: Путь к входному изображению
        output_dir: Директория для результатов
        compression_ratio: Степень сжатия (0-1)
        algorithm: Алгоритм восстановления ('omp', 'ista', 'sa')
        visualize: Если True, создает визуализацию сходимости
    """
    print("\n" + "=" * 80)
    print(f"Обработка: {image_path.name}")
    print("=" * 80 + "\n")

    original = load_image(image_path)
    if original is None:
        return

    print(f"  Размер: {original.shape}")
    print(
        f"  Значения: [{original.min():.0f}, {original.max():.0f}], среднее: {original.mean():.1f}"
    )
    print()

    print(f"Кодирование (compression_ratio={compression_ratio})...")
    encoder = IMCSEncoder(compression_ratio=compression_ratio, seed=42)

    t_start = time.time()
    compressed = encoder.encode(original)
    t_encode = time.time() - t_start

    print(f"  ✓ Сжато за {t_encode:.3f} сек")
    print(f"  Размер: {len(compressed)} байт (было {original.nbytes} байт)")
    print(f"  Степень сжатия: {original.nbytes / len(compressed):.2f}x")
    print()

    print(f"Декодирование (алгоритм: {algorithm.upper()})...")

    algorithm_map = {"omp": "omp", "ista": "iterative_threshold", "sa": "simulated_annealing"}

    decoder = IMCSDecoder(reconstruction_algorithm=algorithm_map[algorithm])

    t_start = time.time()
    reconstructed = decoder.decode(compressed, return_history=visualize)
    t_decode = time.time() - t_start

    print(f"  ✓ Восстановлено за {t_decode:.3f} сек")
    print()

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

    output_subdir = output_dir / image_path.stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    compressed_file = output_subdir / "compressed.imcs"
    with open(compressed_file, "wb") as f:
        f.write(compressed)

    save_image(original, output_subdir / "original.png")
    save_image(reconstructed, output_subdir / f"reconstructed_{algorithm}.png")

    if visualize and decoder.last_history is not None:
        print("Создание визуализации сходимости...")

        history = decoder.last_history
        residuals = decoder.last_residuals

        metadata, measurements = decoder._deserialize(compressed)

        if metadata["is_2d"]:
            n_row, n_col = metadata["original_shape"]
            m_total = metadata["m_row"]
            n_total = n_row * n_col

            Phi = generate_measurement_matrix(
                m_total, n_total, metadata["matrix_type"], metadata["seed"]
            )

            def create_dct_matrix(n):
                return dct(np.eye(n), axis=0, norm="ortho")

            Psi_row = create_dct_matrix(n_row)
            Psi_col = create_dct_matrix(n_col)
            Psi_2d = np.kron(Psi_col, Psi_row)

            A = Phi @ Psi_2d.T
            y = measurements

            original_flat = original.flatten()
            Psi = dct(np.eye(n_row * n_col), norm="ortho")
            s_true = Psi @ original_flat

            plot_convergence_path(
                history,
                residuals,
                s_true,
                A,
                y,
                decoder.lambda_param,
                algorithm.upper(),
                output_dir=str(output_subdir),
                filename_prefix=f"convergence_{algorithm}",
            )
            print("  ✓ Визуализация сохранена")

        print()

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
    if visualize:
        print(f"  - convergence_{algorithm}.png")


def main():
    """Главная функция."""
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

    args = parser.parse_args()

    input_dir = Path("examples/input")
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

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

    print("\n" + "=" * 80)
    print("IMCS - Compressed Sensing Image Codec")
    print("=" * 80 + "\n")
    print(f"Найдено изображений: {len(files)}")
    print(f"Compression ratio: {args.ratio}")
    print(f"Алгоритм: {args.algorithm.upper()}")

    for img_file in files:
        try:
            process_image(img_file, output_dir, args.ratio, args.algorithm, args.visualize)
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
