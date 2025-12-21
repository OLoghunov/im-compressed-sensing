import time
from pathlib import Path
import numpy as np

from imcs import IMCSEncoder, IMCSDecoder
from imcs.utils import calculate_compression_metrics
from imcs.cli import (
    setup_argument_parser,
    find_input_files,
    load_image,
    save_image,
    load_signal,
    save_signal,
    visualize_signal_comparison,
    create_output_directory,
    save_report,
    interpret_psnr,
)
from visualization.plot_convergence import create_convergence_plot_from_decoder


def process_signal(
    signal_path: Path,
    output_dir: Path,
    compression_ratio: float,
    algorithm: str,
):
    """Обрабатывает 1D сигнал: кодирует и декодирует."""
    print("\n" + "=" * 80)
    print(f"Обработка 1D сигнала: {signal_path.name}")
    print("=" * 80 + "\n")

    original = load_signal(signal_path)
    if original is None:
        return

    print(f"  Размер: {len(original)} отсчётов")
    print(
        f"  Значения: [{original.min():.2f}, {original.max():.2f}], "
        f"среднее: {original.mean():.2f}"
    )
    print()

    print(f"Кодирование (compression_ratio={compression_ratio})...")
    t_encode, compressed = encode_signal(original, compression_ratio)
    print(f"  ✓ Сжато за {t_encode:.3f} сек")
    print(f"  Размер: {len(compressed)} байт (было {original.nbytes} байт)")
    print(f"  Степень сжатия: {original.nbytes / len(compressed):.2f}x")
    print()

    print(f"Декодирование (алгоритм: {algorithm.upper()})...")
    t_decode, reconstructed, _ = decode_signal(compressed, algorithm)
    print(f"  ✓ Восстановлено за {t_decode:.3f} сек")
    print()

    metrics = calculate_compression_metrics(original, reconstructed)
    quality = interpret_psnr(metrics["psnr"])

    print("Качество восстановления:")
    print(f"  MSE:   {metrics['mse']:.2f}")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  MAE:   {metrics['mae']:.2f}")
    print(f"  Оценка качества: {quality}")
    print()

    output_subdir = create_output_directory(output_dir, signal_path.stem)

    with open(output_subdir / "compressed.imcs", "wb") as f:
        f.write(compressed)

    save_signal(original, output_subdir / "original.npy")
    save_signal(reconstructed, output_subdir / f"reconstructed_{algorithm}.npy")

    visualize_signal_comparison(
        original, reconstructed, output_subdir / f"comparison_{algorithm}.png"
    )

    save_report(
        output_subdir / "report.txt",
        signal_path,
        (len(original),),
        compression_ratio,
        algorithm,
        t_encode,
        t_decode,
        original.nbytes,
        len(compressed),
        metrics,
        quality,
    )

    print(f"✓ Результаты сохранены в: {output_subdir}/")
    print("  - original.npy")
    print(f"  - reconstructed_{algorithm}.npy")
    print("  - compressed.imcs")
    print("  - report.txt")
    print(f"  - comparison_{algorithm}.png")


def process_image(
    image_path: Path,
    output_dir: Path,
    compression_ratio: float,
    algorithm: str,
    visualize: bool = False,
):
    """Обрабатывает одно изображение: кодирует и декодирует."""
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
    t_encode, compressed = encode_image(original, compression_ratio)
    print(f"  ✓ Сжато за {t_encode:.3f} сек")
    print(f"  Размер: {len(compressed)} байт (было {original.nbytes} байт)")
    print(f"  Степень сжатия: {original.nbytes / len(compressed):.2f}x")
    print()

    print(f"Декодирование (алгоритм: {algorithm.upper()})...")
    t_decode, reconstructed, decoder = decode_image(compressed, algorithm, visualize)
    print(f"  ✓ Восстановлено за {t_decode:.3f} сек")
    print()

    metrics = calculate_compression_metrics(original, reconstructed)
    quality = interpret_psnr(metrics["psnr"])

    print("Качество восстановления:")
    print(f"  MSE:   {metrics['mse']:.2f}")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  MAE:   {metrics['mae']:.2f}")
    print(f"  Оценка качества: {quality}")
    print()

    output_subdir = create_output_directory(output_dir, image_path.stem)

    save_results(
        output_subdir,
        image_path,
        original,
        reconstructed,
        compressed,
        algorithm,
        compression_ratio,
        t_encode,
        t_decode,
        metrics,
        quality,
    )

    if visualize and decoder.last_history is not None:
        print("Создание визуализации сходимости...")
        create_convergence_plot_from_decoder(
            decoder, compressed, original, algorithm, output_subdir
        )
        print("  ✓ Визуализация сохранена")
        print()

    print(f"✓ Результаты сохранены в: {output_subdir}/")
    print("  - original.png")
    print(f"  - reconstructed_{algorithm}.png")
    print("  - compressed.imcs")
    print("  - report.txt")
    if visualize:
        print(f"  - convergence_{algorithm}.png")


def encode_signal(original: np.ndarray, compression_ratio: float) -> tuple:
    encoder = IMCSEncoder(compression_ratio=compression_ratio, seed=42)
    t_start = time.time()
    compressed = encoder.encode(original)
    t_encode = time.time() - t_start
    return t_encode, compressed


def decode_signal(compressed: bytes, algorithm: str) -> tuple:
    algorithm_map = {
        "omp": "omp",
        "ista": "iterative_threshold",
        "sa": "simulated_annealing",
    }
    decoder = IMCSDecoder(reconstruction_algorithm=algorithm_map[algorithm])
    t_start = time.time()
    reconstructed = decoder.decode(compressed, return_history=False)
    t_decode = time.time() - t_start
    return t_decode, reconstructed, decoder


def encode_image(original: np.ndarray, compression_ratio: float) -> tuple:
    encoder = IMCSEncoder(compression_ratio=compression_ratio, seed=42)
    t_start = time.time()
    compressed = encoder.encode(original)
    t_encode = time.time() - t_start
    return t_encode, compressed


def decode_image(compressed: bytes, algorithm: str, visualize: bool) -> tuple:
    algorithm_map = {
        "omp": "omp",
        "ista": "iterative_threshold",
        "sa": "simulated_annealing",
    }
    decoder = IMCSDecoder(reconstruction_algorithm=algorithm_map[algorithm])
    t_start = time.time()
    reconstructed = decoder.decode(compressed, return_history=visualize)
    t_decode = time.time() - t_start
    return t_decode, reconstructed, decoder


def save_results(
    output_subdir: Path,
    image_path: Path,
    original: np.ndarray,
    reconstructed: np.ndarray,
    compressed: bytes,
    algorithm: str,
    compression_ratio: float,
    t_encode: float,
    t_decode: float,
    metrics: dict,
    quality: str,
):
    with open(output_subdir / "compressed.imcs", "wb") as f:
        f.write(compressed)

    save_image(original, output_subdir / "original.png")
    save_image(reconstructed, output_subdir / f"reconstructed_{algorithm}.png")

    save_report(
        output_subdir / "report.txt",
        image_path,
        original.shape,
        compression_ratio,
        algorithm,
        t_encode,
        t_decode,
        original.nbytes,
        len(compressed),
        metrics,
        quality,
    )


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    input_dir = Path("examples/input")
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    files = find_input_files(args.input, input_dir)

    if not files:
        print("⚠️  Не найдено файлов!")
        print(f"Положите изображения (PNG/JPG) или сигналы (.npy/.txt) в: {input_dir}/")
        return

    print("\n" + "=" * 80)
    print("IMCS - Compressed Sensing Codec")
    print("=" * 80 + "\n")
    print(f"Найдено файлов: {len(files)}")
    print(f"Compression ratio: {args.ratio}")
    print(f"Алгоритм: {args.algorithm.upper()}")

    for file_path in files:
        try:
            suffix = file_path.suffix.lower()

            if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                process_image(file_path, output_dir, args.ratio, args.algorithm, args.visualize)
            elif suffix in [".npy", ".txt"]:
                process_signal(file_path, output_dir, args.ratio, args.algorithm)
            else:
                print(f"\n⚠️  Пропуск {file_path.name}: неподдерживаемый формат")
                continue

        except Exception as e:
            print(f"\n❌ Ошибка при обработке {file_path.name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✓ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80 + "\n")
    print(f"Результаты: {output_dir}/")


if __name__ == "__main__":
    main()
