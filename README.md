# IMCS - Image Compression using Compressed Sensing

Дипломная работа: инженерная реализация формата сжатия данных `.imcs` на основе Compressed Sensing.

---

## Быстрый старт

> **Активируйте виртуальное окружение:**
> ```bash
> source venv/bin/activate
> ```

### Главная команда

```bash
python run.py
```

Открывается **окно**: список файлов из `examples/input/`, кнопка «Запустить кодирование и декодирование», предпросмотр (Matplotlib). Параметры по умолчанию: `ratio=0.5`, алгоритм `omp`, блоки `8×8` для 2D, цветной режим `ycbcr` — можно поменять в форме.

По умолчанию запускается **новый Qt GUI** с боковой панелью `Настройки стенда`, где можно менять параметры блоков, алгоритма, цвета, `measurement_mode`, `matrix_type` и управлять числом процессов для распараллеливания. Если Qt-зависимость недоступна, `run.py` автоматически откатывается к legacy GUI.

### GUI зависимости

```bash
pip install -r requirements.txt
```

Если нужен только современный GUI-слой:

```bash
pip install PySide6
```

### Консоль без GUI (один файл)

```bash
python run.py examples/input/02_gradient_h.png
```

Опции при необходимости: `--ratio`, `--algorithm`, `--block`, `--basis`, `--matrix`,
`--measurement-mode`, `--color-mode` (см. `python run.py -h`).

### Совместимость

```bash
python main.py
```

Это то же самое, что `python run.py` (точка входа перенаправлена).

### Создание 1D сигналов

```bash
python create_signal.py
```

### Benchmark-исследования

```bash
python benchmark.py --block-sizes 8 16 32 --ratios 0.3 0.5 0.7 --algorithms omp ista
```

Сводка и артефакты сохраняются в `examples/output/benchmarks/block_study/`.

---

## Структура проекта

| Модуль | Назначение |
|--------|------------|
| `run.py` | Точка входа: GUI, один прогон в консоли, цвет и последовательности кадров |
| `benchmark.py` | Воспроизводимые benchmark-запуски для дипломных экспериментов |
| `imcs/pipeline.py` | Логика кодирования/декодирования и значения по умолчанию |
| `imcs/gui_qt.py` | Основной современный Qt GUI с панелью `Настройки стенда` |
| `imcs/gui.py` | Legacy GUI на Tkinter/Matplotlib, используется как fallback |
| `imcs/encoder.py`, `imcs/decoder.py` | Кодек |
| `imcs/utils.py` | Матрицы измерений, OMP/ISTA/FISTA, метрики |
| `imcs/benchmarking.py` | Фиксированный benchmark-набор, summary и графики |
| `imcs/cli.py` | Загрузка/сохранение изображений и сигналов |

---

## Что реализовано

### Кодер (`imcs/encoder.py`)
- 1D: `y = Φ · x`
- 2D: блочный режим (JPEG-подобно) или полный кадр на малых размерах
- Формат `.imcs`

### Декодер (`imcs/decoder.py`)
- Восстановление: **OMP**, **ISTA**, **FISTA**, **SA**

### Исследовательские расширения
- блочная обработка с `shared` и `per_block` стратегиями для `Phi`
- базисы `DCT` и `wavelet` (Haar)
- цветные изображения через `RGB` и `YCbCr`
- baseline для последовательностей кадров
- benchmark summary с `PSNR`, `SSIM`, временем кодирования и декодирования

### Утилиты
- **`imcs/utils.py`**: DCT, алгоритмы восстановления, PSNR и др.

---

## Python API

```python
import numpy as np
from imcs import IMCSEncoder, IMCSDecoder
from imcs.utils import calculate_compression_metrics

x = np.zeros(100)
x[[10, 30, 50]] = [1.0, 2.0, 1.5]

encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
compressed = encoder.encode(x)

decoder = IMCSDecoder(reconstruction_algorithm="iterative_threshold", lambda_param=0.01)
x_reconstructed = decoder.decode(compressed)

metrics = calculate_compression_metrics(x, x_reconstructed)
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

---

## Визуализация сходимости (отдельный сценарий)

Графики сходимости можно строить через `visualization/plot_convergence.py`, если вызывать декодер с `return_history=True` из своего кода. В GUI по умолчанию сходимость не рисуется, чтобы не усложнять окно.

---

## Benchmarks

Фиксированный baseline и схема результата описаны в `BENCHMARKS.md`.

---

## Автор

Oleg Y. Logunov
