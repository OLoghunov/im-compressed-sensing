# IMCS - Image Compression using Compressed Sensing

Дипломная работа: Инженерная реализация формата сжатия данных `.imcs` на основе Compressed Sensing.

---

## Быстрый старт

> **Активируйте виртуальное окружение перед работой:**
> ```bash
> source venv/bin/activate
> ```

### Обработка изображений:
```bash
python main.py --input test_gradient.png --ratio 0.8 --algorithm ista
```

### Создание и обработка 1D сигналов:
```bash
# Cигнал можно создать интерактивно, либо загрузить готовый .npy
python create_signal.py
```

---

## Что реализовано

### Кодер (`imcs/encoder.py`)
- Сжатие 1D сигналов: `y = Φ · x`
- Сжатие 2D изображений: `Y = Φ_row · X · Φ_col^T`
- Формат файла `.imcs`
- Поддержка PNG, JPEG, NumPy массивов

### Декодер (`imcs/decoder.py`)
- Восстановление из сжатых данных
- 3 алгоритма:
  - **OMP** (Orthogonal Matching Pursuit)
  - **ISTA** (Iterative Shrinkage-Thresholding)
  - **SA** (Simulated Annealing)

### Утилиты
- **`imcs/utils.py`**: Генерация матриц, DCT/IDCT, алгоритмы восстановления, метрики (MSE, PSNR, MAE)
- **`imcs/cli.py`**: CLI-утилиты (парсинг аргументов, поиск файлов, загрузка/сохранение изображений)

### Тесты (`test_imcs/test_basic.py`)
- 15 unit-тестов
- Проверка корректности базовых функций

---

## Использование

### Простой пример

```python
from imcs import IMCSEncoder, IMCSDecoder
import numpy as np

# Разреженный сигнал
x = np.zeros(100)
x[[10, 30, 50]] = [1.0, 2.0, 1.5]

# Кодирование
encoder = IMCSEncoder(compression_ratio=0.5, seed=42)
compressed = encoder.encode(x)

# Декодирование
decoder = IMCSDecoder(reconstruction_algorithm="ista", lambda_param=0.01)
x_reconstructed = decoder.decode(compressed)

# Метрики
from imcs.utils import calculate_compression_metrics
metrics = calculate_compression_metrics(x, x_reconstructed)
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

### Обработка изображений

```bash
# Положите изображение в examples/input/
cp ~/Pictures/photo.png examples/input/

# Обработайте
python main.py --input photo.png --ratio 0.8 --algorithm ista

# Результаты
open examples/output/photo/original.png
open examples/output/photo/reconstructed_ista.png
```

---

## Документация

- **`examples/README.md`** - инструкции по примерам

### Визуализация сходимости алгоритмов

Графики показывают:
- Скорость сходимости (residual по итерациям)
- Эволюцию sparsity (количество ненулевых коэффициентов)
- 2D ландшафт функции потерь с траекторией алгоритма

Графики сохраняются в `examples/output/{image_name}/convergence_{algorithm}.png`

```bash
# С визуализацией (по умолчанию)
python main.py --input photo.png

# Без визуализации
python main.py --input photo.png --no-visualize
```

---

## Автор

Oleg Y. Logunov  
Дипломная работа, 2025
