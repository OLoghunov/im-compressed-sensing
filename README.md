# IMCS - Image Compression using Compressed Sensing

Дипломная работа: Инженерная реализация формата сжатия данных `.imcs` на основе Compressed Sensing.

---

## 🎯 Быстрый старт

> **⚠️ Активируйте виртуальное окружение перед работой:**
> ```bash
> source venv/bin/activate
> ```

```bash
# Обработать изображение
python main.py --input test_gradient.png --ratio 0.8 --algorithm ista

# Результаты → examples/output/
open examples/output/test_gradient/original.png
open examples/output/test_gradient/reconstructed_ista.png
cat examples/output/test_gradient/report.txt
```

---

## 📦 Что реализовано

### Кодер (`imcs/encoder.py`)
- Сжатие 1D сигналов: `y = Φ · x`
- Сжатие 2D изображений: `Y = Φ_row · X · Φ_col^T`
- Формат файла `.imcs`
- Поддержка PNG, JPEG, NumPy массивов

### Декодер (`imcs/decoder.py`)
- Восстановление из сжатых данных
- 3 алгоритма:
  - **OMP** (Orthogonal Matching Pursuit) - быстрый, жадный
  - **ISTA** (Iterative Shrinkage-Thresholding) - точный, медленный
  - **SA** (Simulated Annealing) - гибкий

### Утилиты (`imcs/utils.py`)
- Генерация измерительных матриц (Gaussian, Bernoulli)
- DCT/IDCT преобразования
- Алгоритмы восстановления
- Метрики качества (MSE, PSNR, MAE)

### Тесты (`test_imcs/test_basic.py`)
- 15 unit-тестов
- Проверка корректности базовых функций

---

## 🚀 Использование

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

## 📖 Документация

- **`examples/README.md`** - инструкции по примерам

---

## 👤 Автор

Oleg Y. Logunov  
Дипломная работа, 2025
