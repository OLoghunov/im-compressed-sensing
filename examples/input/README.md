# Тестовые изображения и сигналы

## 2D Изображения (16×16 и 32×32)

### Простые 16×16:
- `01_constant.png` — константа (разреженность: отличная, PSNR: ~∞)
- `02_gradient_h.png` — горизонтальный градиент (разреженность: хорошая, PSNR: ~300 dB)
- `03_gradient_v.png` — вертикальный градиент
- `04_square.png` — белый квадрат в центре
- `05_circle.png` — круг
- `06_stripes_v.png` — вертикальные полосы
- `07_stripes_h.png` — горизонтальные полосы
- `08_checkerboard.png` — шахматная доска 4×4
- `09_diagonal.png` — диагональные линии
- `10_cross.png` — крест
- `11_cloud_small.png` — облачный паттерн

### Сложные 32×32:
- `12_gradient_32x32.png` — радиальный градиент (PSNR: ~45 dB)
- `13_circle_32x32.png` — круг с размытием
- `14_checkerboard_8x8_32x32.png` — шахматная доска 8×8 (PSNR: ~83 dB)
- `15_stripes_32x32.png` — вертикальные полосы
- `16_diagonal_stripes_32x32.png` — диагональные полосы
- `17_mixed_pattern_32x32.png` — смешанный паттерн
- `18_texture_random_32x32.png` — случайная текстура (PSNR: ~14 dB ⚠️)
- `19_concentric_circles_32x32.png` — концентрические круги

### Крупнее (для нагрузочных тестов декодера):
- `20_radial_gradient_128x128.png` — радиальный синус + горизонтальный градиент, **128×128** (удобно сравнивать полный режим и `--block-size 8`)

### Большие цветные изображения:
- `21_color_spectrum_512x512.png` — плавный цветовой градиент, **512×512**
- `22_color_landscape_1280x720.png` — синтетический ландшафт, **1280×720**
- `23_color_topography_1920x1080.png` — цветная топографическая карта, **1920×1080**

## 1D Сигналы

### Разреженные:
- **`test_sparse_signal.npy`** (128 отсчётов)
  - 5 ненулевых значений на позициях [10, 30, 50, 70, 90]
  - Амплитуды: [100, -80, 120, -60, 95]
  - **Идеален для CS!** PSNR: ~21 dB (ISTA)

- **`signal_impulse.npy`** (200 отсчётов)
  - 5 импульсов: [150, 120, 100, 130, 110]
  - Хорошо сжимается, PSNR: ~20 dB

### Регулярные:
- **`signal_sinusoidal.npy`** (256 отсчётов)
  - `100*sin(3t) + 50*cos(7t)`
  - **Отличная разреженность в DCT!** PSNR: ~32 dB ✓

- **`signal_step.npy`** (150 отсчётов)
  - Ступенчатый сигнал (3 уровня)
  - Средняя разреженность, PSNR: ~18 dB

## Визуализация

### Изображения:
- Автоматически сохраняются `original.png` и `reconstructed_*.png`
- Можно открыть прямо в IDE

### Сигналы:
- **`.npy` файлы** — бинарные, не читаются в текстовом редакторе
- **Автоматически создаётся `comparison_*.png`** — график сравнения!
  - График 1: Оригинал vs Восстановлено (наложение)
  - График 2: Оригинал (stem plot)
  - График 3: Ошибка восстановления с MAE

Пример:
```bash
python run.py examples/input/signal_sinusoidal.npy --algorithm ista

# Результат в examples/output/signal_sinusoidal/:
#   - comparison_ista.png — график сравнения
#   - original.npy
#   - reconstructed_ista.npy
#   - report.txt
```

## Использование

### Окно (рекомендуется)
```bash
python run.py
```
Выберите файл в списке (или «Выбрать файл…») и нажмите «Запустить…».

### Консоль — изображения
```bash
python run.py examples/input/18_texture_random_32x32.png --algorithm ista
python run.py examples/input/14_checkerboard_8x8_32x32.png --algorithm omp
python run.py examples/input/22_color_landscape_1280x720.png --algorithm fista --color-mode ycbcr
```

### Консоль — сигналы
```bash
python run.py examples/input/test_sparse_signal.npy --algorithm omp
python run.py examples/input/signal_sinusoidal.npy --algorithm ista
```

## Выводы

**Для изображений:**
- ✅ Простые геометрические паттерны: PSNR > 30 dB
- ⚠️ Случайные текстуры: PSNR < 15 dB (требуется wavelet basis)
- ✅ Цветные синтетические сцены можно использовать для демонстрации режима `rgb`/`ycbcr`

**Для 1D сигналов:**
- ✅ Разреженные импульсы: PSNR > 20 dB
- ✅ Синусоиды (разреженны в DCT): PSNR > 30 dB
- ⚠️ Ступенчатые: PSNR ~18 dB (разрывы создают много частот)
