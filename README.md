# IMCS - Image/data Compression using Compressed Sensing

Дипломная работа: Инженерная реализация формата сжатия данных `.imcs` на основе алгоритма compressed sensing.

## Описание проекта

IMCS (Image/data Compression using Compressed Sensing) - это формат сжатия данных, основанный на теории сжатого восприятия (compressed sensing). Проект включает в себя реализацию кодера и декодера для сжатия и восстановления данных.

### Основные компоненты

- **Encoder (Кодер)**: Сжимает данные используя алгоритмы compressed sensing
- **Decoder (Декодер)**: Восстанавливает данные из сжатого формата
- **Utils**: Вспомогательные функции для работы с измерительными матрицами и метриками

## Установка

### 1. Создание виртуального окружения

```bash
python3 -m venv venv
```

### 2. Активация виртуального окружения

На macOS/Linux:
```bash
source venv/bin/activate
```

На Windows:
```bash
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
pip install -e .
```

Или используя Makefile:
```bash
make install
```

## Структура проекта

```
imcs/
├── imcs/                  # Основной пакет
│   ├── __init__.py       # Инициализация пакета
│   ├── encoder.py        # Реализация кодера
│   ├── decoder.py        # Реализация декодера
│   └── utils.py          # Вспомогательные функции
├── test_imcs/             # Тесты (102 теста)
│   ├── conftest.py       # Fixtures
│   ├── test_encoder.py   # Тесты encoder (31 тест)
│   ├── test_decoder.py   # Тесты decoder (25 тестов)
│   └── test_utils.py     # Тесты utils (46 тестов)
├── venv/                  # Виртуальное окружение
├── main.py               # Точка входа в приложение
├── requirements.txt      # Зависимости проекта
├── setup.py              # Конфигурация пакета
├── pytest.ini            # Конфигурация pytest
├── Makefile              # Команды для разработки
└── README.md             # Этот файл
```

## Использование

### Запуск основного приложения

```bash
python main.py
```

### Пример использования кодера

```python
from imcs import IMCSEncoder
import numpy as np

# Создание кодера
encoder = IMCSEncoder(compression_ratio=0.5, sparsity_basis='dct')

# Сжатие данных
data = np.random.rand(100)
compressed = encoder.encode(data)
```

### Пример использования декодера

```python
from imcs import IMCSDecoder

# Создание декодера
decoder = IMCSDecoder(reconstruction_algorithm='omp')

# Восстановление данных
reconstructed = decoder.decode(compressed)
```

## Тестирование

Проект использует pytest для тестирования. Все тесты находятся в директории `test_imcs/`.

### Особенности тестов

Тесты написаны с использованием `@pytest.mark.parametrize` для эффективного тестирования различных входных данных и сценариев.

**Пример параметризованного теста:**

```python
@pytest.mark.parametrize("compression_ratio,sparsity_basis", [
    (0.5, 'dct'),
    (0.3, 'wavelet'),
    (0.7, 'dct'),
])
def test_encoder_initialization(compression_ratio, sparsity_basis):
    """Test that encoder initializes with valid parameters."""
    encoder = IMCSEncoder(compression_ratio=compression_ratio, sparsity_basis=sparsity_basis)
    assert encoder.compression_ratio == compression_ratio
    assert encoder.sparsity_basis == sparsity_basis
```

Такой подход позволяет автоматически запускать один и тот же тест с разными параметрами.

**Подробные примеры команд см. в файле `TEST_COMMANDS.md`**

### Запуск всех тестов

```bash
pytest
```

Или:
```bash
make test
```

### Запуск тестов с фильтрацией по названию/модулю

Используя pytest напрямую:
```bash
# Запуск тестов, содержащих "encoder" в названии
pytest -k encoder

# Запуск тестов из конкретного модуля
pytest test_imcs/test_encoder.py             # Все тесты encoder
pytest test_imcs/test_decoder.py             # Все тесты decoder
pytest test_imcs/test_utils.py               # Все тесты utils

# Запуск конкретного теста
pytest test_imcs/test_encoder.py::test_encoder_initialization

# Запуск конкретного параметризованного теста
pytest test_imcs/test_encoder.py::test_encoder_initialization[0.5-dct]
```

Используя Makefile:
```bash
# Запуск тестов с фильтром
make test-filter FILTER=encoder

# Запуск тестов декодера
make test-filter FILTER=decoder

# Запуск тестов утилит
make test-filter FILTER=utils

# Запуск тестов по части названия
make test-filter FILTER=initialization
```

### Запуск тестов с покрытием кода

```bash
pytest --cov=imcs --cov-report=html --cov-report=term
```

После выполнения откройте `htmlcov/index.html` в браузере для просмотра детального отчета.

### Запуск тестов с подробным выводом

```bash
pytest -v
```

### Запуск только не пропущенных тестов

```bash
pytest -v --ignore-glob="*skip*"
```

### Маркеры тестов

В проекте используются следующие маркеры для категоризации тестов:

- `@pytest.mark.slow` - медленные тесты
- `@pytest.mark.integration` - интеграционные тесты
- `@pytest.mark.unit` - unit-тесты
- `@pytest.mark.encoder` - тесты кодера
- `@pytest.mark.decoder` - тесты декодера
- `@pytest.mark.utils` - тесты утилит

Запуск тестов по маркерам:
```bash
# Запустить только unit-тесты
pytest -m unit

# Запустить все тесты кроме медленных
pytest -m "not slow"

# Запустить только тесты кодера
pytest -m encoder
```

## Разработка

### Форматирование кода

```bash
# Форматирование с помощью black
make format

# Или напрямую
black imcs test_imcs
```

### Проверка кода (linting)

```bash
# Запуск flake8
make lint

# Или напрямую
flake8 imcs test_imcs
```

### Очистка временных файлов

```bash
make clean
```

### Деактивация виртуального окружения

```bash
deactivate
```

## Зависимости

### Основные
- `numpy` - работа с массивами и математические операции
- `scipy` - научные вычисления и оптимизация

### Тестирование
- `pytest` - фреймворк для тестирования
- `pytest-cov` - покрытие кода тестами

### Разработка
- `black` - форматирование кода
- `flake8` - проверка стиля кода
- `mypy` - статическая проверка типов

## Roadmap

- [ ] Реализация базового кодера
- [ ] Реализация базового декодера
- [ ] Реализация генерации измерительных матриц
- [ ] Реализация алгоритмов восстановления (OMP, Basis Pursuit)
- [ ] Определение формата файла .imcs
- [ ] Реализация работы с файлами
- [ ] Добавление метрик качества сжатия
- [ ] Оптимизация производительности
- [ ] Документация API
- [ ] Примеры использования

## Автор

Дипломная работа

## Лицензия

MIT License (или укажите вашу лицензию)
