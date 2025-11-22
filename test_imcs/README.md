"""Тесты IMCS"""

## Структура

```
test_imcs/
├── conftest.py        # Fixtures
├── test_encoder.py    # Тесты encoder (31)
├── test_decoder.py    # Тесты decoder (25)
└── test_utils.py      # Тесты utils (46)
```

## Запуск

```bash
# Все тесты
pytest

# Конкретный модуль
pytest test_imcs/test_encoder.py
pytest test_imcs/test_decoder.py
pytest test_imcs/test_utils.py

# По названию
pytest -k encoder
pytest -k decoder
pytest -k initialization
```

## Стиль

- Без классов
- Без лишних комментариев
- Параметризация через @pytest.mark.parametrize
- Простые функции test_*

## Статус

✅ 34 активны
⏭️ 68 готовы к активации
