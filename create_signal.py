import numpy as np
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ГЕНЕРАТОР 1D СИГНАЛОВ")
    print("=" * 60 + "\n")

    while True:
        name = input("Имя файла (без .npy): ").strip()
        if name:
            break
        print("  ⚠️  Имя не может быть пустым!")

    output_path = Path("examples/input") / f"{name}.npy"
    if output_path.exists():
        confirm = input(f"  ⚠️  Файл {name}.npy уже существует. Перезаписать? (y/n): ")
        if confirm.lower() != "y":
            print("Отменено.")
            return

    print("\nСпособ создания:")
    print("  1. Ввести числа вручную")
    print("  2. Начать с нулей (потом указать ненулевые)")
    print("  3. Случайный шум")
    print("  4. Синусоида")
    print("  5. Импульсы")

    while True:
        choice = input("\nВыбор (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            break
        print("  ⚠️  Введите число от 1 до 5")

    if choice == "1":
        print("\nВведите числа через пробел:")
        print("  Пример: 1.5 2.3 -4.1 0 5.6")
        numbers_str = input("Числа: ").strip()

        try:
            signal = np.array([float(x) for x in numbers_str.split()])
            if len(signal) == 0:
                print("  ⚠️  Не введено ни одного числа!")
                return
        except ValueError:
            print("  ⚠️  Ошибка: не все значения являются числами!")
            return

    elif choice == "2":
        while True:
            try:
                length = int(input("\nДлина сигнала: "))
                if length > 0:
                    break
                print("  ⚠️  Длина должна быть > 0")
            except ValueError:
                print("  ⚠️  Введите целое число")

        signal = np.zeros(length)
        print(f"\nСоздан сигнал из {length} нулей.")
        print("Теперь укажите ненулевые элементы.")
        print("  Формат: индекс значение")
        print("  Пример: 10 100.5")
        print("  Для завершения введите пустую строку")

        while True:
            entry = input("Индекс и значение: ").strip()
            if not entry:
                break

            try:
                idx, val = entry.split()
                idx = int(idx)
                val = float(val)

                if 0 <= idx < length:
                    signal[idx] = val
                    print(f"  ✓ signal[{idx}] = {val}")
                else:
                    print(f"  ⚠️  Индекс {idx} вне диапазона [0, {length-1}]")
            except ValueError:
                print("  ⚠️  Неверный формат! Введите: индекс значение")

    elif choice == "3":
        while True:
            try:
                length = int(input("\nДлина сигнала: "))
                if length > 0:
                    break
                print("  ⚠️  Длина должна быть > 0")
            except ValueError:
                print("  ⚠️  Введите целое число")

        amplitude = float(input("Амплитуда шума (например, 50): ") or "50")
        signal = np.random.randn(length) * amplitude

    elif choice == "4":
        while True:
            try:
                length = int(input("\nДлина сигнала: "))
                if length > 0:
                    break
                print("  ⚠️  Длина должна быть > 0")
            except ValueError:
                print("  ⚠️  Введите целое число")

        freq = float(input("Частота (например, 3): ") or "3")
        amplitude = float(input("Амплитуда (например, 100): ") or "100")

        t = np.linspace(0, 4 * np.pi, length)
        signal = amplitude * np.sin(freq * t)

    elif choice == "5":
        while True:
            try:
                length = int(input("\nДлина сигнала: "))
                if length > 0:
                    break
                print("  ⚠️  Длина должна быть > 0")
            except ValueError:
                print("  ⚠️  Введите целое число")

        signal = np.zeros(length)

        while True:
            try:
                n_impulses = int(input("Количество импульсов: "))
                if 0 < n_impulses <= length:
                    break
                print(f"  ⚠️  Количество должно быть от 1 до {length}")
            except ValueError:
                print("  ⚠️  Введите целое число")

        positions = np.random.choice(length, n_impulses, replace=False)
        amplitudes = np.random.randn(n_impulses) * 100
        signal[positions] = amplitudes

        print(f"\n  Импульсы размещены на позициях: {sorted(positions)}")

    np.save(output_path, signal)

    print("\n" + "=" * 60)
    print("✓ СИГНАЛ СОЗДАН")
    print("=" * 60)
    print(f"\nФайл:     {output_path}")
    print(f"Длина:    {len(signal)} отсчётов")
    print(f"Мин:      {signal.min():.2f}")
    print(f"Макс:     {signal.max():.2f}")
    print(f"Среднее:  {signal.mean():.2f}")
    print(f"Ненулевых: {np.count_nonzero(signal)}")

    print("\nПервые 10 значений:")
    preview = signal[:10]
    for i, val in enumerate(preview):
        print(f"  [{i}] = {val:.2f}")
    if len(signal) > 10:
        print(f"  ... (ещё {len(signal) - 10} значений)")

    print("\n" + "-" * 60)
    print("Для обработки запустите:")
    print(f"  python main.py --input {name} --algorithm ista")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nОтменено пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
