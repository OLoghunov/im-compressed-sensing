import sys
import shutil
from pathlib import Path


def main():
    force = "--force" in sys.argv or "-f" in sys.argv

    output_dir = Path("examples/output")

    if not output_dir.exists():
        print("✓ Директория examples/output/ не существует (уже чиста)")
        return

    # Получаем список поддиректорий
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print("✓ Директория examples/output/ пуста")
        return

    print("\n" + "=" * 60)
    print("ОЧИСТКА OUTPUT")
    print("=" * 60 + "\n")
    print(f"Найдено директорий: {len(subdirs)}\n")

    for subdir in subdirs:
        print(f"  • {subdir.name}")

    if not force:
        print("\n" + "-" * 60)
        confirm = input("Удалить все эти директории? (y/n): ").strip().lower()

        if confirm != "y":
            print("\nОтменено.")
            return

    print("\nУдаление...\n")
    deleted_count = 0
    for subdir in subdirs:
        try:
            shutil.rmtree(subdir)
            print(f"  ✓ Удалено: {subdir.name}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Ошибка при удалении {subdir.name}: {e}")

    print("\n" + "=" * 60)
    print(f"✓ ОЧИЩЕНО: {deleted_count} из {len(subdirs)} директорий")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nОтменено пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
