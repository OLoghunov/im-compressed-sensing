"""
Диалог выбора файла с initialdir (проводник / Finder).

1) tkinter.filedialog — если доступен _tkinter.
2) macOS: osascript choose file.
3) Linux: zenity, если есть в PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def pick_file(initial_dir: Path, title: str = "Выберите файл") -> Optional[Path]:
    initial_dir = initial_dir.resolve()
    if not initial_dir.is_dir():
        initial_dir = initial_dir.parent

    path = _pick_tkinter(initial_dir, title)
    if path is not None:
        return path

    if sys.platform == "darwin":
        path = _pick_macos(initial_dir, title)
        if path is not None:
            return path

    if sys.platform.startswith("linux") and shutil.which("zenity"):
        path = _pick_zenity(initial_dir, title)
        if path is not None:
            return path

    return None


def _pick_tkinter(initial_dir: Path, title: str) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        p = filedialog.askopenfilename(
            title=title,
            initialdir=str(initial_dir),
            filetypes=[
                ("Изображения и сигналы", "*.png *.jpg *.jpeg *.bmp *.tiff *.npy *.txt"),
                ("Все файлы", "*.*"),
            ],
        )
        root.destroy()
        return Path(p) if p else None
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        return None


def _pick_macos(initial_dir: Path, title: str) -> Optional[Path]:
    d = str(initial_dir)
    # Экранирование для AppleScript
    d_esc = d.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    try
        set defaultPath to POSIX file "{d_esc}"
        set f to choose file with prompt "{title.replace('"', '\\"')}" default location defaultPath
        return POSIX path of f
    on error
        return ""
    end try
    '''
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        out = (r.stdout or "").strip()
        if not out:
            return None
        p = Path(out)
        return p if p.is_file() else None
    except Exception:
        return None


def _pick_zenity(initial_dir: Path, title: str) -> Optional[Path]:
    try:
        r = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title",
                title,
                "--filename",
                str(initial_dir) + "/",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        out = (r.stdout or "").strip()
        if not out:
            return None
        p = Path(out)
        return p if p.is_file() else None
    except Exception:
        return None
