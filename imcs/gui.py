from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Optional

import numpy as np
from PIL import Image
from imcs.file_dialog import pick_file
from imcs.pipeline import (
    DEFAULT_ALGORITHM,
    DEFAULT_BASIS,
    DEFAULT_BLOCK_EDGE,
    DEFAULT_COLOR_MODE,
    DEFAULT_RATIO,
    run_color_image,
    run_image,
    run_signal,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "examples" / "input"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"

_COLOR_BG = "#e8eaed"
_COLOR_PANEL = "#ffffff"
_COLOR_ACCENT = "#1a73e8"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
_SIGNAL_SUFFIXES = {".npy", ".txt"}


def _textbox_str_mpl(tb) -> str:
    return tb.text_disp.get_text().strip()


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_SUFFIXES


def _is_signal_path(path: Path) -> bool:
    return path.suffix.lower() in _SIGNAL_SUFFIXES


def _draw_preview(ax1, ax2, canvas, path: Path) -> None:
    ax1.clear()
    ax2.clear()

    if _is_image_path(path):
        img = Image.open(path)
        if img.mode in {"RGB", "RGBA"}:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            ax1.imshow(arr)
        else:
            img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)
            ax1.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax1.set_title("Исходное", fontsize=11, pad=8)
        ax1.axis("off")
        ax2.set_title("После IMCS", fontsize=11, pad=8)
        ax2.text(
            0.5,
            0.5,
            "Нажмите «Запустить IMCS»",
            ha="center",
            va="center",
            fontsize=10,
            color="#5f6368",
            transform=ax2.transAxes,
        )
        ax2.axis("off")
    elif _is_signal_path(path):
        if path.suffix.lower() == ".npy":
            signal = np.load(path)
        else:
            signal = np.loadtxt(path)
        signal = np.asarray(signal, dtype=np.float64).ravel()
        ax1.plot(signal, color=_COLOR_ACCENT, lw=1.2, alpha=0.9)
        ax1.set_title("Исходный сигнал", fontsize=11, pad=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        ax2.set_title("После IMCS", fontsize=11, pad=8)
        ax2.text(
            0.5,
            0.5,
            "Нажмите «Запустить IMCS»",
            ha="center",
            va="center",
            fontsize=10,
            color="#5f6368",
            transform=ax2.transAxes,
        )
        ax2.grid(False)
        ax2.tick_params(labelsize=8)
    else:
        ax1.set_title("Исходное", fontsize=11, pad=8)
        ax2.set_title("После IMCS", fontsize=11, pad=8)
        ax1.text(
            0.5,
            0.5,
            "Неподдерживаемый формат",
            ha="center",
            va="center",
            fontsize=10,
            color="#5f6368",
            transform=ax1.transAxes,
        )
        ax1.axis("off")
        ax2.axis("off")

    canvas.draw_idle()


def _run_processing_task(task: dict, result_queue) -> None:
    try:
        task_type = task["task_type"]
        kwargs = task["kwargs"]
        if task_type == "image":
            result = run_image(**kwargs)
        elif task_type == "color_image":
            result = run_color_image(**kwargs)
        elif task_type == "signal":
            result = run_signal(**kwargs)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        result_queue.put({"status": "ok", "task_type": task_type, "result": result})
    except Exception as exc:
        result_queue.put({"status": "error", "message": str(exc)})


def run_app() -> None:
    try:
        import tkinter as tk  # noqa: F401
    except ImportError:
        _run_app_matplotlib()
        return
    _run_app_tk()


def _run_app_tk() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    import matplotlib as mpl

    mpl.rcParams["toolbar"] = "None"

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected: list[Optional[Path]] = [None]
    worker_process: list[Optional[mp.Process]] = [None]
    worker_queue: list[Optional[mp.Queue]] = [None]
    stop_requested = {"value": False}

    root = tk.Tk()
    root.title("IMCS — кодирование и декодирование")
    root.configure(bg=_COLOR_BG)
    root.minsize(960, 720)
    root.geometry("1200x880")

    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    path_var = tk.StringVar(value="")
    color_var = tk.StringVar(value=DEFAULT_COLOR_MODE)
    basis_var = tk.StringVar(value=DEFAULT_BASIS)

    left = tk.Frame(root, bg=_COLOR_BG, padx=10, pady=8)
    left.grid(row=0, column=0, sticky="nw")

    hint_var = tk.StringVar(
        value=(
            "Файл не выбран.\n"
            "Кнопка — проводник в\nexamples/input."
        )
    )
    hint_lbl = tk.Label(
        left,
        textvariable=hint_var,
        bg=_COLOR_BG,
        fg="#5f6368",
        font=("Helvetica", 8),
        justify=tk.CENTER,
        wraplength=170,
    )

    def on_pick() -> None:
        chosen = filedialog.askopenfilename(
            parent=root,
            title="Выберите изображение или сигнал",
            initialdir=str(INPUT_DIR),
            filetypes=[
                ("Изображения и сигналы", "*.png *.jpg *.jpeg *.bmp *.tiff *.npy *.txt"),
                ("Все файлы", "*.*"),
            ],
        )
        if not chosen:
            set_status("Выбор файла отменён.")
            return
        p = Path(chosen)
        selected[0] = p
        path_var.set(str(p.resolve()))
        short = p.name
        if len(short) > 28:
            short = short[:25] + "…"
        hint_var.set(f"Выбрано:\n{short}")
        hint_lbl.configure(fg="#202124")
        _draw_preview(ax1, ax2, canvas, p)
        set_status(f"Предпросмотр: {p.name}")

    btn_pick = tk.Button(
        left,
        text="Выбрать файл…",
        command=on_pick,
        bg="#dadce0",
        activebackground=_COLOR_ACCENT,
        font=("Helvetica", 10),
        padx=8,
        pady=6,
    )
    btn_pick.pack(fill=tk.X)
    hint_lbl.pack(pady=(10, 0))

    fig = Figure(figsize=(9, 5.2), facecolor=_COLOR_BG, dpi=100)
    ax1 = fig.add_subplot(121, facecolor=_COLOR_PANEL)
    ax2 = fig.add_subplot(122, facecolor=_COLOR_PANEL)
    ax1.set_title("Исходное", fontsize=11, pad=8)
    ax2.set_title("После IMCS", fontsize=11, pad=8)
    ax1.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)

    plot_frame = tk.Frame(root, bg=_COLOR_BG)
    plot_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=8)
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.08)
    fig.suptitle("IMCS", fontsize=14, fontweight="bold", color="#202124", y=0.98)
    fig.text(
        0.5,
        0.93,
        f"Папка по умолчанию: {INPUT_DIR}",
        ha="center",
        fontsize=8,
        color="#5f6368",
    )

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    bottom = tk.Frame(root, bg=_COLOR_BG, padx=12, pady=8)
    bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
    bottom.grid_columnconfigure(1, weight=1)

    status_var = tk.StringVar(value="")
    status_lbl = tk.Label(
        bottom,
        textvariable=status_var,
        bg=_COLOR_BG,
        fg="#202124",
        font=("Helvetica", 9),
        justify=tk.LEFT,
        anchor="w",
        wraplength=1100,
    )
    status_lbl.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

    tk.Label(bottom, text="Путь к файлу:", bg=_COLOR_BG, font=("Helvetica", 9)).grid(
        row=1, column=0, sticky="w"
    )
    path_entry = tk.Entry(bottom, textvariable=path_var, font=("Helvetica", 9))
    path_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0))

    algo_frame = tk.Frame(bottom, bg=_COLOR_BG)
    algo_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 4))
    tk.Label(algo_frame, text="Алгоритм:", bg=_COLOR_BG, font=("Helvetica", 9)).pack(side=tk.LEFT)
    algo_var = tk.StringVar(value=DEFAULT_ALGORITHM)
    for name in ("omp", "ista", "fista", "sa"):
        ttk.Radiobutton(algo_frame, text=name, variable=algo_var, value=name).pack(
            side=tk.LEFT, padx=(12, 0)
        )

    params = tk.Frame(bottom, bg=_COLOR_BG)
    params.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 4))
    tk.Label(params, text="ratio:", bg=_COLOR_BG, font=("Helvetica", 9)).pack(side=tk.LEFT)
    ratio_var = tk.StringVar(value=str(DEFAULT_RATIO))
    tk.Entry(params, textvariable=ratio_var, width=8, font=("Helvetica", 9)).pack(
        side=tk.LEFT, padx=(6, 20)
    )
    tk.Label(params, text="блок:", bg=_COLOR_BG, font=("Helvetica", 9)).pack(side=tk.LEFT)
    block_var = tk.StringVar(value=str(DEFAULT_BLOCK_EDGE))
    tk.Entry(params, textvariable=block_var, width=8, font=("Helvetica", 9)).pack(
        side=tk.LEFT, padx=(6, 0)
    )

    options = tk.Frame(bottom, bg=_COLOR_BG)
    options.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 4))
    tk.Label(options, text="цвет:", bg=_COLOR_BG, font=("Helvetica", 9)).pack(side=tk.LEFT)
    ttk.Combobox(
        options,
        textvariable=color_var,
        values=("gray", "rgb", "ycbcr"),
        state="readonly",
        width=8,
    ).pack(side=tk.LEFT, padx=(6, 18))
    tk.Label(options, text="basis:", bg=_COLOR_BG, font=("Helvetica", 9)).pack(side=tk.LEFT)
    ttk.Combobox(
        options,
        textvariable=basis_var,
        values=("dct", "wavelet"),
        state="readonly",
        width=8,
    ).pack(side=tk.LEFT, padx=(6, 0))

    btn_run = tk.Button(
        bottom,
        text="Запустить IMCS",
        bg="#dadce0",
        activebackground=_COLOR_ACCENT,
        font=("Helvetica", 10, "bold"),
        padx=16,
        pady=10,
    )
    btn_run.grid(row=5, column=2, sticky="e", pady=(8, 0))
    btn_stop = tk.Button(
        bottom,
        text="Остановить",
        bg="#f4c7c3",
        activebackground="#d93025",
        font=("Helvetica", 10, "bold"),
        padx=16,
        pady=10,
        state=tk.DISABLED,
    )
    btn_stop.grid(row=5, column=1, sticky="e", padx=(0, 8), pady=(8, 0))

    def set_status(msg: str) -> None:
        status_var.set(msg)

    def _set_running_state(is_running: bool) -> None:
        btn_run.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        btn_stop.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        btn_pick.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        path_entry.configure(state="disabled" if is_running else "normal")

    def stop_processing() -> None:
        proc = worker_process[0]
        if proc is not None and proc.is_alive():
            stop_requested["value"] = True
            proc.terminate()
            proc.join(timeout=0.5)
        worker_process[0] = None
        worker_queue[0] = None
        _set_running_state(False)
        set_status("Обработка остановлена.")

    def handle_result(payload: dict) -> None:
        if payload["status"] == "error":
            set_status(f"Ошибка: {payload['message']}")
            return

        res = payload.get("result")
        if res is None:
            set_status("Обработка завершилась без результата.")
            return

        ax1.clear()
        ax2.clear()
        if getattr(res.original, "ndim", 2) == 3:
            ax1.imshow(np.clip(res.original, 0, 255).astype(np.uint8))
            ax2.imshow(np.clip(res.reconstructed, 0, 255).astype(np.uint8))
        elif payload["task_type"] == "signal":
            o = np.asarray(res.original, dtype=np.float64).ravel()
            r = np.asarray(res.reconstructed, dtype=np.float64).ravel()
            ax1.plot(o, color=_COLOR_ACCENT, lw=1.2, alpha=0.9)
            ax2.plot(r, color="#d93025", lw=1.2, alpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=8)
            ax2.tick_params(labelsize=8)
            ax1.set_title("Исходный сигнал", fontsize=11, pad=8)
            ax2.set_title("После IMCS", fontsize=11, pad=8)
        else:
            ax1.imshow(res.original, cmap="gray", vmin=0, vmax=255)
            ax2.imshow(res.reconstructed, cmap="gray", vmin=0, vmax=255)

        if payload["task_type"] != "signal":
            ax1.set_title("Исходное", fontsize=11, pad=8)
            ax2.set_title("После IMCS", fontsize=11, pad=8)
            ax1.axis("off")
            ax2.axis("off")

        set_status(
            f"PSNR: {res.metrics['psnr']:.2f} dB | "
            f"SSIM: {res.metrics['ssim']:.4f} | "
            f"код {res.t_encode:.3f} с | декод {res.t_decode:.3f} с\n{res.output_subdir}"
        )
        canvas.draw_idle()

    def poll_worker() -> None:
        proc = worker_process[0]
        result_queue = worker_queue[0]
        if proc is None or result_queue is None:
            return

        try:
            payload = result_queue.get_nowait()
        except Empty:
            if proc.is_alive():
                root.after(120, poll_worker)
                return
            if stop_requested["value"]:
                stop_requested["value"] = False
            else:
                set_status("Процесс завершился без ответа.")
            worker_process[0] = None
            worker_queue[0] = None
            _set_running_state(False)
            return

        proc.join(timeout=0.2)
        worker_process[0] = None
        worker_queue[0] = None
        _set_running_state(False)
        stop_requested["value"] = False
        handle_result(payload)

    def _parse_ratio() -> float:
        r = float(ratio_var.get().strip().replace(",", "."))
        if not 0 < r < 1:
            raise ValueError("ratio ∈ (0, 1)")
        return r

    def _parse_block() -> int:
        return int(block_var.get().strip())

    def _resolve_path() -> Optional[Path]:
        raw = path_var.get().strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                selected[0] = p
                return p
            set_status(f"Файл не найден: {raw}")
            return None
        if selected[0] is not None and selected[0].is_file():
            return selected[0]
        set_status("Выберите файл кнопкой «Выбрать файл…» или вставьте полный путь.")
        return None

    def on_run() -> None:
        if worker_process[0] is not None and worker_process[0].is_alive():
            set_status(
                "IMCS уже обрабатывает файл. "
                "Можно дождаться завершения или нажать «Остановить»."
            )
            return
        path = _resolve_path()
        if path is None:
            return
        algo = algo_var.get()
        try:
            ratio = _parse_ratio()
            block_edge = _parse_block()
        except ValueError as e:
            set_status(str(e))
            return

        set_status(f"Обработка: {path.name} …")
        root.update_idletasks()

        suffix = path.suffix.lower()
        color_mode = color_var.get()
        basis = basis_var.get()
        if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            if color_mode == "gray":
                task = {
                    "task_type": "image",
                    "kwargs": {
                        "image_path": path,
                        "output_dir": OUTPUT_DIR,
                        "compression_ratio": ratio,
                        "algorithm": algo,
                        "block_edge": block_edge,
                        "force_full_frame": False,
                        "visualize_convergence": False,
                        "verbose": False,
                        "basis": basis,
                    },
                }
            else:
                task = {
                    "task_type": "color_image",
                    "kwargs": {
                        "image_path": path,
                        "output_dir": OUTPUT_DIR,
                        "compression_ratio": ratio,
                        "algorithm": algo,
                        "block_edge": block_edge,
                        "force_full_frame": False,
                        "verbose": False,
                        "basis": basis,
                        "color_mode": color_mode,
                    },
                }
        elif suffix in [".npy", ".txt"]:
            task = {
                "task_type": "signal",
                "kwargs": {
                    "signal_path": path,
                    "output_dir": OUTPUT_DIR,
                    "compression_ratio": ratio,
                    "algorithm": algo,
                    "verbose": False,
                    "basis": basis,
                },
            }
        else:
            set_status("Неподдерживаемый формат.")
            return

        ctx = mp.get_context("spawn")
        worker_queue[0] = ctx.Queue()
        worker_process[0] = ctx.Process(
            target=_run_processing_task,
            args=(task, worker_queue[0]),
        )
        _set_running_state(True)
        stop_requested["value"] = False
        worker_process[0].start()
        root.after(120, poll_worker)

    btn_run.configure(command=on_run)
    btn_stop.configure(command=stop_processing)

    set_status(
        "«Выбрать файл…» — проводник (папка по умолчанию: examples/input). "
        "Или вставьте полный путь ниже."
    )

    def on_close() -> None:
        stop_processing()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


def _run_app_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams["toolbar"] = "None"

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, TextBox

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected: list[Optional[Path]] = [None]

    fig = plt.figure(figsize=(12, 8.5), facecolor=_COLOR_BG)
    fig.canvas.manager.set_window_title("IMCS — кодирование и декодирование")

    Y0, Y1 = 0.30, 0.90
    H = Y1 - Y0

    ax_btn = fig.add_axes([0.05, 0.78, 0.18, 0.075])
    ax_btn.set_facecolor(_COLOR_PANEL)
    btn_pick = Button(ax_btn, "Выбрать файл…", color="#dadce0", hovercolor=_COLOR_ACCENT)
    btn_pick.label.set_fontsize(10)

    ax_lbl = fig.add_axes([0.04, Y0, 0.20, 0.48])
    ax_lbl.axis("off")
    ax_lbl.set_facecolor(_COLOR_BG)
    path_hint = ax_lbl.text(
        0.5,
        0.98,
        "Файл не выбран.\nНажмите кнопку — откроется\nпроводник в папке\nexamples/input",
        ha="center",
        va="top",
        fontsize=8,
        color="#5f6368",
        linespacing=1.35,
    )

    ax1 = fig.add_axes([0.27, Y0, 0.34, H], facecolor=_COLOR_PANEL)
    ax2 = fig.add_axes([0.63, Y0, 0.35, H], facecolor=_COLOR_PANEL)
    ax1.set_title("Исходное", fontsize=11, pad=8)
    ax2.set_title("После IMCS", fontsize=11, pad=8)
    for a in (ax1, ax2):
        a.tick_params(labelsize=8)

    ax_status = fig.add_axes([0.06, 0.243, 0.88, 0.055])
    ax_status.axis("off")
    ax_status.set_facecolor(_COLOR_BG)
    status = ax_status.text(
        0.01,
        0.5,
        "",
        fontsize=9,
        va="center",
        ha="left",
        wrap=True,
        color="#202124",
    )

    def set_status(msg: str) -> None:
        status.set_text(msg)
        fig.canvas.draw_idle()

    fig.text(0.5, 0.965, "IMCS", fontsize=14, fontweight="bold", ha="center", color="#202124")
    fig.text(
        0.5,
        0.935,
        f"Папка по умолчанию: {INPUT_DIR}",
        fontsize=8,
        ha="center",
        color="#5f6368",
    )

    ax_path = fig.add_axes([0.08, 0.188, 0.84, 0.048])
    ax_path.set_facecolor(_COLOR_PANEL)
    tb_path = TextBox(ax_path, "", initial="", color=_COLOR_PANEL, hovercolor="#f1f3f4")
    tb_path.text_disp.set_fontsize(9)
    fig.text(
        0.08,
        0.242,
        "Путь к файлу:",
        fontsize=9,
        color="#202124",
        transform=fig.transFigure,
    )

    rax_algo = fig.add_axes([0.06, 0.10, 0.22, 0.11])
    rax_algo.set_facecolor(_COLOR_PANEL)
    _ai = {"omp": 0, "ista": 1, "fista": 2, "sa": 3}.get(DEFAULT_ALGORITHM, 0)
    algo_radio = RadioButtons(
        rax_algo,
        ("omp", "ista", "fista", "sa"),
        active=_ai,
        label_props={"fontsize": [8, 8, 8, 8]},
    )

    fig.text(
        0.58,
        0.122,
        "цвет:",
        fontsize=9,
        ha="right",
        va="center",
        color="#202124",
        transform=fig.transFigure,
    )
    ax_color = fig.add_axes([0.59, 0.102, 0.12, 0.08])
    ax_color.set_facecolor(_COLOR_PANEL)
    color_radio = RadioButtons(
        ax_color,
        ("gray", "rgb", "ycbcr"),
        active={"gray": 0, "rgb": 1, "ycbcr": 2}.get(DEFAULT_COLOR_MODE, 0),
        label_props={"fontsize": [8, 8, 8]},
    )

    fig.text(
        0.75,
        0.122,
        "basis:",
        fontsize=9,
        ha="right",
        va="center",
        color="#202124",
        transform=fig.transFigure,
    )
    ax_basis = fig.add_axes([0.76, 0.102, 0.12, 0.08])
    ax_basis.set_facecolor(_COLOR_PANEL)
    basis_radio = RadioButtons(
        ax_basis,
        ("dct", "wavelet"),
        active={"dct": 0, "wavelet": 1}.get(DEFAULT_BASIS, 0),
        label_props={"fontsize": [8, 8]},
    )

    _y_ctrl = 0.024 + 0.042 / 2
    fig.text(
        0.19,
        _y_ctrl,
        "ratio:",
        fontsize=9,
        ha="right",
        va="center",
        color="#202124",
        transform=fig.transFigure,
    )
    ax_r = fig.add_axes([0.20, 0.022, 0.14, 0.042])
    ax_r.set_facecolor(_COLOR_PANEL)
    tb_ratio = TextBox(ax_r, "", initial=str(DEFAULT_RATIO), color=_COLOR_PANEL)
    fig.text(
        0.38,
        _y_ctrl,
        "блок:",
        fontsize=9,
        ha="right",
        va="center",
        color="#202124",
        transform=fig.transFigure,
    )
    ax_b = fig.add_axes([0.39, 0.022, 0.14, 0.042])
    ax_b.set_facecolor(_COLOR_PANEL)
    tb_block = TextBox(ax_b, "", initial=str(DEFAULT_BLOCK_EDGE), color=_COLOR_PANEL)

    ax_run = fig.add_axes([0.78, 0.022, 0.16, 0.078])
    btn_run = Button(
        ax_run,
        "Запустить\nкодирование",
        color="#dadce0",
        hovercolor=_COLOR_ACCENT,
    )
    btn_run.label.set_fontsize(9)
    btn_run.label.set_fontweight("bold")

    def on_pick(_event) -> None:
        set_status("Открытие диалога выбора файла…")
        p = pick_file(INPUT_DIR, title="Выберите изображение или сигнал")
        if p is None:
            set_status("Выбор файла отменён.")
            return
        selected[0] = p
        tb_path.set_val(str(p.resolve()))
        short = p.name
        if len(short) > 28:
            short = short[:25] + "…"
        path_hint.set_text(f"Выбрано:\n{short}")
        path_hint.set_color("#202124")
        _draw_preview(ax1, ax2, fig.canvas, p)
        set_status(f"Предпросмотр: {p.name}")

    btn_pick.on_clicked(on_pick)

    def _parse_ratio() -> float:
        r = float(_textbox_str_mpl(tb_ratio).replace(",", "."))
        if not 0 < r < 1:
            raise ValueError("ratio ∈ (0, 1)")
        return r

    def _parse_block() -> int:
        return int(_textbox_str_mpl(tb_block))

    def _resolve_path() -> Optional[Path]:
        raw = _textbox_str_mpl(tb_path)
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                selected[0] = p
                return p
            set_status(f"Файл не найден: {raw}")
            return None
        if selected[0] is not None and selected[0].is_file():
            return selected[0]
        set_status("Выберите файл кнопкой «Выбрать файл…» или вставьте полный путь.")
        return None

    def on_run(_event) -> None:
        path = _resolve_path()
        if path is None:
            return
        algo = algo_radio.value_selected
        try:
            ratio = _parse_ratio()
            block_edge = _parse_block()
        except ValueError as e:
            set_status(str(e))
            return

        set_status(f"Обработка: {path.name} …")
        fig.canvas.flush_events()

        suffix = path.suffix.lower()
        current_color_mode = color_radio.value_selected
        current_basis = basis_radio.value_selected
        try:
            if suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                if current_color_mode == "gray":
                    res = run_image(
                        path,
                        OUTPUT_DIR,
                        compression_ratio=ratio,
                        algorithm=algo,
                        block_edge=block_edge,
                        force_full_frame=False,
                        visualize_convergence=False,
                        verbose=False,
                        basis=current_basis,
                    )
                else:
                    res = run_color_image(
                        path,
                        OUTPUT_DIR,
                        compression_ratio=ratio,
                        algorithm=algo,
                        block_edge=block_edge,
                        force_full_frame=False,
                        verbose=False,
                        basis=current_basis,
                        color_mode=current_color_mode,
                    )
                if res is None:
                    set_status("Не удалось загрузить изображение.")
                    return
                ax1.clear()
                ax2.clear()
                if getattr(res.original, "ndim", 2) == 3:
                    ax1.imshow(np.clip(res.original, 0, 255).astype(np.uint8))
                    ax2.imshow(np.clip(res.reconstructed, 0, 255).astype(np.uint8))
                else:
                    ax1.imshow(res.original, cmap="gray", vmin=0, vmax=255)
                    ax2.imshow(res.reconstructed, cmap="gray", vmin=0, vmax=255)
                ax1.set_title("Исходное", fontsize=11, pad=8)
                ax2.set_title("После IMCS", fontsize=11, pad=8)
                ax1.axis("off")
                ax2.axis("off")
                set_status(
                    f"PSNR: {res.metrics['psnr']:.2f} dB | "
                    f"SSIM: {res.metrics['ssim']:.4f} | "
                    f"код {res.t_encode:.3f} с | декод {res.t_decode:.3f} с\n{res.output_subdir}"
                )
            elif suffix in [".npy", ".txt"]:
                res = run_signal(
                    path,
                    OUTPUT_DIR,
                    compression_ratio=ratio,
                    algorithm=algo,
                    verbose=False,
                    basis=current_basis,
                )
                if res is None:
                    set_status("Не удалось загрузить сигнал.")
                    return
                ax1.clear()
                ax2.clear()
                o = np.asarray(res.original, dtype=np.float64).ravel()
                r = np.asarray(res.reconstructed, dtype=np.float64).ravel()
                ax1.plot(o, color=_COLOR_ACCENT, lw=1.2, alpha=0.9)
                ax2.plot(r, color="#d93025", lw=1.2, alpha=0.9)
                ax1.set_title("Исходный сигнал", fontsize=11, pad=8)
                ax2.set_title("После IMCS", fontsize=11, pad=8)
                ax1.grid(True, alpha=0.3)
                ax2.grid(True, alpha=0.3)
                ax1.tick_params(labelsize=8)
                ax2.tick_params(labelsize=8)
                set_status(
                    f"PSNR: {res.metrics['psnr']:.2f} dB | "
                    f"SSIM: {res.metrics['ssim']:.4f} | "
                    f"код {res.t_encode:.3f} с | декод {res.t_decode:.3f} с\n{res.output_subdir}"
                )
            else:
                set_status("Неподдерживаемый формат.")
        except Exception as e:
            set_status(f"Ошибка: {e}")

        fig.canvas.draw_idle()

    btn_run.on_clicked(on_run)
    set_status(
        "«Выбрать файл…» — проводник (папка по умолчанию: examples/input). "
        "Или вставьте полный путь ниже."
    )
    plt.show()
