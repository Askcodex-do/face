"""Tkinter interface for LightSwapConverter.

Tkinter is part of the Python standard library, so it adds nothing to the
install size and starts instantly on a 2 GB machine. The window is a thin shell
around :class:`~LightSwapConverter.core.processor.VideoFaceProcessor`: it
collects options, runs the conversion on a worker thread and streams progress,
preview frames and log lines back to the widgets through a queue.

Nothing in the interface talks to the network. Every file is chosen locally.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from ..core.processor import ProcessorError
from ..core.swapper import ProcessingStats, VideoFaceProcessor
from ..core.video_reader import VideoReadError, VideoReader
from ..utils.config import ASSETS_DIR, AppConfig
from ..utils.logger import get_logger, get_memory_handler, setup_logging

log = get_logger("gui.window")

IMAGE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
]
VIDEO_TYPES = [
    ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
    ("All files", "*.*"),
]

#: How often the interface polls the worker queue and the log buffer, in ms.
POLL_INTERVAL_MS = 120


class ConverterWindow(tk.Tk):
    """Main application window."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.config_obj = config or AppConfig()
        setup_logging(self.config_obj.logging)

        self.title("LightSwapConverter")
        self.minsize(880, 620)
        self._set_icon()

        # Communication with the worker thread.
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._processor: Optional[VideoFaceProcessor] = None
        self._cancelling = False

        # Tkinter variables bound to the widgets.
        self._source_face_var = tk.StringVar(value=self.config_obj.last_source_face)
        self._target_video_var = tk.StringVar(value=self.config_obj.last_source_video)
        self._output_video_var = tk.StringVar(value=self.config_obj.last_output_video)
        self._status_var = tk.StringVar(value="Ready. Choose a source face and a video.")
        self._progress_var = tk.DoubleVar(value=0.0)
        self._log_verbosity_var = tk.StringVar(value="INFO")

        self._resize_width_var = tk.IntVar(value=self.config_obj.video.resize_width)
        self._resize_height_var = tk.IntVar(value=self.config_obj.video.resize_height)
        self._max_frames_var = tk.IntVar(value=self.config_obj.video.max_frames)
        self._codec_var = tk.StringVar(value=self.config_obj.video.codec)
        self._copy_audio_var = tk.BooleanVar(value=self.config_obj.video.copy_audio)
        self._feather_var = tk.DoubleVar(value=self.config_obj.blend.feather)
        self._opacity_var = tk.DoubleVar(value=self.config_obj.blend.opacity)
        self._seamless_var = tk.BooleanVar(value=self.config_obj.blend.seamless)
        self._color_match_var = tk.DoubleVar(value=self.config_obj.transform.color_match)
        self._sharpen_var = tk.DoubleVar(value=self.config_obj.transform.sharpen)
        self._threads_var = tk.IntVar(value=self.config_obj.performance.num_threads)

        self._preview_image: Optional[np.ndarray] = None
        self._preview_photo: Optional[tk.PhotoImage] = None
        self._last_log_count = 0

        self._build_menu()
        self._build_layout()
        self._apply_cpu_limits()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_INTERVAL_MS, self._poll)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _set_icon(self) -> None:
        """Use ``assets/icon.png`` as the window icon when it is present."""
        icon = ASSETS_DIR / "icon.png"
        if not icon.is_file():
            return
        try:
            self._icon_photo = tk.PhotoImage(file=str(icon))
            self.iconphoto(True, self._icon_photo)
        except tk.TclError:  # pragma: no cover - depends on the local install
            log.debug("could not load window icon from %s", icon)

    def _apply_cpu_limits(self) -> None:
        """Cap OpenCV's thread pool, matching the configuration."""
        threads = max(1, int(self.config_obj.performance.num_threads))
        try:
            cv2.setNumThreads(threads)
        except cv2.error:  # pragma: no cover - older OpenCV builds
            log.debug("cv2.setNumThreads is unavailable")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open source face...", command=self._choose_source_face)
        file_menu.add_command(label="Open target video...", command=self._choose_target_video)
        file_menu.add_command(label="Choose output file...", command=self._choose_output)
        file_menu.add_separator()
        file_menu.add_command(label="Save settings", command=self._save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Preview one frame", command=self._on_preview)
        tools_menu.add_command(label="Clear log", command=self._clear_log)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=8)
        container.pack(fill="both", expand=True)

        panes = ttk.PanedWindow(container, orient="vertical")
        panes.pack(fill="both", expand=True)

        top = ttk.Frame(panes)
        panes.add(top, weight=3)

        self._build_files_section(top)
        self._build_options_section(top)
        self._build_actions_section(top)

        bottom = ttk.Frame(panes)
        panes.add(bottom, weight=2)
        self._build_progress_section(bottom)
        self._build_output_area(bottom)

    def _build_files_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Files", padding=8)
        frame.pack(fill="x", pady=(0, 6))
        frame.columnconfigure(1, weight=1)

        rows = (
            ("Source face image:", self._source_face_var, self._choose_source_face),
            ("Target video:", self._target_video_var, self._choose_target_video),
            ("Output video:", self._output_video_var, self._choose_output),
        )
        for row, (label, variable, command) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            entry = ttk.Entry(frame, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            ttk.Button(frame, text="Browse...", command=command, width=10).grid(
                row=row, column=2, sticky="e", pady=2
            )

    def _build_options_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Options", padding=8)
        frame.pack(fill="x", pady=(0, 6))

        # Row 0: resize, frame limit and codec.
        ttk.Label(frame, text="Resize width:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            frame, from_=0, to=4096, increment=2, width=7, textvariable=self._resize_width_var
        ).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(frame, text="Resize height:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            frame, from_=0, to=4096, increment=2, width=7, textvariable=self._resize_height_var
        ).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(frame, text="Max frames (0 = all):").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(
            frame, from_=0, to=1_000_000, increment=10, width=9, textvariable=self._max_frames_var
        ).grid(row=0, column=5, sticky="w", padx=(4, 12))
        ttk.Label(frame, text="Codec:").grid(row=0, column=6, sticky="w")
        ttk.Combobox(
            frame, width=6, textvariable=self._codec_var,
            values=("mp4v", "XVID", "MJPG", "DIVX"),
        ).grid(row=0, column=7, sticky="w", padx=4)

        # Row 1: blend options.
        ttk.Label(frame, text="Feather:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(
            frame, from_=0.05, to=1.0, orient="horizontal", length=110,
            variable=self._feather_var,
        ).grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(frame, text="Opacity:").grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Scale(
            frame, from_=0.1, to=1.0, orient="horizontal", length=110,
            variable=self._opacity_var,
        ).grid(row=1, column=4, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(frame, text="Colour match:").grid(row=1, column=6, sticky="w", pady=(8, 0))
        ttk.Scale(
            frame, from_=0.0, to=1.0, orient="horizontal", length=110,
            variable=self._color_match_var,
        ).grid(row=1, column=7, sticky="ew", pady=(8, 0))

        # Row 2: quality and resource options.
        ttk.Checkbutton(
            frame, text="Seamless blending (slower)", variable=self._seamless_var
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            frame, text="Copy audio track", variable=self._copy_audio_var
        ).grid(row=2, column=3, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="CPU threads:").grid(row=2, column=6, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            frame, from_=1, to=16, width=5, textvariable=self._threads_var
        ).grid(row=2, column=7, sticky="w", padx=4, pady=(8, 0))

        ttk.Label(frame, text="Sharpness:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(
            frame, from_=0.0, to=1.0, orient="horizontal", length=110,
            variable=self._sharpen_var,
        ).grid(row=3, column=1, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_actions_section(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 6))

        self._preview_button = ttk.Button(
            frame, text="Preview frame", command=self._on_preview
        )
        self._preview_button.pack(side="left")

        self._convert_button = ttk.Button(
            frame, text="Convert video", command=self._on_convert
        )
        self._convert_button.pack(side="left", padx=6)

        self._cancel_button = ttk.Button(
            frame, text="Cancel", command=self._on_cancel, state="disabled"
        )
        self._cancel_button.pack(side="left")

        ttk.Button(frame, text="Open output folder", command=self._open_output_folder).pack(
            side="right"
        )

    def _build_progress_section(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 4))
        self._progress = ttk.Progressbar(
            frame, mode="determinate", maximum=100.0, variable=self._progress_var
        )
        self._progress.pack(fill="x", side="top")
        ttk.Label(frame, textvariable=self._status_var).pack(anchor="w", pady=(4, 0))

    def _build_output_area(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        preview_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview")
        self._canvas = tk.Canvas(
            preview_tab, background="#1e1e1e", highlightthickness=0
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda _event: self._render_preview())

        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Log")
        controls = ttk.Frame(log_tab)
        controls.pack(fill="x")
        ttk.Label(controls, text="Show:").pack(side="left")
        ttk.Combobox(
            controls, width=8, state="readonly", textvariable=self._log_verbosity_var,
            values=("DEBUG", "INFO", "WARNING", "ERROR"),
        ).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear", command=self._clear_log).pack(side="left")

        self._log_text = tk.Text(
            log_tab, height=10, wrap="none", background="#111111",
            foreground="#dddddd", insertbackground="#dddddd",
        )
        scrollbar = ttk.Scrollbar(log_tab, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _choose_source_face(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose the source face image", filetypes=IMAGE_TYPES
        )
        if path:
            self._source_face_var.set(path)

    def _choose_target_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose the target video", filetypes=VIDEO_TYPES
        )
        if not path:
            return
        self._target_video_var.set(path)
        if not self._output_video_var.get():
            target = Path(path)
            self._output_video_var.set(
                str(target.with_name(f"{target.stem}_swapped.mp4"))
            )

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Choose the output video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")],
        )
        if path:
            self._output_video_var.set(path)

    def _open_output_folder(self) -> None:
        """Reveal the output folder in the platform file manager."""
        path = self._output_video_var.get()
        folder = Path(path).parent if path else Path.cwd()
        if not folder.exists():
            messagebox.showinfo("LightSwapConverter", "The output folder does not exist yet.")
            return
        import subprocess
        import sys

        try:
            if sys.platform.startswith("win"):
                import os

                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("cannot open %s: %s", folder, exc)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _collect_config(self) -> AppConfig:
        """Read the widgets into a fresh configuration object."""
        config = self.config_obj
        config.video.resize_width = _safe_int(self._resize_width_var, 0)
        config.video.resize_height = _safe_int(self._resize_height_var, 0)
        config.video.max_frames = _safe_int(self._max_frames_var, 0)
        config.video.codec = self._codec_var.get().strip() or "mp4v"
        config.video.copy_audio = bool(self._copy_audio_var.get())

        config.blend.feather = _safe_float(self._feather_var, 0.35)
        config.blend.opacity = _safe_float(self._opacity_var, 1.0)
        config.blend.seamless = bool(self._seamless_var.get())

        config.transform.color_match = _safe_float(self._color_match_var, 0.7)
        config.transform.sharpen = _safe_float(self._sharpen_var, 0.25)

        config.performance.num_threads = max(1, _safe_int(self._threads_var, 2))
        config.logging.level = self._log_verbosity_var.get()

        config.last_source_face = self._source_face_var.get()
        config.last_source_video = self._target_video_var.get()
        config.last_output_video = self._output_video_var.get()
        return config

    def _apply_config(self) -> None:
        """Push the configuration values back into the widgets."""
        config = self.config_obj
        self._resize_width_var.set(config.video.resize_width)
        self._resize_height_var.set(config.video.resize_height)
        self._max_frames_var.set(config.video.max_frames)
        self._codec_var.set(config.video.codec)
        self._copy_audio_var.set(config.video.copy_audio)
        self._feather_var.set(config.blend.feather)
        self._opacity_var.set(config.blend.opacity)
        self._seamless_var.set(config.blend.seamless)
        self._color_match_var.set(config.transform.color_match)
        self._sharpen_var.set(config.transform.sharpen)
        self._threads_var.set(config.performance.num_threads)
        self._source_face_var.set(config.last_source_face)
        self._target_video_var.set(config.last_source_video)
        self._output_video_var.set(config.last_output_video)

    def _save_settings(self) -> None:
        config = self._collect_config()
        errors = config.validate()
        if errors:
            messagebox.showwarning(
                "Invalid settings", "Please fix these values:\n\n" + "\n".join(errors)
            )
            return
        try:
            path = config.save()
        except OSError as exc:
            messagebox.showerror("LightSwapConverter", f"Cannot save settings:\n{exc}")
            return
        self._set_status(f"Settings saved to {path}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _validate_inputs(self, need_output: bool) -> bool:
        source = self._source_face_var.get().strip()
        target = self._target_video_var.get().strip()
        if not source or not Path(source).is_file():
            messagebox.showwarning("LightSwapConverter", "Choose a source face image.")
            return False
        if not target or not Path(target).is_file():
            messagebox.showwarning("LightSwapConverter", "Choose a target video.")
            return False
        if need_output:
            output = self._output_video_var.get().strip()
            if not output:
                messagebox.showwarning("LightSwapConverter", "Choose an output file.")
                return False
        return True

    def _on_preview(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("LightSwapConverter", "A job is already running.")
            return
        if not self._validate_inputs(need_output=False):
            return

        config = self._collect_config()
        self._apply_cpu_limits()
        source = self._source_face_var.get()
        target = self._target_video_var.get()
        self._set_status("Rendering a preview frame...")
        self._preview_button.configure(state="disabled")

        def work() -> None:
            try:
                frame = self._grab_preview_frame(target)
                processor = VideoFaceProcessor(config)
                result = processor.preview_swap(source, frame, side_by_side=True)
                self._queue.put(("preview", result))
                self._queue.put(("status", "Preview ready."))
            except (ProcessorError, VideoReadError, cv2.error, ValueError) as exc:
                self._queue.put(("error", str(exc)))
            except Exception as exc:  # pragma: no cover - unexpected failure
                log.error("preview failed: %s", traceback.format_exc())
                self._queue.put(("error", f"Unexpected error: {exc}"))
            finally:
                self._queue.put(("preview_done", None))

        self._worker = threading.Thread(target=work, name="preview", daemon=True)
        self._worker.start()

    def _grab_preview_frame(self, video_path: str) -> np.ndarray:
        """Return a frame from the middle of the video, or the first one."""
        with VideoReader(video_path, self.config_obj.video) as reader:
            info = reader.info
            if info.frame_count > 10:
                reader.seek(info.frame_count // 2)
            ok, frame = reader.read()
            if not ok or frame is None:
                raise VideoReadError(f"cannot decode a frame from {video_path}")
            return frame

    def _on_convert(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("LightSwapConverter", "A job is already running.")
            return
        if not self._validate_inputs(need_output=True):
            return

        config = self._collect_config()
        errors = config.validate()
        if errors:
            messagebox.showwarning(
                "Invalid settings", "Please fix these values:\n\n" + "\n".join(errors)
            )
            return

        self._apply_cpu_limits()
        self._cancelling = False
        self._progress_var.set(0.0)
        self._convert_button.configure(state="disabled")
        self._preview_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")
        self._set_status("Starting conversion...")

        source = self._source_face_var.get()
        target = self._target_video_var.get()
        output = self._output_video_var.get()

        def work() -> None:
            processor = VideoFaceProcessor(config)
            self._processor = processor
            try:
                stats = processor.process_video(
                    source,
                    target,
                    output,
                    progress=lambda done, total, message: self._queue.put(
                        ("progress", done, total, message)
                    ),
                    preview=lambda index, frame: self._queue.put(
                        ("preview", frame.copy())
                    ),
                    on_error=lambda index, message: self._queue.put(
                        ("frame_error", index, message)
                    ),
                )
                self._queue.put(("finished", stats))
            except (ProcessorError, VideoReadError, cv2.error, ValueError) as exc:
                self._queue.put(("error", str(exc)))
            except Exception as exc:  # pragma: no cover - unexpected failure
                log.error("conversion failed: %s", traceback.format_exc())
                self._queue.put(("error", f"Unexpected error: {exc}"))
            finally:
                self._processor = None

        self._worker = threading.Thread(target=work, name="convert", daemon=True)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._processor is None:
            return
        self._cancelling = True
        self._processor.cancel()
        self._set_status("Cancelling after the current frame...")
        self._cancel_button.configure(state="disabled")

    # ------------------------------------------------------------------
    # Worker communication
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Drain the worker queue and the log buffer, then reschedule."""
        try:
            while True:
                message = self._queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
        self._drain_log()
        self.after(POLL_INTERVAL_MS, self._poll)

    def _handle_message(self, message: tuple) -> None:
        kind = message[0]
        if kind == "progress":
            _kind, done, total, text = message
            if total > 0:
                self._progress_var.set(min(100.0, done * 100.0 / total))
                self._set_status(f"Processing {text} of {total}...")
            else:
                self._progress_var.set(0.0)
                self._set_status(f"Processing {text}...")
        elif kind == "preview":
            self._preview_image = message[1]
            self._render_preview()
        elif kind == "preview_done":
            self._preview_button.configure(state="normal")
            if self._worker is not None and not self._worker.is_alive():
                self._worker = None
        elif kind == "status":
            self._set_status(message[1])
        elif kind == "frame_error":
            log.warning("frame %s failed: %s", message[1], message[2])
        elif kind == "finished":
            self._on_finished(message[1])
        elif kind == "error":
            self._on_error(message[1])

    def _on_finished(self, stats: ProcessingStats) -> None:
        self._convert_button.configure(state="normal")
        self._preview_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self._progress_var.set(100.0)
        self._set_status(f"Finished. {stats.summary()}")
        self._worker = None
        if self._cancelling:
            self._cancelling = False
            return
        messagebox.showinfo(
            "Conversion finished",
            f"{stats.summary()}\n\nOutput:\n{stats.output_path}",
        )

    def _on_error(self, message: str) -> None:
        self._convert_button.configure(state="normal")
        self._preview_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self._set_status(f"Failed: {message}")
        self._worker = None
        messagebox.showerror("LightSwapConverter", message)

    def _drain_log(self) -> None:
        """Append new lines from the in-memory log handler to the log widget."""
        handler = get_memory_handler()
        if handler is None:
            return
        lines = handler.records()
        if len(lines) <= self._last_log_count:
            self._last_log_count = len(lines)
            return
        threshold = self._log_verbosity_var.get().upper()
        allowed = {
            "DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4
        }.get(threshold, 1)
        for line in lines[self._last_log_count :]:
            if _line_level(line) >= allowed:
                self._log_text.insert("end", line + "\n")
        self._last_log_count = len(lines)
        self._log_text.see("end")

    def _clear_log(self) -> None:
        self._log_text.delete("1.0", "end")
        handler = get_memory_handler()
        if handler is not None:
            handler.drain()
        self._last_log_count = 0

    # ------------------------------------------------------------------
    # Preview rendering
    # ------------------------------------------------------------------

    def _render_preview(self) -> None:
        """Draw the current preview frame scaled to fit the canvas."""
        if self._preview_image is None:
            return
        canvas_width = max(1, self._canvas.winfo_width())
        canvas_height = max(1, self._canvas.winfo_height())
        image = self._preview_image
        scale = min(
            canvas_width / image.shape[1], canvas_height / image.shape[0], 1.0
        )
        target = (
            max(1, int(image.shape[1] * scale)),
            max(1, int(image.shape[0] * scale)),
        )
        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # PhotoImage accepts raw PPM bytes, which avoids a Pillow dependency.
        header = f"P6 {target[0]} {target[1]} 255 ".encode("ascii")
        self._preview_photo = tk.PhotoImage(
            data=header + rgb.tobytes(), format="PPM"
        )
        self._canvas.delete("all")
        self._canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=self._preview_photo
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About LightSwapConverter",
            "LightSwapConverter\n\n"
            "A lightweight, fully offline video face replacement tool.\n"
            "Classical computer vision only: no ONNX, no TensorFlow, no PyTorch.\n\n"
            f"Python {_python_version()}",
        )

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno(
                "LightSwapConverter", "A job is running. Stop it and exit?"
            ):
                return
            if self._processor is not None:
                self._processor.cancel()
        try:
            self._collect_config().save()
        except OSError as exc:
            log.warning("could not save settings on exit: %s", exc)
        self.destroy()


def _line_level(line: str) -> int:
    """Extract the numeric level from a formatted log line."""
    for level, value in (
        ("CRITICAL", 4), ("ERROR", 3), ("WARNING", 2), ("INFO", 1), ("DEBUG", 0)
    ):
        if f"[{level}" in line:
            return value
    return 1


def _safe_int(variable: tk.IntVar, default: int) -> int:
    """Read an IntVar without raising when the widget holds invalid text."""
    try:
        return int(variable.get())
    except (tk.TclError, ValueError):
        return default


def _safe_float(variable: tk.DoubleVar, default: float) -> float:
    try:
        return float(variable.get())
    except (tk.TclError, ValueError):
        return default


def _python_version() -> str:
    import platform

    return platform.python_version()


def run(config: Optional[AppConfig] = None) -> None:
    """Create the window and start the Tkinter event loop."""
    window = ConverterWindow(config)
    window.mainloop()
