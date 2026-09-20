"""Tkinter desktop window.

Tkinter is used because it ships with CPython on Windows: no extra runtime, no
bundled Qt binaries, and a small memory footprint. The window only collects
paths and options and hands them to :class:`core.processor.Processor`; all
image work happens off the UI thread.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from core.processor import ConversionJob, ConversionResult, Processor
from utils.config import APP_NAME, APP_VERSION, AppConfig
from utils.logger import get_logger

logger = get_logger("gui.window")

VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All files", "*.*")]
IMAGE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
METHODS = ("copy", "blend", "monochrome")
FRAME_SKIPS = ("1", "2", "3", "5")


class MainWindow:
    """Main application window."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.processor = Processor(self.config)

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.geometry("640x420")
        self.root.minsize(560, 380)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._draining = False

        self.source_var = tk.StringVar()
        self.face_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.method_var = tk.StringVar(value=METHODS[0])
        self.skip_var = tk.StringVar(value=FRAME_SKIPS[0])
        self.status_var = tk.StringVar(value="Ready")

        self._build_widgets()

    # -------------------------------------------------------------- layout
    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Source video").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse", command=self._pick_video).grid(row=0, column=2)

        ttk.Label(frame, text="Replacement face").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.face_var).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse", command=self._pick_face).grid(row=1, column=2)

        ttk.Label(frame, text="Output video").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse", command=self._pick_output).grid(row=2, column=2)

        options = ttk.LabelFrame(frame, text="Options", padding=8)
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Method").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            options, textvariable=self.method_var, values=METHODS, width=12, state="readonly"
        ).grid(row=0, column=1, padx=(6, 16))

        ttk.Label(options, text="Frame step").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            options, textvariable=self.skip_var, values=FRAME_SKIPS, width=6, state="readonly"
        ).grid(row=0, column=3, sticky="w", padx=6)

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 2))

        ttk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=5, column=0, columnspan=3, sticky="ew"
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.start_button = ttk.Button(buttons, text="Start", command=self.on_start)
        self.start_button.pack(side="right")
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=6)

    # ------------------------------------------------------------ file pickers
    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(title="Select source video", filetypes=VIDEO_TYPES)
        if path:
            self.source_var.set(path)
            if not self.output_var.get():
                self.output_var.set(str(self.processor.default_output_path(path)))

    def _pick_face(self) -> None:
        path = filedialog.askopenfilename(title="Select replacement face", filetypes=IMAGE_TYPES)
        if path:
            self.face_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select output video",
            defaultextension=".mp4",
            filetypes=VIDEO_TYPES,
        )
        if path:
            self.output_var.set(path)

    # --------------------------------------------------------------- actions
    def on_start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        source = self.source_var.get().strip()
        face = self.face_var.get().strip()
        if not source or not Path(source).is_file():
            messagebox.showerror(APP_NAME, "Please select a valid source video.")
            return
        if not face or not Path(face).is_file():
            messagebox.showerror(APP_NAME, "Please select a valid replacement face image.")
            return

        job = ConversionJob(
            source_video=source,
            target_face_image=face,
            output_video=self.output_var.get().strip() or None,
            frame_skip=int(self.skip_var.get()),
            method=self.method_var.get(),
            strength=self.config.processing.blend_strength,
            color_match=self.config.processing.color_match,
        )

        self._cancel.clear()
        self._set_busy(True)
        self.status_var.set("Starting...")

        self._worker = threading.Thread(
            target=self._run_job, args=(job,), name="lightswap-worker", daemon=True
        )
        self._worker.start()
        self._start_draining()

    def on_cancel(self) -> None:
        self._cancel.set()
        self.status_var.set("Cancelling...")

    def on_close(self) -> None:
        self._cancel.set()
        self.root.destroy()

    # ---------------------------------------------------------------- worker
    def _run_job(self, job: ConversionJob) -> None:
        """Runs on the worker thread; never touches Tk widgets directly."""
        try:
            result = self.processor.process(
                job,
                progress=lambda done, total, message: self._events.put(
                    ("progress", (done, total, message))
                ),
                is_cancelled=self._cancel.is_set,
            )
            self._events.put(("done", result))
        except Exception as exc:  # defensive: keep the UI alive
            logger.exception("Worker crashed: %s", exc)
            self._events.put(("error", str(exc)))

    def _start_draining(self) -> None:
        """Begin polling the worker queue, at most one poll loop at a time."""
        if self._draining:
            return
        self._draining = True
        self.root.after(100, self._drain_events)

    def _drain_events(self) -> None:
        """Poll the worker queue from the Tk main loop."""
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "progress":
                    done, total, message = payload  # type: ignore[misc]
                    self._update_progress(done, total, message)
                elif kind == "done":
                    self._draining = False
                    self._on_finished(payload)  # type: ignore[arg-type]
                    return
                elif kind == "error":
                    self._draining = False
                    self._set_busy(False)
                    self.status_var.set("Failed")
                    messagebox.showerror(APP_NAME, f"Conversion failed:\n{payload}")
                    return
        except queue.Empty:
            pass

        if self._worker is not None and self._worker.is_alive():
            self.root.after(100, self._drain_events)
        else:
            self._draining = False

    def _update_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress["value"] = min(done / total * 100.0, 100.0)
            self.status_var.set(f"{message}: {done}/{total}")
        else:
            self.status_var.set(message)

    def _on_finished(self, result: ConversionResult) -> None:
        self._set_busy(False)
        self.progress["value"] = 100.0 if result.succeeded else 0.0
        if result.cancelled:
            self.status_var.set("Cancelled")
        elif result.errors:
            self.status_var.set("Failed")
            messagebox.showerror(APP_NAME, "\n".join(result.errors))
        else:
            self.status_var.set(f"Done: {result.frames_written} frames written")
            messagebox.showinfo(APP_NAME, f"Output saved to:\n{result.output_path}")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.start_button.configure(state=state)
        self.cancel_button.configure(state="normal" if busy else "disabled")

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        self.root.mainloop()


def create_window(config: Optional[AppConfig] = None) -> MainWindow:
    """Build the main window without entering the Tk main loop."""
    return MainWindow(config)
