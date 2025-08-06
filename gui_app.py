#!/usr/bin/env python3
"""GUI application for MOHRE document processing."""

import os
import sys
import shutil
import threading
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Iterable, List

# Optional drag and drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND_AVAILABLE = True
except Exception:
    TKDND_AVAILABLE = False

# Ensure src directory is in path when running standalone
if os.path.join(os.path.dirname(__file__), "src") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from main_pipeline import main as run_full_pipeline
from pdf_converter import convert_pdf_to_jpg
from yolo_crop_ocr_pipeline import run_yolo_crop, run_enhanced_ocr
from structure_with_gemini import structure_with_gemini

TEMP_DIR = os.path.join("data", "temp")


def run_gui() -> None:
    """Launch the main GUI window.

    This function sets up the top level window that allows a user to choose
    between running the full automated pipeline or manually processing files.

    Returns:
        None

    Raises:
        RuntimeError: If the GUI fails to initialize.
    """
    root = TkinterDnD.Tk() if TKDND_AVAILABLE else tk.Tk()
    root.title("MOHRE Document Processor")
    root.geometry("400x200")

    tk.Label(root, text="Choose Processing Mode", font=("Arial", 14)).pack(pady=10)

    tk.Button(
        root,
        text="Run Full Pipeline",
        command=lambda: threading.Thread(target=run_full_pipeline, daemon=True).start(),
        width=25,
    ).pack(pady=10)

    tk.Button(
        root,
        text="Manual File Processing",
        command=lambda: ManualProcessingWindow(root),
        width=25,
    ).pack(pady=10)

    root.mainloop()


class ManualProcessingWindow(tk.Toplevel):
    """Window for manually selecting and processing documents."""

    def __init__(self, master: tk.Misc) -> None:
        """Initialize the manual processing window.

        Args:
            master: Parent widget that owns this window.

        Returns:
            None
        """
        super().__init__(master)
        self.title("Manual Processing")
        self.geometry("500x400")
        self.file_paths: List[str] = []

        self.file_area: tk.Frame = tk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        self.file_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if TKDND_AVAILABLE:
            self.file_area.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            self.file_area.dnd_bind("<<Drop>>", self._drop_files)  # type: ignore[attr-defined]

        button_frame = tk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(button_frame, text="Add Files", command=self._browse_files).pack(side=tk.LEFT)
        self.start_button: tk.Button = tk.Button(
            button_frame, text="Start Processing", command=self._start_processing, state=tk.DISABLED
        )
        self.start_button.pack(side=tk.RIGHT)

        self.status_label: tk.Label = tk.Label(self, text="")
        self.status_label.pack(pady=5)

    def _browse_files(self) -> None:
        """Open a file dialog for the user to select files.

        Returns:
            None
        """
        paths = filedialog.askopenfilenames(
            filetypes=[("Documents", "*.pdf *.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        self._add_files(paths)

    def _drop_files(self, event: tk.Event) -> None:
        """Handle file drops when drag-and-drop is available.

        Args:
            event: Tkinter drop event containing file paths.

        Returns:
            None
        """
        paths = self.tk.splitlist(event.data)  # type: ignore[attr-defined]
        self._add_files(paths)

    def _add_files(self, paths: Iterable[str]) -> None:
        """Add file paths to the list and create display widgets.

        Args:
            paths: Iterable collection of file paths selected by the user.

        Returns:
            None
        """
        for path in paths:
            if path and path not in self.file_paths:
                self.file_paths.append(path)
                self._add_file_widget(path)
        if self.file_paths:
            self.start_button.config(state=tk.NORMAL)

    def _add_file_widget(self, path: str) -> None:
        """Create a row widget showing the file name with a remove button.

        Args:
            path: File path represented by the widget.

        Returns:
            None
        """
        row = tk.Frame(self.file_area)
        row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row, text=os.path.basename(path), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        def callback(p: str = path, r: tk.Widget = row) -> None:
            self._remove_file(p, r)

        tk.Button(row, text="X", command=callback).pack(side=tk.RIGHT)

    def _remove_file(self, path: str, row: tk.Widget) -> None:
        """Remove a file from the list and destroy its widget.

        Args:
            path: File path to remove.
            row: Row widget associated with ``path``.

        Returns:
            None
        """
        if path in self.file_paths:
            self.file_paths.remove(path)
            row.destroy()
        if not self.file_paths:
            self.start_button.config(state=tk.DISABLED)

    def _start_processing(self) -> None:
        """Start processing of the selected files in a background thread.

        Returns:
            None
        """
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            output_dir = os.path.join("data", "processed", "manual")
        os.makedirs(output_dir, exist_ok=True)

        threading.Thread(
            target=self._process_files, args=(self.file_paths.copy(), output_dir), daemon=True
        ).start()
        self.status_label.config(text="Processing...")

    def _process_files(self, paths: Iterable[str], output_dir: str) -> None:
        """Process the provided files and save structured output.

        Args:
            paths: Iterable of file paths selected by the user.
            output_dir: Directory where processed outputs are saved.

        Returns:
            None

        Raises:
            RuntimeError: If file processing fails for an individual file.
        """
        os.makedirs(TEMP_DIR, exist_ok=True)
        for file_path in paths:
            try:
                images: List[str] = []
                if file_path.lower().endswith(".pdf"):
                    images = convert_pdf_to_jpg(file_path, TEMP_DIR)
                else:
                    temp_path = os.path.join(TEMP_DIR, os.path.basename(file_path))
                    shutil.copy2(file_path, temp_path)
                    images = [temp_path]

                for img in images:
                    cropped = run_yolo_crop(img, TEMP_DIR)
                    ocr = run_enhanced_ocr(cropped)
                    structured = structure_with_gemini(
                        "",
                        "",
                        "",
                        "",
                        "",
                        ocr.get("ocr_text", ""),
                        {},
                        "",
                        "manual",
                        {},
                    )
                    out_name = os.path.splitext(os.path.basename(img))[0] + "_output.json"
                    out_path = os.path.join(output_dir, out_name)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(structured, f, ensure_ascii=False, indent=2)
            except Exception as e:  # pragma: no cover - runtime safeguard
                print(f"Error processing {file_path}: {e}")
        self.status_label.config(text="Processing complete")
        messagebox.showinfo("MOHRE", "Manual processing completed")


if __name__ == "__main__":  # pragma: no cover
    run_gui()
