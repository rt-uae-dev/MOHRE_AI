#!/usr/bin/env python3
"""GUI application for MOHRE document processing."""

import os
import shutil
import threading
import json
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

# Optional drag and drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND_AVAILABLE = True
except Exception:
    TKDND_AVAILABLE = False

from mohre_ai.main_pipeline import main as run_full_pipeline
from mohre_ai.pdf_converter import convert_pdf_to_jpg
from mohre_ai.yolo_crop_ocr_pipeline import run_yolo_crop, run_enhanced_ocr
from mohre_ai.structure_with_gemini import structure_with_gemini


def run_gui():
    """Launch the main GUI window."""
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
    """Window for manual file processing."""

    def __init__(self, master):
        super().__init__(master)
        self.title("Manual Processing")
        self.geometry("500x400")
        self.file_paths = []

        self.file_area = tk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        self.file_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if TKDND_AVAILABLE:
            self.file_area.drop_target_register(DND_FILES)
            self.file_area.dnd_bind("<<Drop>>", self._drop_files)

        button_frame = tk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(button_frame, text="Add Files", command=self._browse_files).pack(side=tk.LEFT)
        self.start_button = tk.Button(
            button_frame, text="Start Processing", command=self._start_processing, state=tk.DISABLED
        )
        self.start_button.pack(side=tk.RIGHT)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack(pady=5)

    def _browse_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Documents", "*.pdf *.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        self._add_files(paths)

    def _drop_files(self, event):
        paths = self.tk.splitlist(event.data)
        self._add_files(paths)

    def _add_files(self, paths):
        for path in paths:
            if path and path not in self.file_paths:
                self.file_paths.append(path)
                self._add_file_widget(path)
        if self.file_paths:
            self.start_button.config(state=tk.NORMAL)

    def _add_file_widget(self, path):
        row = tk.Frame(self.file_area)
        row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row, text=os.path.basename(path), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="X", command=lambda p=path, r=row: self._remove_file(p, r)).pack(side=tk.RIGHT)

    def _remove_file(self, path, row):
        if path in self.file_paths:
            self.file_paths.remove(path)
            row.destroy()
        if not self.file_paths:
            self.start_button.config(state=tk.DISABLED)

    def _start_processing(self):
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            output_dir = os.path.join("data", "processed", "manual")
        os.makedirs(output_dir, exist_ok=True)

        threading.Thread(
            target=self._process_files, args=(self.file_paths.copy(), output_dir), daemon=True
        ).start()
        self.status_label.config(text="Processing...")

    def _process_files(self, paths, output_dir):
        temp_dir = tempfile.mkdtemp(prefix="mohre_manual_")
        try:
            for file_path in paths:
                try:
                    images = []
                    if file_path.lower().endswith(".pdf"):
                        images = convert_pdf_to_jpg(file_path, temp_dir)
                    else:
                        temp_path = os.path.join(temp_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, temp_path)
                        images = [temp_path]

                    for img in images:
                        cropped = run_yolo_crop(img, temp_dir)
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
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.status_label.config(text="Processing complete")
        messagebox.showinfo("MOHRE", "Manual processing completed")


if __name__ == "__main__":  # pragma: no cover
    run_gui()
