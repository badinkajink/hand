from __future__ import annotations
# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer

# Allow direct script execution without editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.tools.morphology_xml import (  # noqa: E402
    apply_morphology_to_qpos,
    FINGER_ORDER,
    MorphologyValues,
    create_rigid_hand_and_scene_xmls,
    extract_morphology_from_qpos,
)


@dataclass
class SliderTriplet:
    x: tk.DoubleVar
    y: tk.DoubleVar
    length: tk.DoubleVar


class MorphologyEditor:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Morphology XML Generator")

        self.base_hand_xml = tk.StringVar(value=str(PROJECT_ROOT / "assets" / "mjcf" / "hand.xml"))
        self.base_scene_xml = tk.StringVar(value=str(PROJECT_ROOT / "assets" / "mjcf" / "scene.xml"))
        self.output_dir = tk.StringVar(value=str(PROJECT_ROOT / "assets" / "mjcf" / "generated"))
        self.hand_prefix = tk.StringVar(value="hand")
        self.scene_prefix = tk.StringVar(value="scene")

        self.viewer_model: mujoco.MjModel | None = None
        self.viewer_data: mujoco.MjData | None = None
        self.viewer_lock = threading.Lock()
        self.viewer_thread: threading.Thread | None = None
        self.viewer_running = False

        self.triples: dict[str, SliderTriplet] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)

        path_frame = ttk.LabelFrame(top, text="I/O")
        path_frame.grid(row=0, column=0, sticky="ew")
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="Base Hand XML").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.base_hand_xml).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(path_frame, text="Browse", command=self._browse_hand_base).grid(row=0, column=2)

        ttk.Label(path_frame, text="Base Scene XML").grid(row=1, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.base_scene_xml).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(path_frame, text="Browse", command=self._browse_scene_base).grid(row=1, column=2)

        ttk.Label(path_frame, text="Output Dir").grid(row=2, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.output_dir).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(path_frame, text="Browse", command=self._browse_output).grid(row=2, column=2)

        ttk.Label(path_frame, text="Hand Prefix").grid(row=3, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.hand_prefix).grid(row=3, column=1, sticky="ew", padx=6)

        ttk.Label(path_frame, text="Scene Prefix").grid(row=4, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.scene_prefix).grid(row=4, column=1, sticky="ew", padx=6)

        sliders = ttk.LabelFrame(top, text="Morphology (meters)")
        sliders.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        for row, finger in enumerate(FINGER_ORDER):
            ttk.Label(sliders, text=finger.title()).grid(row=row * 2, column=0, sticky="w", pady=(6, 0))
            x_var = tk.DoubleVar(value=0.0)
            y_var = tk.DoubleVar(value=0.0)
            l_var = tk.DoubleVar(value=0.0)
            self.triples[finger] = SliderTriplet(x=x_var, y=y_var, length=l_var)

            self._slider(sliders, row * 2 + 1, 0, "x", x_var, -0.03, 0.03)
            self._slider(sliders, row * 2 + 1, 2, "y", y_var, -0.03, 0.03)
            self._slider(sliders, row * 2 + 1, 4, "len", l_var, 0.0, 0.035)

        qpos_frame = ttk.LabelFrame(top, text="Load from qpos")
        qpos_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        qpos_frame.columnconfigure(0, weight=1)

        self.qpos_text = tk.Text(qpos_frame, height=4, width=100)
        self.qpos_text.grid(row=0, column=0, columnspan=3, sticky="ew")

        self.scene_qpos = tk.BooleanVar(value=False)
        ttk.Checkbutton(qpos_frame, text="Scene qpos layout (cube+palm prefix)", variable=self.scene_qpos).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Button(qpos_frame, text="Load qpos", command=self._load_qpos_text).grid(row=1, column=1, padx=6)
        ttk.Button(qpos_frame, text="Load open keyframe", command=self._load_open_keyframe).grid(row=1, column=2)

        actions = ttk.Frame(top)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        ttk.Button(actions, text="Launch Live Viewer", command=self._start_viewer).grid(row=0, column=0)
        ttk.Button(actions, text="Apply Sliders -> Viewer", command=self._apply_to_viewer).grid(row=0, column=1, padx=6)
        ttk.Button(actions, text="Read Viewer -> Sliders", command=self._read_from_viewer).grid(row=0, column=2)
        ttk.Button(actions, text="Save Hand+Scene XML", command=self._save).grid(row=0, column=3, padx=6)
        ttk.Button(actions, text="Close Viewer", command=self._stop_viewer).grid(row=0, column=4, padx=6)
        ttk.Button(actions, text="Reset", command=self._reset).grid(row=0, column=5, padx=6)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _slider(
        self,
        parent: ttk.LabelFrame,
        row: int,
        col: int,
        label: str,
        var: tk.DoubleVar,
        lower: float,
        upper: float,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="e")
        tk.Scale(
            parent,
            from_=lower,
            to=upper,
            orient="horizontal",
            resolution=0.0005,
            showvalue=True,
            variable=var,
            length=220,
        ).grid(row=row, column=col + 1, sticky="w")

    def _browse_hand_base(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose base hand XML",
            filetypes=(("MJCF XML", "*.xml"), ("All Files", "*.*")),
            initialdir=str(PROJECT_ROOT / "assets" / "mjcf"),
        )
        if selected:
            self.base_hand_xml.set(selected)

    def _browse_scene_base(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose base scene XML",
            filetypes=(("MJCF XML", "*.xml"), ("All Files", "*.*")),
            initialdir=str(PROJECT_ROOT / "assets" / "mjcf"),
        )
        if selected:
            self.base_scene_xml.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose output directory",
            initialdir=str(PROJECT_ROOT / "assets" / "mjcf"),
        )
        if selected:
            self.output_dir.set(selected)

    def _morphology(self) -> MorphologyValues:
        return MorphologyValues(
            thumb_x=self.triples["thumb"].x.get(),
            thumb_y=self.triples["thumb"].y.get(),
            thumb_len=self.triples["thumb"].length.get(),
            index_x=self.triples["index"].x.get(),
            index_y=self.triples["index"].y.get(),
            index_len=self.triples["index"].length.get(),
            middle_x=self.triples["middle"].x.get(),
            middle_y=self.triples["middle"].y.get(),
            middle_len=self.triples["middle"].length.get(),
        )

    def _apply_morphology(self, morphology: MorphologyValues) -> None:
        self.triples["thumb"].x.set(morphology.thumb_x)
        self.triples["thumb"].y.set(morphology.thumb_y)
        self.triples["thumb"].length.set(morphology.thumb_len)
        self.triples["index"].x.set(morphology.index_x)
        self.triples["index"].y.set(morphology.index_y)
        self.triples["index"].length.set(morphology.index_len)
        self.triples["middle"].x.set(morphology.middle_x)
        self.triples["middle"].y.set(morphology.middle_y)
        self.triples["middle"].length.set(morphology.middle_len)

    def _load_qpos_text(self) -> None:
        raw = self.qpos_text.get("1.0", "end").strip()
        if not raw:
            messagebox.showerror("Load qpos", "Paste qpos values first.")
            return
        try:
            qpos = [float(v) for v in raw.replace(",", " ").split()]
            morphology = extract_morphology_from_qpos(
                qpos=qpos,
                has_scene_prefix=self.scene_qpos.get(),
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load qpos", str(exc))
            return

        self._apply_morphology(morphology)
        messagebox.showinfo("Load qpos", "Morphology loaded from qpos.")

    def _load_open_keyframe(self) -> None:
        path = Path(self.base_hand_xml.get())
        try:
            root = ET.parse(path).getroot()
            keyframe = root.find("keyframe")
            if keyframe is None:
                raise ValueError("No <keyframe> found in base XML")

            open_key = None
            for key in keyframe.findall("key"):
                if key.get("name") == "open":
                    open_key = key
                    break
            if open_key is None:
                raise ValueError("No keyframe named 'open' found")

            qpos_raw = open_key.get("qpos", "")
            qpos = [float(v) for v in qpos_raw.split()]
            has_scene_prefix = len(qpos) >= 31
            morphology = extract_morphology_from_qpos(qpos=qpos, has_scene_prefix=has_scene_prefix)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load open keyframe", str(exc))
            return

        self._apply_morphology(morphology)
        self.scene_qpos.set(has_scene_prefix)
        messagebox.showinfo("Load open keyframe", f"Loaded morphology from {path.name} open keyframe.")

    def _start_viewer(self) -> None:
        if self.viewer_running:
            messagebox.showinfo("Live Viewer", "Viewer is already running.")
            return

        scene_path = Path(self.base_scene_xml.get())
        try:
            self.viewer_model = mujoco.MjModel.from_xml_path(str(scene_path))
            self.viewer_data = mujoco.MjData(self.viewer_model)
            apply_morphology_to_qpos(self.viewer_data.qpos, self._morphology(), has_scene_prefix=True)
            mujoco.mj_forward(self.viewer_model, self.viewer_data)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Live Viewer", str(exc))
            self.viewer_model = None
            self.viewer_data = None
            return

        self.viewer_running = True
        self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.viewer_thread.start()
        messagebox.showinfo("Live Viewer", "Viewer launched. Use Apply/Read buttons to sync morphology.")

    def _viewer_loop(self) -> None:
        if self.viewer_model is None or self.viewer_data is None:
            return

        with mujoco.viewer.launch_passive(self.viewer_model, self.viewer_data) as viewer:
            while self.viewer_running and viewer.is_running():
                with self.viewer_lock:
                    viewer.sync()
                time.sleep(0.01)

        self.viewer_running = False

    def _apply_to_viewer(self) -> None:
        if not self.viewer_running or self.viewer_model is None or self.viewer_data is None:
            messagebox.showerror("Live Viewer", "Viewer is not running.")
            return

        try:
            with self.viewer_lock:
                apply_morphology_to_qpos(self.viewer_data.qpos, self._morphology(), has_scene_prefix=True)
                mujoco.mj_forward(self.viewer_model, self.viewer_data)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Live Viewer", str(exc))
            return

    def _read_from_viewer(self) -> None:
        if not self.viewer_running or self.viewer_data is None:
            messagebox.showerror("Live Viewer", "Viewer is not running.")
            return

        try:
            with self.viewer_lock:
                morph = extract_morphology_from_qpos(
                    qpos=[float(v) for v in self.viewer_data.qpos],
                    has_scene_prefix=True,
                )
            self._apply_morphology(morph)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Live Viewer", str(exc))
            return

    def _stop_viewer(self) -> None:
        self.viewer_running = False
        self.viewer_model = None
        self.viewer_data = None

    def _on_close(self) -> None:
        self._stop_viewer()
        self.root.destroy()

    def _save(self) -> None:
        try:
            hand_base = Path(self.base_hand_xml.get())
            scene_base = Path(self.base_scene_xml.get())
            out_dir = Path(self.output_dir.get())
            morphology = self._morphology()
            hand_out, scene_out = create_rigid_hand_and_scene_xmls(
                base_hand_xml_path=hand_base,
                base_scene_xml_path=scene_base,
                morphology=morphology,
                output_dir=out_dir,
                hand_prefix=self.hand_prefix.get().strip() or "hand",
                scene_prefix=self.scene_prefix.get().strip() or "scene",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save Morphology XML", str(exc))
            return

        messagebox.showinfo(
            "Save Morphology XML",
            f"Saved hand XML:\n{hand_out}\n\nSaved scene XML:\n{scene_out}",
        )

    def _reset(self) -> None:
        for finger in FINGER_ORDER:
            self.triples[finger].x.set(0.0)
            self.triples[finger].y.set(0.0)
            self.triples[finger].length.set(0.0)


def main() -> None:
    root = tk.Tk()
    MorphologyEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
