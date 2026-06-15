"""GUI tool to manually annotate gripper state (open/closed) for each NYU ROT episode.

Loads pkl files, shows image frames, lets user toggle open/closed per episode,
saves result to gripper_states.json.
"""

import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

EXPERT_DEMOS_ZIP = Path.home() / "Downloads" / "osfstorage-archive.zip"
EXTRACT_DIR = Path("/tmp/nyu_rot_extract")
OUTPUT_FILE = Path(__file__).parent / "gripper_states.json"

TASK_KEYS = [
    "RobotEraseBoard-v1/expert_demos",
    "RobotHangHanger-v1/expert_demos",
    "RobotReach-v1/expert_demos",
    "RobotDoorClose-v1/expert_demos",
    "RobotCupStacking-v1/expert_demos",
    "RobotTurnKnob-v1/expert_demos",
    "RobotInsertPeg-v1/expert_demos_easy",
    "RobotInsertPeg-v1/expert_demos_medium",
    "RobotInsertPeg-v1/expert_demos_hard",
    "RobotButtonPress-v1/expert_demos",
    "RobotHangBag-v1/expert_demos",
    "RobotBoxOpen-v1/expert_demos",
    "RobotPour-v1/expert_demos",
    "RobotHangMug-v1/expert_demos",
]

# Defaults from import script
DEFAULTS = {
    "RobotEraseBoard-v1/expert_demos": True,
    "RobotHangHanger-v1/expert_demos": True,
    "RobotReach-v1/expert_demos": False,
    "RobotDoorClose-v1/expert_demos": True,
    "RobotCupStacking-v1/expert_demos": True,
    "RobotTurnKnob-v1/expert_demos": True,
    "RobotInsertPeg-v1/expert_demos_easy": True,
    "RobotInsertPeg-v1/expert_demos_medium": True,
    "RobotInsertPeg-v1/expert_demos_hard": True,
    "RobotButtonPress-v1/expert_demos": True,
    "RobotHangBag-v1/expert_demos": True,
    "RobotBoxOpen-v1/expert_demos": False,
    "RobotPour-v1/expert_demos": True,
    "RobotHangMug-v1/expert_demos": True,
}


def load_episodes() -> dict[str, np.ndarray]:
    """Load images for all episodes from pkl files."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    robotgym_dir = EXTRACT_DIR / "expert_demos" / "robotgym"
    if not robotgym_dir.exists():
        with zipfile.ZipFile(EXPERT_DEMOS_ZIP) as zf:
            names = zf.namelist()
            if "expert_demos.zip" in names:
                print("Extracting expert_demos.zip...")
                zf.extract("expert_demos.zip", EXTRACT_DIR)
                with zipfile.ZipFile(EXTRACT_DIR / "expert_demos.zip") as inner:
                    print("Extracting robotgym pkl files...")
                    for name in inner.namelist():
                        if name.startswith("expert_demos/robotgym/"):
                            inner.extract(name, EXTRACT_DIR)
            elif any(name.startswith("expert_demos/robotgym/") for name in names):
                print("Extracting robotgym pkl files...")
                for name in names:
                    if name.startswith("expert_demos/robotgym/"):
                        zf.extract(name, EXTRACT_DIR)
            else:
                raise FileNotFoundError(
                    f"Could not find expert_demos/robotgym/ in {EXPERT_DEMOS_ZIP}"
                )

    episodes = {}
    for task_dir in sorted(robotgym_dir.iterdir()):
        for pkl_path in sorted(task_dir.iterdir()):
            key = f"{task_dir.name}/{pkl_path.stem}"
            if key not in TASK_KEYS:
                continue
            with open(pkl_path, "rb") as f:
                images, *_ = pickle.load(f)
            # images: (n_eps, n_steps, 3, H, W) — take episode 0, CHW→HWC
            episodes[key] = np.transpose(images[0], (0, 2, 3, 1))  # (steps, H, W, 3)
    return episodes


def ndarray_to_pixmap(frame: np.ndarray, scale: int = 4) -> QPixmap:
    h, w, c = frame.shape
    img = QImage(frame.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img).scaled(
        w * scale, h * scale, Qt.AspectRatioMode.KeepAspectRatio
    )


class GripperAnnotator(QMainWindow):
    def __init__(self, episodes: dict[str, np.ndarray], saved: dict[str, bool]) -> None:
        super().__init__()
        self.episodes = episodes
        self.keys = TASK_KEYS
        self.states: dict[str, bool] = {
            k: saved.get(k, DEFAULTS.get(k, False)) for k in self.keys
        }
        self.current = 0
        self.playing = False
        self.frame_idx = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._next_frame)

        self.setWindowTitle("NYU ROT Gripper Annotator")
        self._build_ui()
        self._load_episode(0)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Episode label
        self.ep_label = QLabel()
        self.ep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ep_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        root.addWidget(self.ep_label)

        # Progress
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.progress_label)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.image_label)

        # Frame slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        root.addWidget(self.slider)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.frame_label)

        # Play/pause
        play_row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        play_row.addWidget(self.play_btn)
        root.addLayout(play_row)

        # Gripper toggle
        grip_row = QHBoxLayout()
        self.btn_group = QButtonGroup()
        self.radio_closed = QRadioButton("Closed (gripper holding object)")
        self.radio_open = QRadioButton("Open (gripper not holding)")
        self.btn_group.addButton(self.radio_closed)
        self.btn_group.addButton(self.radio_open)
        self.radio_closed.toggled.connect(self._on_gripper_changed)
        grip_row.addWidget(self.radio_closed)
        grip_row.addWidget(self.radio_open)
        root.addLayout(grip_row)

        # Nav buttons
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._next)
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        self.save_btn.clicked.connect(self._save)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.save_btn)
        nav_row.addWidget(self.next_btn)
        root.addLayout(nav_row)

        # Saved indicator
        self.saved_label = QLabel()
        self.saved_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.saved_label)

    def _load_episode(self, idx: int) -> None:
        self._timer.stop()
        self.playing = False
        self.play_btn.setText("▶ Play")
        self.current = idx
        key = self.keys[idx]
        self.frames = self.episodes[key]
        self.frame_idx = 0
        n = len(self.frames)

        self.ep_label.setText(key)
        self.progress_label.setText(f"Episode {idx + 1} / {len(self.keys)}")
        self.slider.setMaximum(n - 1)
        self.slider.setValue(0)

        closed = self.states[key]
        self.radio_closed.blockSignals(True)
        self.radio_open.blockSignals(True)
        (self.radio_closed if closed else self.radio_open).setChecked(True)
        self.radio_closed.blockSignals(False)
        self.radio_open.blockSignals(False)

        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < len(self.keys) - 1)
        self._show_frame(0)

    def _show_frame(self, idx: int) -> None:
        self.frame_idx = idx
        self.image_label.setPixmap(ndarray_to_pixmap(self.frames[idx]))
        self.frame_label.setText(f"Frame {idx + 1} / {len(self.frames)}")

    def _on_slider(self, val: int) -> None:
        self._show_frame(val)

    def _toggle_play(self) -> None:
        if self.playing:
            self._timer.stop()
            self.playing = False
            self.play_btn.setText("▶ Play")
        else:
            self.playing = True
            self.play_btn.setText("⏸ Pause")
            self._timer.start(200)  # 5 Hz

    def _next_frame(self) -> None:
        next_idx = (self.frame_idx + 1) % len(self.frames)
        self.slider.setValue(next_idx)

    def _on_gripper_changed(self) -> None:
        self.states[self.keys[self.current]] = self.radio_closed.isChecked()

    def _prev(self) -> None:
        if self.current > 0:
            self._load_episode(self.current - 1)

    def _next(self) -> None:
        if self.current < len(self.keys) - 1:
            self._load_episode(self.current + 1)

    def _save(self) -> None:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(self.states, f, indent=2)
        self.saved_label.setText(f"Saved to {OUTPUT_FILE}")
        self.saved_label.setStyleSheet("color: green;")


def main() -> None:
    print("Loading episodes...")
    episodes = load_episodes()

    saved = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            saved = json.load(f)
        print(f"Loaded existing annotations from {OUTPUT_FILE}")

    app = QApplication(sys.argv)
    window = GripperAnnotator(episodes, saved)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
