"""
YOLO Boss Label Detector
=========================
Wraps a trained YOLOv8n model to detect the "[Lv.6000] Velik" name tag
on the current screen frame.

Used by ExplorationManager to know when the boss is visible so it can
trigger G BossLock and walk toward it - without any camera rotation.

Model path: models/boss_detector.pt
  Trained with train_yolo.py after labeling data from collect_yolo_data.py.

Returns: list of Detection(x_center, y_center, width, height, confidence)
  All coordinates are in pixel space of the original frame.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass

import numpy as np

MODEL_PATH = "models/boss_detector.pt"
CONF_THRESHOLD = 0.45   # detections below this are discarded
IOU_THRESHOLD  = 0.45   # NMS overlap threshold


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


class BossLabelDetector:
    """
    Detect the Velik boss label using a trained YOLOv8n model.

    Usage:
        detector = BossLabelDetector()
        if detector.available:
            dets = detector.run(frame_bgr)
            if dets:
                best = dets[0]           # highest confidence detection
                print(best.center_x)     # pixel x of label center
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._model = None
        self.available = False
        self._load(model_path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            print(f"[YOLO] No model at {path} - run train_yolo.py after labeling data.")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            # Warm-up inference
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            self.available = True
            print(f"[YOLO] Boss label detector loaded from {path}")
        except Exception as exc:
            print(f"[YOLO] Failed to load model: {exc}")

    def run(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Run inference on a BGR frame.
        Returns list of Detection objects sorted by confidence (highest first).
        Returns [] if model not loaded or no detections.
        """
        if not self.available or self._model is None:
            return []

        try:
            results = self._model(
                frame_bgr,
                verbose=False,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
            )
            dets = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append(Detection(x1, y1, x2, y2, conf))
            dets.sort(key=lambda d: d.confidence, reverse=True)
            return dets
        except Exception as exc:
            print(f"[YOLO] Inference error: {exc}")
            return []


# -- Test / calibrate ----------------------------------------------------------
if __name__ == "__main__":
    import mss
    import cv2
    import sys

    print("YOLO Live Detection Test - press Esc in the preview window to quit")

    detector = BossLabelDetector()
    if not detector.available:
        print("Model not available. Train it first with train_yolo.py")
        sys.exit(1)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            frame = np.ascontiguousarray(np.array(sct.grab(monitor))[:, :, :3])
            dets  = detector.run(frame)

            for d in dets:
                cv2.rectangle(frame,
                              (int(d.x1), int(d.y1)),
                              (int(d.x2), int(d.y2)),
                              (0, 255, 0), 2)
                label = f"Velik {d.confidence:.2f}"
                cv2.putText(frame, label, (int(d.x1), int(d.y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Downscale for display
            h, w = frame.shape[:2]
            disp = cv2.resize(frame, (w // 2, h // 2))
            cv2.imshow("YOLO Boss Detection", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
