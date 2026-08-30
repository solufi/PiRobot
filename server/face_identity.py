"""Optional local face-profile storage and OpenCV SFace matching."""
from __future__ import annotations

import json
import os
import pathlib
import threading

import cv2
import numpy as np


class FaceIdentity:
    def __init__(self) -> None:
        self.model_path = os.environ.get(
            "PICAR_SFACE_MODEL",
            "/home/solufi/models/face_recognition_sface_2021dec.onnx",
        )
        self.profiles_path = pathlib.Path(os.environ.get(
            "PICAR_FACE_PROFILES",
            "/home/solufi/models/face_profiles.json",
        ))
        self.threshold = float(os.environ.get("PICAR_FACE_THRESHOLD", "0.363"))
        self.lock = threading.RLock()
        self.recognizer = None
        if os.path.isfile(self.model_path) and hasattr(cv2, "FaceRecognizerSF_create"):
            try:
                self.recognizer = cv2.FaceRecognizerSF_create(self.model_path, "")
            except Exception:
                self.recognizer = None

    @property
    def available(self) -> bool:
        return self.recognizer is not None

    def _read(self) -> dict[str, list[float]]:
        try:
            data = json.loads(self.profiles_path.read_text())
            return {
                str(name): [float(value) for value in feature]
                for name, feature in data.items()
                if isinstance(name, str) and isinstance(feature, list)
            }
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def profiles(self) -> list[str]:
        with self.lock:
            return sorted(self._read())

    def _write(self, profiles: dict[str, list[float]]) -> None:
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.profiles_path.with_suffix(".tmp")
        temp.write_text(json.dumps(profiles, separators=(",", ":")))
        temp.replace(self.profiles_path)

    def feature(self, frame_rgb: np.ndarray, face: tuple) -> list[float] | None:
        if self.recognizer is None or len(face) < 15:
            return None
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        aligned = self.recognizer.alignCrop(bgr, np.asarray(face[:15], dtype=np.float32))
        return self.recognizer.feature(aligned).flatten().astype(float).tolist()

    def enroll(self, name: str, feature: list[float]) -> None:
        name = name.strip()
        if not name or len(name) > 40:
            raise ValueError("nom invalide")
        with self.lock:
            profiles = self._read()
            profiles[name] = feature
            self._write(profiles)

    def remove(self, name: str) -> bool:
        with self.lock:
            profiles = self._read()
            removed = profiles.pop(name, None) is not None
            if removed:
                self._write(profiles)
            return removed

    def identify(self, feature: list[float]) -> str | None:
        if self.recognizer is None:
            return None
        probe = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        best_name = None
        best_score = self.threshold
        with self.lock:
            for name, values in self._read().items():
                candidate = np.asarray(values, dtype=np.float32).reshape(1, -1)
                score = float(self.recognizer.match(
                    probe, candidate, cv2.FaceRecognizerSF_FR_COSINE
                ))
                if score >= best_score:
                    best_name = name
                    best_score = score
        return best_name
