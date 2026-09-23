"""Image measurements and motion predictions remain separate throughout."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Config:
    gate: float = 5.991
    measurement_std: float = 4.0  # pixels; empirical tuning required
    acceleration_std: float = 800.0  # pixels / second squared
    deceleration_rate: float = 2.0  # velocity decay rate, per second
    initial_velocity_std: float = 200.0
    coast_seconds: float = 0.5
    size_ratio_limit: float = 4.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.size_ratio_limit <= 1:
            raise ValueError("size_ratio_limit must exceed 1")


class MotionFilter:
    def __init__(self, center, config):
        self.config = config
        self.x = np.array([*center, 0.0, 0.0], dtype=float)
        self.P = np.diag([config.measurement_std**2] * 2
                         + [config.initial_velocity_std**2] * 2)
        self.H = np.eye(2, 4)
        self.R = np.eye(2) * config.measurement_std**2

    def predict(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Frame timestamps must increase")
        rate = self.config.deceleration_rate
        decay = math.exp(-rate * dt)
        travel = -math.expm1(-rate * dt) / rate
        F = np.eye(4)
        F[0, 2] = F[1, 3] = travel
        F[2, 2] = F[3, 3] = decay
        # Response to an unknown constant acceleration over this time step.
        displacement_noise = (dt - travel) / rate
        G = np.array([[displacement_noise, 0], [0, displacement_noise],
                      [travel, 0], [0, travel]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.config.acceleration_std**2 * G @ G.T
        return self.x[:2].copy()

    def innovation(self, center):
        residual = np.asarray(center) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return residual, S

    def distance_squared(self, center):
        residual, S = self.innovation(center)
        return float(residual @ np.linalg.solve(S, residual))

    def correct(self, center):
        residual, S = self.innovation(center)
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x += K @ residual
        A = np.eye(4) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2


def validate_box(box, shape):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Bounding box requires four finite numbers: x y width height")
    x, y, w, h = [int(round(v)) for v in box]
    height, width = shape[:2]
    if w < 2 or h < 2 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("Bounding box must lie inside the frame with width/height >= 2")
    return x, y, w, h


def center_of(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2
