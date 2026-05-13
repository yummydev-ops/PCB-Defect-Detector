#!/usr/bin/env python
"""
Heuristic severity scoring for PCB defect detections.

The score is normalized to the range 0.0 to 1.0 and is meant for downstream
reporting and prioritization, not as a replacement for expert inspection.
"""

from __future__ import annotations

from typing import Dict, Tuple


CLASS_SEVERITY_PRIORS: Dict[str, float] = {
    "missing_hole": 0.70,
    "mouse_bite": 0.55,
    "open_circuit": 1.00,
    "short": 1.00,
    "spur": 0.60,
    "spurious_copper": 0.75,
}

CLASS_WEIGHT = 0.50
AREA_WEIGHT = 0.30
CONFIDENCE_WEIGHT = 0.20
REFERENCE_AREA_RATIO = 0.01


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def compute_relative_area(xyxy: Tuple[float, float, float, float], image_width: int, image_height: int) -> float:
    x1, y1, x2, y2 = xyxy
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    image_area = max(float(image_width * image_height), 1.0)
    return (box_width * box_height) / image_area


def compute_area_score(relative_area: float, reference_area_ratio: float = REFERENCE_AREA_RATIO) -> float:
    if reference_area_ratio <= 0:
        raise ValueError("reference_area_ratio must be positive")
    return clamp(relative_area / reference_area_ratio)


def classify_severity_band(score: float) -> str:
    if score < 0.34:
        return "low"
    if score < 0.67:
        return "medium"
    return "high"


def score_detection(
    class_name: str,
    confidence: float,
    xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Dict[str, float | str]:
    class_prior = CLASS_SEVERITY_PRIORS.get(class_name, 0.60)
    relative_area = compute_relative_area(xyxy, image_width, image_height)
    area_score = compute_area_score(relative_area)
    confidence_score = clamp(confidence)
    severity_score = clamp(
        (CLASS_WEIGHT * class_prior)
        + (AREA_WEIGHT * area_score)
        + (CONFIDENCE_WEIGHT * confidence_score)
    )

    return {
        "class_prior": round(class_prior, 4),
        "relative_area": round(relative_area, 6),
        "area_score": round(area_score, 4),
        "confidence_score": round(confidence_score, 4),
        "severity_score": round(severity_score, 4),
        "severity_band": classify_severity_band(severity_score),
    }
