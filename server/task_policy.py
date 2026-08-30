"""Validation rules for bounded robot task sequences."""
from __future__ import annotations


VALID_DIRECTIONS = {"forward", "backward", "left", "right", "stop"}


def validate_steps(steps: list[dict]) -> tuple[list[dict], int]:
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise ValueError("une tâche doit contenir 1 à 8 étapes")

    normalized: list[dict] = []
    total_ms = 0
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("étape invalide")
        action = step.get("action")
        if action not in {"drive", "wait"}:
            raise ValueError("action invalide")
        try:
            duration_ms = int(step.get("duration_ms", 600 if action == "drive" else 100))
        except (TypeError, ValueError):
            raise ValueError("durée invalide") from None
        if not 100 <= duration_ms <= 1500:
            raise ValueError("durée invalide")
        if action == "drive" and step.get("direction") not in VALID_DIRECTIONS:
            raise ValueError("direction invalide")
        total_ms += duration_ms
        if total_ms > 12000:
            raise ValueError("durée totale maximale dépassée")
        normalized.append({
            "action": action,
            "direction": step.get("direction"),
            "duration_ms": duration_ms,
        })
    return normalized, total_ms
