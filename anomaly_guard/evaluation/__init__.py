"""Evaluation splits, metrics, and experiment runners."""

from anomaly_guard.evaluation.splits import (
    training_normal_records,
    validation_legitimate_records,
    validation_ordinary_records,
)

__all__ = [
    "training_normal_records",
    "validation_legitimate_records",
    "validation_ordinary_records",
]
