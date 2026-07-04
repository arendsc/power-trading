from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def finite_1d_array(
    values: ArrayLike,
    *,
    name: str,
    positive: bool = False,
    unit_interval: bool = False,
) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive values")
    if unit_interval and np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def finite_2d_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a 2D array with at least one row")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def finite_float(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def non_negative_float(value: float, *, name: str) -> float:
    value = finite_float(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def positive_float(value: float, *, name: str) -> float:
    value = finite_float(value, name=name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def unit_interval_float(value: float, *, name: str) -> float:
    value = finite_float(value, name=name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)
