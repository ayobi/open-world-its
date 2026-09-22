"""Split-conformal helpers for one-sided open-world novelty detection."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def conformal_pvalues(
    calibration_novelty_scores: Sequence[float],
    query_novelty_scores: Sequence[float],
) -> np.ndarray:
    """Return conservative split-conformal p-values for high=more-novel scores.

    For query score ``s`` and ``n`` calibration knowns, this computes

        p = (1 + #{cal_i >= s}) / (n + 1)

    so ties are conservative.  The +1 finite-sample correction prevents zero
    p-values and is the usual inductive conformal anomaly-detection rule.
    """
    cal = np.asarray(calibration_novelty_scores, dtype=np.float64)
    qry = np.asarray(query_novelty_scores, dtype=np.float64)
    if cal.ndim != 1 or qry.ndim != 1:
        raise ValueError("calibration and query scores must be one-dimensional")
    if cal.size == 0:
        raise ValueError("calibration scores must be non-empty")
    if not np.isfinite(cal).all() or not np.isfinite(qry).all():
        raise ValueError("scores must be finite")

    # searchsorted avoids materializing n_cal x n_query comparisons.  side='left'
    # makes equal calibration scores count as >= the query, preserving conservative
    # tie handling.
    ordered = np.sort(cal)
    n_lt = np.searchsorted(ordered, qry, side="left")
    n_ge = cal.size - n_lt
    return (1.0 + n_ge.astype(np.float64)) / (cal.size + 1.0)


def novelty_flags(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """Return p <= alpha novelty decisions."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    p = np.asarray(pvalues, dtype=np.float64)
    return p <= float(alpha)
