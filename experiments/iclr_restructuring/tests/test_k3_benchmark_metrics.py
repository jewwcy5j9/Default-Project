import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

from k3_benchmark import metrics


def test_direction_uses_active_state_and_excludes_target_ties():
    wt = np.array([0.72, 0.07, 0.21])
    targets = {
        "same": np.array([0.90, 0.00, 0.10]),
        "wrong": np.array([0.20, 0.10, 0.70]),
        "tie": np.array([0.73, 0.27, 0.00]),
    }
    preds = {
        "same": np.array([0.80, 0.15, 0.05]),
        "wrong": np.array([0.80, 0.00, 0.20]),
        "tie": np.array([0.10, 0.80, 0.10]),
    }

    result = metrics(preds, targets, wt)

    assert result["direction"] == "1/2"
    assert result["direction_detail"] == {
        "same": "OK",
        "wrong": "WRONG",
        "tie": "TIE",
    }


def test_direction_threshold_includes_exact_boundary():
    wt = np.array([0.72, 0.07, 0.21])
    targets = {"boundary": np.array([0.77, 0.03, 0.20])}
    preds = {"boundary": np.array([0.80, 0.10, 0.10])}

    assert metrics(preds, targets, wt)["direction"] == "1/1"
