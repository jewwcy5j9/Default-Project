import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from classify_states import classify_state


def test_threshold_argument_controls_assignment():
    assert classify_state(2.8, 4.0, 5.0, threshold=2.5) == "unclassified"
    assert classify_state(2.8, 4.0, 5.0, threshold=3.0) == "active"


def test_threshold_is_strict():
    assert classify_state(3.0, 4.0, 5.0, threshold=3.0) == "unclassified"


def test_argmin_resolves_multiple_matches():
    assert classify_state(2.0, 1.5, 4.0, threshold=3.0) == "I1"
