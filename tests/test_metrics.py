"""Test the QC good-vs-defect framing (miss rate vs false-alarm rate)."""

import numpy as np

from qc.evaluate import qc_good_vs_defect


def test_miss_and_false_alarm_counts():
    names = ["good", "scratch", "dent"]
    #            good good  defect defect defect
    y = np.array([0,   0,    1,     2,     1])
    p = np.array([0,   1,    0,     2,     1])  # good#2 -> false alarm; defect#1 -> missed
    r = qc_good_vs_defect(y, p, names)
    assert r["tp"] == 2 and r["fn"] == 1   # 3 defects: 2 caught, 1 missed
    assert r["fp"] == 1 and r["tn"] == 1   # 2 good: 1 false alarm, 1 correct
    assert abs(r["miss_rate_FN"] - 1 / 3) < 1e-9
    assert abs(r["false_alarm_rate_FP"] - 1 / 2) < 1e-9


def test_returns_none_without_good_class():
    assert qc_good_vs_defect(np.array([0, 1]), np.array([0, 1]), ["scratch", "dent"]) is None
