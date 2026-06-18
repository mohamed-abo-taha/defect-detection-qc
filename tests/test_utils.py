"""Tests for class-weighting and the VOC->YOLO box conversion."""

import torch

from qc.utils import bbox_voc_to_yolo, class_weights_from_counts


def test_balanced_counts_give_unit_weights():
    w = class_weights_from_counts([100, 100, 100], "inverse")
    assert torch.allclose(w, torch.ones(3), atol=1e-6)


def test_inverse_weighting_favours_the_rare_class():
    w = class_weights_from_counts([240, 80, 80, 60, 24], "inverse")
    assert w[-1] > w[0]  # the rarest class (24) gets the largest weight


def test_inverse_sqrt_is_gentler_than_inverse():
    inv = class_weights_from_counts([240, 24], "inverse")
    sqrt = class_weights_from_counts([240, 24], "inverse_sqrt")
    assert sqrt[-1] < inv[-1]


def test_voc_to_yolo_centre_box():
    cx, cy, w, h = bbox_voc_to_yolo(50, 50, 150, 150, 200, 200)
    assert abs(cx - 0.5) < 1e-9 and abs(cy - 0.5) < 1e-9
    assert abs(w - 0.5) < 1e-9 and abs(h - 0.5) < 1e-9


def test_voc_to_yolo_full_frame_box():
    cx, cy, w, h = bbox_voc_to_yolo(0, 0, 200, 100, 200, 100)
    assert (cx, cy, w, h) == (0.5, 0.5, 1.0, 1.0)
