"""Model factory and the Grad-CAM target-layer helper.

We use timm so any of its backbones (resnet18, efficientnet, convnext, ...) work by
name. ``pretrained=True`` (ImageNet) is strongly recommended for real datasets —
defect sets are small, and transfer learning is most of the win. It is made optional
only so the synthetic smoke test can run fully offline.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(num_classes: int, arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last Conv2d module — the conventional Grad-CAM target.

    Searching for the final Conv2d keeps this backbone-agnostic instead of
    hard-coding e.g. ``model.layer4[-1]`` for ResNet.
    """
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("no Conv2d layer found to attach Grad-CAM to")
    return last
