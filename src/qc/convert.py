"""Pascal-VOC -> YOLO label conversion (shared by the NEU-DET prep script and tests)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from qc.utils import bbox_voc_to_yolo


def voc_xml_to_yolo_lines(xml_text, class_to_idx, default_size=None):
    """Convert one Pascal-VOC annotation (XML text) to YOLO label lines.

    Objects whose class is not in ``class_to_idx`` are skipped. If the XML has no
    ``<size>``, pass ``default_size=(width, height)``.
    """
    root = ET.fromstring(xml_text)
    size = root.find("size")
    if size is not None and size.find("width") is not None:
        W, H = float(size.find("width").text), float(size.find("height").text)
    elif default_size is not None:
        W, H = float(default_size[0]), float(default_size[1])
    else:
        raise ValueError("annotation has no <size> and no default_size was given")

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_to_idx:
            continue
        b = obj.find("bndbox")
        cx, cy, w, h = bbox_voc_to_yolo(
            float(b.find("xmin").text), float(b.find("ymin").text),
            float(b.find("xmax").text), float(b.find("ymax").text), W, H,
        )
        lines.append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
