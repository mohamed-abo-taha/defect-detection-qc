"""cv-quality-control: computer-vision pipeline for manufacturing defect QC.

Pipeline stages (see README): data audit -> baseline -> imbalance handling ->
evaluation -> explainability (Grad-CAM) -> ONNX export. Every reported number
is traceable to a logged run; no fabricated metrics.
"""

__version__ = "0.0.1"
