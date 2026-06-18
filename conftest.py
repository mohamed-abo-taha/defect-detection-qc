"""Make the ``src`` layout importable as ``qc`` without an editable install."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
