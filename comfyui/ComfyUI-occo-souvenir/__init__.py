"""occo souvenir nodes for ComfyUI.

Install: make this folder visible in ComfyUI/custom_nodes (junction or copy, see ../README.md).
The nodes import the `souvenir` package from ../../src (resolved through the junction), or from
OCCO_SOUVENIR_SRC, or from a `pip install -e` of the project.
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for candidate in (os.environ.get("OCCO_SOUVENIR_SRC"), str(_here.parent.parent / "src")):
    if candidate and Path(candidate, "souvenir").is_dir() and candidate not in sys.path:
        sys.path.insert(0, candidate)
        break

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
