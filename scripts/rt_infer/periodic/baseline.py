"""Externally supplied model map; use the active project Python environment."""

import json
import os
from pathlib import Path

OM = json.loads(Path(os.environ["RTINFER_MODELS"]).read_text())
if set(OM) != {"pear", "yolox", "te", "fm", "vision", "prefill", "decode"}:
    raise ValueError("Expected exactly seven named OM models")
for name, path in OM.items():
    if not Path(path).is_absolute() or not Path(path).is_file():
        raise ValueError(f"Missing absolute OM path for {name}: {path}")
