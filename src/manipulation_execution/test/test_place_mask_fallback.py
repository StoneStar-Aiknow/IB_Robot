"""The placement exclusion-mask loader's source-tree fallback (design D3).

``place/node.py`` sits one directory deeper than the flat layout the fallback
was written for. This test pins the corrected depth the way design D3 of the
package-layout change prescribes: force the ament lookup to fail, call the
real loader with a ``$(find robot_config)/...`` path, and assert the path it
resolved. The expectation is anchored on the package ``__init__.py``, which
does not move, so it is independent of the moved module's depth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import manipulation_execution
from manipulation_execution.place.containment import load_exclusion_mask

_WORKSPACE_ROOT = Path(manipulation_execution.__file__).resolve().parents[2]


def test_source_tree_fallback_resolves_the_workspace_robot_config(monkeypatch):
    import ament_index_python.packages

    # Force the ament lookup to fail so the source-tree fallback runs.
    def refusing_lookup(package):
        raise RuntimeError(f"forced failure for {package}")

    monkeypatch.setattr(ament_index_python.packages, "get_package_share_directory", refusing_lookup)
    relative = "config/place/nonexistent_mask.png"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_exclusion_mask(f"$(find robot_config)/{relative}", (4, 4))
    assert str(excinfo.value) == f"target exclusion mask not found: {_WORKSPACE_ROOT / 'robot_config' / relative}"
