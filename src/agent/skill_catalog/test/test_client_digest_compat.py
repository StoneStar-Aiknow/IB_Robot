from __future__ import annotations

import embodied_common.catalog_digest as neutral
import skill_catalog.digest as legacy


def test_digest_compatibility_exports_share_implementation() -> None:
    assert legacy.to_canonical_json is neutral.to_canonical_json
    assert legacy.derive_registry_digest is neutral.derive_registry_digest
    assert legacy.derive_capability_digest is neutral.derive_capability_digest
    assert legacy.derive_provenance_digest is neutral.derive_provenance_digest


def test_canonical_json_bytes_are_identical() -> None:
    value = {"z": 1, "a": [2, 3.5, True, None, "中文"]}
    assert legacy.to_canonical_json(value) == neutral.to_canonical_json(value)
