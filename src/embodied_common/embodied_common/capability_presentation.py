"""Optional public display metadata, independent of execution identifiers."""

import re
from collections.abc import Mapping
from typing import Any


def presentation_errors(metadata: Mapping[str, Any]) -> list[tuple[str, str]]:
    errors = []
    if "display_name" in metadata:
        name = metadata["display_name"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 120
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
        ):
            errors.append(
                ("display_name", "display_name must be non-empty text of at most 120 characters without controls")
            )
    if "tags" in metadata:
        tags = metadata["tags"]
        if (
            not isinstance(tags, list | tuple)
            or len(tags) > 8
            or any(not isinstance(tag, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", tag) for tag in tags)
            or len(set(tags)) != len(tags)
        ):
            errors.append(("tags", "tags must contain at most 8 unique lowercase identifiers of at most 32 characters"))
    return errors
