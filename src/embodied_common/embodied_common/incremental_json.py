"""Read a top-level field out of a JSON document that is still arriving.

A streamed planner answer is a partial JSON string that grows one chunk at a
time. The short field the robot can already speak ("user_message") closes long
before the long field it cannot ("steps"), so the wait for the first sound is
the wait for that one field rather than for the whole document.

Scanning has to be structural. A substring search for the key would also match
the same name appearing inside another string value, and a count of quotes
would miscount escaped ones -- either way the robot would speak a fragment, or
speak something that is not the field at all.
"""

from __future__ import annotations

import json

__all__ = ["first_complete_string_field"]


def _string_end(buffer: str, start: int) -> int | None:
    """Index of the closing quote of the JSON string starting at ``start``.

    ``None`` means the string has not finished arriving.
    """
    index = start + 1
    length = len(buffer)
    while index < length:
        char = buffer[index]
        if char == "\\":
            # Skip the escape and whatever it escapes. A trailing backslash
            # means the escape itself is still arriving.
            index += 2
            continue
        if char == '"':
            return index
        index += 1
    return None


def first_complete_string_field(buffer: str, field: str) -> str | None:
    """Return a top-level string field's value once it has fully arrived.

    ``None`` while the field is absent, still arriving, or not a string. Only
    depth-1 keys are considered, so a nested object carrying the same key -- or
    another value whose text happens to contain it -- cannot be mistaken for
    it. The returned value is JSON-decoded, so escapes are resolved.
    """
    index = 0
    depth = 0
    length = len(buffer)
    while index < length:
        char = buffer[index]
        if char == '"':
            end = _string_end(buffer, index)
            if end is None:
                return None
            literal = buffer[index : end + 1]
            index = end + 1
            if depth != 1:
                continue
            probe = index
            while probe < length and buffer[probe].isspace():
                probe += 1
            if probe >= length or buffer[probe] != ":":
                # A value, not a key. Keep scanning from after it.
                continue
            try:
                key = json.loads(literal)
            except ValueError:
                return None
            if key != field:
                index = probe + 1
                continue
            value_start = probe + 1
            while value_start < length and buffer[value_start].isspace():
                value_start += 1
            if value_start >= length:
                return None
            if buffer[value_start] != '"':
                # Present but not a string; nothing speakable here.
                return None
            value_end = _string_end(buffer, value_start)
            if value_end is None:
                return None
            try:
                value = json.loads(buffer[value_start : value_end + 1])
            except ValueError:
                return None
            return value if isinstance(value, str) else None
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        index += 1
    return None
