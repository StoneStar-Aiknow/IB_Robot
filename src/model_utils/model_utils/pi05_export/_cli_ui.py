# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Shared console UI helpers for the PI05 export toolchain.

This module centralises the *look and feel* of every PI05 export script so the
whole pipeline speaks one visual language:

* :func:`setup_logging` — one logging format / level setup, used by every
  ``main()`` instead of each script hand-rolling its own handler juggling.
* :class:`Stage` — a context manager that prints a clear ``▶ start`` / ``✓ done``
  / ``✗ failed`` banner with elapsed time, so the user always sees progress and
  never wonders whether the tool is stuck.
* :func:`print_summary` — a compact, structured result block printed at the end
  of a run (paths produced, pass/fail, next step).

Keeping this in one place means a single edit changes the style everywhere.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER = logging.getLogger("pi05_export")

# Unified log line format for the whole toolchain.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with the toolchain's unified format.

    Idempotent: safe to call from every script's ``main()``. Existing handlers
    are re-levelled and re-formatted rather than duplicated, so a parent
    orchestrator and the script it imports do not stack handlers.
    """
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    if not stream_handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(lvl)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for h in stream_handlers:
            h.setLevel(lvl)
            h.setFormatter(formatter)


@contextmanager
def Stage(name: str, *, index: int | None = None, total: int | None = None) -> Iterator[None]:
    """Context manager that brackets a pipeline stage with start/end banners.

    Prints an immediate ``▶`` line on entry (so the user sees work begin right
    away), and a ``✓`` line with elapsed time on success or a ``✗`` line on
    failure. The exception is re-raised so callers can still stop the pipeline.
    """
    prefix = f"[{index}/{total}] " if index is not None and total is not None else ""
    LOGGER.info("▶ %s%s …", prefix, name)
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - start
        LOGGER.error("✗ %s%s failed after %.1fs", prefix, name, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        LOGGER.info("✓ %s%s done (%.1fs)", prefix, name, elapsed)


def print_summary(title: str, rows: list[tuple[str, str]], *, status: str | None = None) -> None:
    """Print a compact, aligned result block.

    Args:
        title: Heading shown above the block.
        rows: ``(label, value)`` pairs, printed left-aligned.
        status: Optional final status line (e.g. ``"✅ PASS"``).
    """
    width = max((len(label) for label, _ in rows), default=0)
    bar = "─" * max(len(title), 40)
    LOGGER.info(bar)
    LOGGER.info("%s", title)
    LOGGER.info(bar)
    for label, value in rows:
        LOGGER.info("  %-*s : %s", width, label, value)
    if status is not None:
        LOGGER.info(bar)
        LOGGER.info("  %s", status)
    LOGGER.info(bar)
