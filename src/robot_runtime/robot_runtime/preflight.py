"""Calibration preflight of a physical runtime profile against the calibration on this machine.

Checks the runtime-layer steps a physical ``runtime.launch.py`` performs before
it creates any launch action: ``render_robot_description`` (which requires the
calibration file), the model projection, and ``build_description`` (model
validation and the public interface description). The report lists, per
position joint, the calibrated range, the effective limit, which authority set
each end, the profile home pose and its distance to the nearest limit, and
warns when that distance is small.

Two entry points share the check:

- ``runtime_preflight`` (``main``), offline: runs the steps itself, touches no
  serial port, driver or controller, prints the report and exits 1 when a step
  fails.
- ``checked_physical_description``, inside a physical launch: wraps the
  launch's own render and ``build_description`` calls. A pass logs one summary
  line (the full table when a margin is small); a failure logs a FAIL report
  (with the table when the model can still be projected) and re-raises the
  original exception, so the launch still fails before any hardware node
  starts, on the same conditions as without the wrapper.

Arm joints are limited by the calibrated range intersected with the URDF; a
physical gripper keeps its calibrated range (the URDF only carries the
normalized convention). Base wheels are velocity joints and are not listed.

Not checked: the provider's own reading of the calibration file
(for example the SO-101 SDK also requires an in-range ``homing_offset`` and
validates ``drive_mode`` and motor ids when present) and provider launch steps
after ``build_description``; that the file belongs to the attached arm; that
the arm is safe to move. The offline command also uses the profile as is.

Usage::

    ros2 run robot_runtime runtime_preflight --profile <path>
    ros2 run robot_runtime runtime_preflight --package so101_robot --profile so101_single_arm
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from robot_runtime.joint_conversion import NORM_MODE_RANGE

DEFAULT_MIN_MARGIN = 0.05
"""Radians. A home pose closer than this to a limit passes but is reported as a warning."""

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class JointRow:
    name: str
    calibrated: tuple[float, float]
    effective: tuple[float, float]
    home: float | None

    def source(self, side: int) -> str:
        """Which authority set the effective lower (``side=0``) or upper (``side=1``) limit."""
        return "calib" if self.effective[side] == self.calibrated[side] else "urdf"

    @property
    def bound_by(self) -> str:
        return f"{self.source(0)}/{self.source(1)}"

    @property
    def nearest_side(self) -> int | None:
        if self.home is None:
            return None
        low, high = self.effective
        return 0 if self.home - low <= high - self.home else 1

    @property
    def margin(self) -> float | None:
        """Signed distance from home to the nearest effective limit; negative means outside."""
        if self.home is None:
            return None
        low, high = self.effective
        return min(self.home - low, high - self.home)


@dataclass(frozen=True)
class PreflightReport:
    profile_path: Path
    calibration_file: str
    rows: list[JointRow]
    error: str | None

    def warnings(self, min_margin: float) -> list[str]:
        return [
            f"joint {row.name}: home {row.home:+.3f} rad is {row.margin:.3f} rad from its "
            f"{('lower', 'upper')[row.nearest_side]} limit, set by {row.source(row.nearest_side)} "
            f"(< {min_margin:.3f})"
            for row in self.rows
            if row.margin is not None and 0.0 <= row.margin < min_margin
        ]


# Errors a malformed profile or calibration raises inside the render, the
# projection or build_description; the physical launch fails on the same ones.
# ArithmeticError covers a numeric field too large for a float (OverflowError)
# in a place that does not name it yet; the known ones raise ValueError.
_CHECK_ERRORS = (OSError, ValueError, LookupError, TypeError, AttributeError, RuntimeError, ArithmeticError)


def _check(
    profile: dict[str, Any],
    profile_path: Path,
    render: Callable[[], str],
    describe: Callable[[str], dict],
) -> tuple[PreflightReport, str | None, dict | None, BaseException | None]:
    """Run ``render`` then ``describe`` and build the report around them.

    Returns the report, the rendered robot description, the interface
    description and the exception a step raised (``None`` on success). On
    success the rows come from ``description["model"]``, so nothing is
    projected twice; on a failure after the render, one unvalidated projection
    recovers the rows for the report.
    """
    from robot_runtime.model_metadata import build_model_metadata
    from robot_runtime.path_subst import resolve_path

    calibration_file = ""
    robot_description = None
    try:
        hardware = profile.get("hardware") or {}
        calibration_file = resolve_path(hardware["calib_file"]) if hardware.get("calib_file") else ""
        robot_description = render()
        description = describe(robot_description)
    except _CHECK_ERRORS as exc:
        rows: list[JointRow] = []
        if robot_description is not None:
            # Best effort, for the report only: whatever this raises must not
            # replace the launch's own error, so it is dropped with the rows.
            try:
                rows = _rows(
                    build_model_metadata(
                        profile, use_sim=False, sim_platform=None, robot_description=robot_description, validate=False
                    )
                )
            except Exception:
                rows = []
        return PreflightReport(profile_path, calibration_file, rows, _message(exc)), robot_description, None, exc
    rows = _rows(description["model"]) if "model" in description else []
    return PreflightReport(profile_path, calibration_file, rows, None), robot_description, description, None


def run_preflight(profile_path: Path) -> PreflightReport:
    """Run the pre-action runtime steps of a physical launch on ``profile_path`` and report.

    Raises only what ``load_profile`` raises for a profile file that cannot be
    loaded (``ProfileError``, ``OSError``, ``yaml.YAMLError``, and ``TypeError``
    or ``AttributeError`` for a section of the wrong type). Every later failure
    is returned in ``PreflightReport.error``, with the joint rows resolved
    before it.
    """
    from robot_runtime.interface_description import build_description
    from robot_runtime.launch_support import render_robot_description
    from robot_runtime.profile import load_profile

    profile = load_profile(profile_path)
    report, _, _, _ = _check(
        profile,
        profile_path,
        lambda: render_robot_description(profile, profile_path, False),
        lambda urdf: build_description(
            profile, use_sim=False, robot_description=urdf, sim_platform=None, profile_path=profile_path
        ),
    )
    return report


LaunchLog = Callable[[str, str], None]
"""``log(level, text)`` with ``level`` one of ``info``, ``warning``, ``error``."""


def _launch_log(level: str, text: str) -> None:
    import launch.logging

    getattr(launch.logging.get_logger("runtime_preflight"), level)(text)


def checked_physical_description(
    profile: dict[str, Any],
    profile_path: Path,
    render: Callable[[], str],
    describe: Callable[[str], dict],
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    log: LaunchLog | None = None,
) -> tuple[str, dict]:
    """Run a physical launch's render and ``build_description`` with the calibration report around them.

    ``render()`` returns the robot description; ``describe(robot_description)``
    returns the interface description. Both are the launch's own calls, made
    once. Returns ``(robot_description, description)``.

    A pass logs one summary line; when a home pose is closer than
    ``min_margin`` to a limit it logs the full table at warning level and
    continues. A failure logs a FAIL report at error level (with the table
    when the model can still be projected) and re-raises the original
    exception unchanged. Call it on the physical path only: a
    simulated selection reads no calibration.
    """
    emit = log or _launch_log
    report, robot_description, description, exc = _check(profile, profile_path, render, describe)
    if exc is not None:
        emit(
            "error",
            "calibration preflight failed before any hardware node starts:\n" + format_report(report, min_margin),
        )
        raise exc
    if report.warnings(min_margin):
        emit("warning", "calibration preflight passed with small margins:\n" + format_report(report, min_margin))
    else:
        emit("info", summary_line(report))
    return robot_description, description


def summary_line(report: PreflightReport) -> str:
    """One line for a passing report: the tightest home margin, or that no home was checked."""
    checked = [row for row in report.rows if row.margin is not None]
    if not checked:
        return f"calibration preflight: PASS, no home pose to check ({report.calibration_file or 'no calibration'})"
    tightest = min(checked, key=lambda row: row.margin)
    return (
        f"calibration preflight: PASS, {len(checked)} home poses within limits, "
        f"min margin {tightest.margin:.3f} rad at joint {tightest.name} ({report.calibration_file})"
    )


def _message(exc: BaseException) -> str:
    # KeyError reprs its argument with quotes; the other errors carry a plain message.
    text = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return f"{type(exc).__name__}: {text}"


def _rows(model: dict[str, Any]) -> list[JointRow]:
    groups = model["joint_groups"]
    ranges = model["joint_conversions"]["modes"][NORM_MODE_RANGE]
    home = model["home_positions"]
    rows = []
    for name in groups["all"]:
        if name in groups["base"] or name not in ranges or name not in model["joint_limits"]:
            continue
        limit = model["joint_limits"][name]
        rows.append(
            JointRow(
                name=name,
                calibrated=(ranges[name]["min"], ranges[name]["max"]),
                effective=(limit["min"], limit["max"]),
                home=float(home[name]) if name in home else None,
            )
        )
    return rows


def format_report(report: PreflightReport, min_margin: float) -> str:
    lines = [
        f"profile:     {report.profile_path}",
        f"calibration: {report.calibration_file or '-'}",
    ]
    if report.rows:
        header = ("joint", "calibrated [rad]", "effective [rad]", "limit set by", "home", "margin")
        table = [header]
        for row in report.rows:
            table.append(
                (
                    row.name,
                    f"{row.calibrated[0]:+.3f} .. {row.calibrated[1]:+.3f}",
                    f"{row.effective[0]:+.3f} .. {row.effective[1]:+.3f}",
                    row.bound_by,
                    "-" if row.home is None else f"{row.home:+.3f}",
                    "-" if row.margin is None else f"{row.margin:+.3f}",
                )
            )
        widths = [max(len(cells[i]) for cells in table) for i in range(len(header))]
        lines.append("")
        lines.extend(
            "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip() for cells in table
        )
    lines.append("")
    for warning in report.warnings(min_margin):
        lines.append(f"WARN  {warning}")
    if report.error:
        lines.append(f"FAIL  {report.error}")
        return "\n".join(lines)
    checked = [row.name for row in report.rows if row.home is not None]
    unchecked = [row.name for row in report.rows if row.home is None]
    lines.append(f"PASS  runtime checks of a physical launch; home within limits for joints: {_names(checked)}")
    if unchecked:
        lines.append(f"NOTE  no home pose in the profile, not checked: {_names(unchecked)}")
    lines.append("NOTE  not checked: the provider's calibration file schema, the attached arm")
    return "\n".join(lines)


def _names(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runtime_preflight",
        description=(
            "Check a physical runtime profile against the calibration file on this machine, "
            "offline: no serial port, driver or controller is touched."
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="runtime profile file path, or a profile name with --package (no peripherals fragment is applied)",
    )
    parser.add_argument(
        "--package",
        default="",
        help="read --profile as a name, from this package's profiles/<name>.yaml only",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help=f"warn when a home pose is closer than this to a limit, in rad (default {DEFAULT_MIN_MARGIN})",
    )
    return parser


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    from robot_runtime.launch_support import profile_share_path

    out = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.min_margin) or args.min_margin < 0:
        print("ERROR: --min-margin must be a finite non-negative number", file=out)
        return EXIT_USAGE
    try:
        if args.package:
            # Only the package's own profiles/: a same-named file in the
            # working directory, or a path smuggled in as the name, must not
            # stand in for the shipped profile.
            name = args.profile
            if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
                raise ValueError(f"with --package, --profile must be a profile name, not a path: {name!r}")
            profile_path = profile_share_path(args.package, name)
            if not profile_path.is_file():
                raise FileNotFoundError(f"no profile {args.profile!r} in {args.package}/profiles ({profile_path})")
        else:
            profile_path = Path(args.profile).expanduser()
        report = run_preflight(profile_path)
    except (ValueError, OSError, LookupError, TypeError, AttributeError, ArithmeticError, yaml.YAMLError) as exc:
        # ValueError covers ProfileError; LookupError covers an unknown --package;
        # load_profile raises TypeError/AttributeError for a section of the wrong type.
        print(f"ERROR: {_message(exc)}", file=out)
        return EXIT_USAGE
    print(format_report(report, args.min_margin), file=out)
    return EXIT_FAILED if report.error else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
