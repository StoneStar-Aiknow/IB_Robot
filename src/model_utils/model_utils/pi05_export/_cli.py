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
"""Config / wizard / derived-path layer for the PI05 export pipeline.

This module keeps ``__main__.py`` focused on stage orchestration by moving all
the *ergonomics* here, mirroring the proven ``loss_compare_cli`` design so the
two tools share one mental model:

- **Param metadata** (one source of truth): name, help, example, default, type.
  Reused by ``--help``, the interactive wizard, and config validation, so the
  meaning of each flag is documented in exactly one place.
- **YAML config** at ``~/.config/model_utils/pi05_export.yaml`` with three
  sections: ``defaults`` (shared), ``profiles`` (named param groups), and
  ``_last`` (auto-written after each run — replaces a separate "remember last
  args" cache).
- **Merge precedence** (high -> low):
  ``CLI > --profile > defaults > _last (only when no --profile) > builtin``.
- **Derived paths**: a single ``--exp-dir`` expands to ``onnx/`` (ONNX output)
  and ``runtime_save/`` (VLM->AE handoff tensors) so the two long paths collapse
  into one directory. OM artifacts + ``config.om.json`` still live next to the
  policy (the ``device:=ascend_om`` runtime contract), so they are NOT derived.
- **Wizard**: on first use (no config / no _last) or ``--init`` it prompts each
  field with its meaning + example + default, then offers to save a profile.

Backward compatible: every historical explicit flag still works and overrides
whatever a profile/derivation would supply.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

CONFIG_ENV = "PI05_EXPORT_CONFIG"
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/model_utils/pi05_export.yaml")

# Derived sub-directories under --exp-dir.
DERIVED_ONNX_DIR = "onnx"
DERIVED_RUNTIME_DIR = "runtime_save"

# Provenance labels (shown by print_effective so the user knows where each
# value came from). Kept as constants because some control flow checks them.
SRC_CLI = "cli"
SRC_PROFILE = "profile"  # used as f"{SRC_PROFILE}:{name}"
SRC_DEFAULTS = "defaults"
SRC_LAST = "last"
SRC_BUILTIN = "builtin"
SRC_WIZARD = "wizard"
SRC_DERIVED = "derived(exp-dir)"


@dataclass
class Param:
    """Single source of truth for one CLI/config field.

    ``cli`` is the argparse flag (e.g. ``--policy-path``); ``dest`` is the
    attribute name on the resolved namespace (argparse derives it the same
    way). ``example`` and ``meaning`` feed the wizard prompts.
    """

    dest: str
    cli: str
    meaning: str
    example: str = ""
    default: Any = None
    type: Callable[[str], Any] = str
    choices: list[str] | None = None
    is_flag: bool = False  # store_true style
    required_for_run: bool = False  # must be present (post-merge) to run
    in_wizard: bool = True
    help_extra: str = ""

    @property
    def help_text(self) -> str:
        parts = [self.meaning]
        if self.choices:
            parts.append(f"choices: {', '.join(self.choices)}")
        if self.example:
            parts.append(f"e.g. {self.example}")
        if self.help_extra:
            parts.append(self.help_extra)
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Param table — order matters for wizard prompt sequence.
# ---------------------------------------------------------------------------
PARAMS: list[Param] = [
    Param(
        dest="policy_path",
        cli="--policy-path",
        meaning="Local PI05 policy directory (config + weights); also where OM + config.om.json are written",
        required_for_run=True,
    ),
    Param(
        dest="exp_dir",
        cli="--exp-dir",
        meaning="Experiment directory: auto-derives onnx/ and runtime_save/ to avoid typing 2 long paths",
    ),
    Param(
        dest="dtype",
        cli="--dtype",
        meaning="Export precision, applied consistently to BOTH segments",
        example="fp16",
        default="fp16",
        choices=["fp16", "fp32", "auto"],
    ),
    Param(
        dest="soc_version",
        cli="--soc-version",
        meaning="Target Ascend SoC; when given, ATC->OM compile runs too (see `npu-smi info`)",
        example="Ascend310P3",
    ),
    Param(
        dest="device",
        cli="--device",
        meaning="Torch device for export / verification; the bare type is encoded in the ONNX filename",
        example="cpu",
        default="cpu",
    ),
    Param(
        dest="task",
        cli="--task",
        meaning="Task prompt required by --verify; must match the deployment default_task",
        example='"pick up the cup"',
    ),
    # The two paths below are normally derived from --exp-dir; kept as explicit
    # overrides for backward compatibility / non-standard layouts.
    Param(
        dest="output_dir",
        cli="--output-dir",
        meaning="Directory for the exported ONNX files (normally derived from --exp-dir)",
        default="outputs/onnx",
        in_wizard=False,
    ),
    Param(
        dest="runtime_save_dir",
        cli="--runtime-save-dir",
        meaning="Directory for the VLM->AE handoff tensors (normally derived from --exp-dir)",
        default="runtime_save",
        in_wizard=False,
    ),
    Param(
        dest="log_level",
        cli="--log-level",
        meaning="Logging level",
        default="INFO",
        in_wizard=False,
    ),
    # ---- W8A8 quantization (optional; off by default) ----
    Param(
        dest="quantize",
        cli="--quantize",
        meaning="Quantize BOTH segments to W8A8 (shorthand for --quantize-vlm --quantize-ae; needs --batch-path)",
        default=False,
        is_flag=True,
        in_wizard=False,
    ),
    Param(
        dest="quantize_vlm",
        cli="--quantize-vlm",
        meaning="Quantize the VLM (gemma_2b) ONNX to W8A8 (needs --batch-path for real calibration)",
        default=False,
        is_flag=True,
        in_wizard=False,
    ),
    Param(
        dest="quantize_ae",
        cli="--quantize-ae",
        meaning="Quantize the Action Expert (gemma_300m) ONNX to W8A8",
        default=False,
        is_flag=True,
        in_wizard=False,
    ),
    Param(
        dest="batch_path",
        cli="--batch-path",
        meaning="Real calibration batches JSON (REQUIRED for VLM quantization; random data yields a garbage model)",
        in_wizard=False,
    ),
    Param(
        dest="calib_dir",
        cli="--calib-dir",
        meaning="AE calibration sample dir (past_kv_tensor.* + prefix_pad_masks.*); defaults to --runtime-save-dir",
        in_wizard=False,
    ),
    Param(
        dest="num_calib",
        cli="--num-calib",
        meaning="Number of calibration samples to use (<=0 = all)",
        default=16,
        type=int,
        in_wizard=False,
    ),
    Param(
        dest="amp_num",
        cli="--amp-num",
        meaning="msModelSlim auto-mixed-precision fp16 fallback layer count (accuracy safety valve)",
        default=0,
        type=int,
        in_wizard=False,
    ),
    Param(
        dest="verify",
        cli="--verify",
        meaning="Run split-vs-monolithic equivalence verification at the end (needs --task)",
        default=False,
        is_flag=True,
        in_wizard=False,  # asked explicitly in the wizard as a flow choice
    ),
]

PARAMS_BY_DEST = {p.dest: p for p in PARAMS}

# Keys that live on the resolved namespace but are not user "run params"
# (control flow only); they are never persisted into _last/profiles.
_META_KEYS = {
    "config",
    "profile",
    "save_as",
    "init",
    "list_profiles",
    "force",
}


@dataclass
class ResolvedConfig:
    """Outcome of resolution: final args + provenance for printing."""

    args: argparse.Namespace
    sources: dict[str, str] = field(default_factory=dict)
    config_path: str = DEFAULT_CONFIG_PATH
    profile: str | None = None


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file (expected a mapping): {path}")
    return data


def save_config(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _clean_run_params(values: dict[str, Any]) -> dict[str, Any]:
    """Keep only persistable run params (drop meta / None)."""
    return {k: v for k, v in values.items() if k in PARAMS_BY_DEST and k not in _META_KEYS and v is not None}


# ---------------------------------------------------------------------------
# argparse construction (help text comes from the param table)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m model_utils.pi05_export",
        description="One-command PI05 export pipeline — profile / wizard / derived-path support (see README)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    for p in PARAMS:
        kwargs: dict[str, Any] = {"dest": p.dest, "help": p.help_text}
        if p.is_flag:
            kwargs["action"] = "store_true"
            kwargs["default"] = None  # None => "not given on CLI"
        else:
            kwargs["type"] = p.type
            kwargs["default"] = None  # resolution layer applies real defaults
            if p.choices:
                kwargs["choices"] = p.choices
        parser.add_argument(p.cli, **kwargs)

    # --- meta flags (not part of run params) ---
    meta = parser.add_argument_group("Config / profile / wizard")
    meta.add_argument(
        "--config", default=None, help=f"Config file path (default {DEFAULT_CONFIG_PATH}, or env var {CONFIG_ENV})"
    )
    meta.add_argument("--profile", default=None, help="Use a named profile parameter group")
    meta.add_argument(
        "--save-as", dest="save_as", default=None, help="Save this run's effective params as a named profile"
    )
    meta.add_argument("--init", action="store_true", help="Force interactive wizard (first-time setup)")
    meta.add_argument(
        "--list-profiles", dest="list_profiles", action="store_true", help="List available profiles and exit"
    )
    meta.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every stage even if its output already exists",
    )
    return parser


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def apply_exp_dir_derivation(values: dict[str, Any], sources: dict[str, str]) -> None:
    """Fill output_dir / runtime_save_dir from exp_dir unless explicitly provided.

    These two params carry builtin defaults, so "explicitly provided" means a
    source stronger than builtin (CLI / profile / defaults / last). Derivation
    therefore overrides only the builtin fallback, never a user-chosen value.
    """
    exp_dir = values.get("exp_dir")
    if not exp_dir:
        return
    derivations = {
        "output_dir": os.path.join(exp_dir, DERIVED_ONNX_DIR),
        "runtime_save_dir": os.path.join(exp_dir, DERIVED_RUNTIME_DIR),
    }
    for dest, derived in derivations.items():
        if values.get(dest) is None or sources.get(dest) == SRC_BUILTIN:
            values[dest] = derived
            sources[dest] = SRC_DERIVED


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------
def _prompt(param: Param, current_default: Any) -> Any:
    default_repr = "" if current_default in (None, "") else str(current_default)
    print(f"\n{param.dest}  ({param.cli})")
    print(f"  {param.meaning}")
    if param.choices:
        print(f"  choices: {', '.join(param.choices)}")
    if param.example:
        print(f"  e.g. {param.example}")
    suffix = f" [{default_repr}]" if default_repr else ""
    raw = input(f"  › {suffix}: ").strip()
    if raw == "":
        return current_default
    if param.type is int:
        try:
            return int(raw)
        except ValueError:
            print("  (cannot parse as int, using default)")
            return current_default
    return raw


def run_wizard(seed_values: dict[str, Any]) -> dict[str, Any]:
    """Prompt each wizard param; return collected run params."""
    print("=" * 60)
    print(" pi05_export interactive setup wizard")
    print(" Press Enter to use default (shown in brackets)")
    print(" Defaults are taken from last run or profile")
    print("=" * 60)
    collected: dict[str, Any] = {}
    for p in PARAMS:
        if not p.in_wizard:
            continue
        cur = seed_values.get(p.dest, p.default)
        collected[p.dest] = _prompt(p, cur)

    # --verify is a flow choice, ask explicitly (not a normal param).
    ver = input("\nRun equivalence verification at the end? (verify=y / N) [N]: ").strip().lower()
    collected["verify"] = ver in ("y", "yes")
    # The verify task prompt only matters when verification is enabled.
    if collected["verify"] and not collected.get("task"):
        collected["task"] = _prompt(PARAMS_BY_DEST["task"], seed_values.get("task"))
    return _clean_run_params(collected)


def maybe_save_profile(config: dict[str, Any], config_path: str, run_params: dict[str, Any]) -> None:
    ans = input("\nSave as profile for reuse? (y/N): ").strip().lower()
    if ans not in ("y", "yes"):
        return
    name = input("Profile name [default]: ").strip() or "default"
    config.setdefault("profiles", {})[name] = _strip_for_persist(_clean_run_params(run_params))
    save_config(config_path, config)
    print(f"✓ Profile '{name}' saved to {config_path}")
    print(f"  Next time run: python -m model_utils.pi05_export --profile {name} --policy-path <dir>")


# ---------------------------------------------------------------------------
# Resolution (the heart of the merge precedence)
# ---------------------------------------------------------------------------
def resolve(argv: list[str] | None = None) -> ResolvedConfig:
    parser = build_parser()
    ns = parser.parse_args(argv)

    config_path = ns.config or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    if ns.list_profiles:
        profiles = config.get("profiles", {})
        if not profiles:
            print(f"(no profiles) config: {config_path}")
        else:
            print(f"profiles in {config_path}:")
            for name, vals in profiles.items():
                print(f"  - {name}: {_clean_run_params(vals)}")
        raise SystemExit(0)

    # CLI-provided run params (non-None means user passed it).
    cli_values = {p.dest: getattr(ns, p.dest) for p in PARAMS}
    cli_given = {k: v for k, v in cli_values.items() if v is not None}

    defaults = _clean_run_params(config.get("defaults", {}))
    last = _clean_run_params(config.get("_last", {}))
    profiles = config.get("profiles", {})

    # Decide whether to run the wizard.
    no_config = not os.path.exists(config_path)
    wizard_triggered = ns.init or (not cli_given and not last and not ns.profile and no_config)
    # Also run wizard when invoked truly bare and there is nothing to fall back on.
    if not argv and not cli_given and not ns.profile and not last and not ns.init:
        wizard_triggered = True

    sources: dict[str, str] = {}

    if wizard_triggered:
        seed = {**defaults, **last}
        run_params = run_wizard(seed)
        for k in run_params:
            sources[k] = SRC_WIZARD
        maybe_save_profile(config, config_path, run_params)
        merged = {**defaults, **run_params}
        for k in defaults:
            sources.setdefault(k, SRC_DEFAULTS)
        # builtin defaults for anything still missing
        for p in PARAMS:
            if p.default is not None and p.dest not in merged:
                merged[p.dest] = p.default
                sources[p.dest] = SRC_BUILTIN
    else:
        profile_values: dict[str, Any] = {}
        if ns.profile:
            if ns.profile not in profiles:
                raise SystemExit(f"Profile '{ns.profile}' not found (config: {config_path}). Use --list-profiles.")
            profile_values = _clean_run_params(profiles[ns.profile])

        # Precedence: builtin < _last(only w/o profile) < defaults < profile < CLI
        merged = {}
        # builtin defaults
        for p in PARAMS:
            if p.default is not None:
                merged[p.dest] = p.default
                sources[p.dest] = SRC_BUILTIN
        # _last only applies when no explicit profile chosen
        if not ns.profile:
            for k, v in last.items():
                merged[k] = v
                sources[k] = SRC_LAST
        for k, v in defaults.items():
            merged[k] = v
            sources[k] = SRC_DEFAULTS
        for k, v in profile_values.items():
            merged[k] = v
            sources[k] = f"{SRC_PROFILE}:{ns.profile}"
        for k, v in cli_given.items():
            merged[k] = v
            sources[k] = SRC_CLI

    # Derive output_dir / runtime_save_dir from exp-dir (unless explicitly set).
    apply_exp_dir_derivation(merged, sources)

    # --quantize is shorthand: expand it into per-segment flags so the rest of
    # the pipeline only ever inspects quantize_vlm / quantize_ae.
    if merged.get("quantize"):
        for dest in ("quantize_vlm", "quantize_ae"):
            if not merged.get(dest):
                merged[dest] = True
                sources[dest] = sources.get("quantize", SRC_CLI)

    # Validate required-for-run params.
    missing = [p.cli for p in PARAMS if p.required_for_run and not merged.get(p.dest)]
    if missing:
        raise SystemExit(
            "Missing required params: "
            + ", ".join(missing)
            + "\nTip: use --init for the wizard, or --profile to reuse a saved param group."
        )
    # --verify needs a task prompt; check post-merge so profile/wizard can supply it.
    if merged.get("verify") and not merged.get("task"):
        raise SystemExit(
            "--verify requires --task (the deployment task prompt)."
            "\nTip: pass --task '<prompt>', or drop --verify to skip verification."
        )
    # VLM W8A8 calibration needs a real batch (random data => garbage model).
    if merged.get("quantize_vlm") and not merged.get("batch_path"):
        raise SystemExit(
            "--quantize-vlm requires --batch-path (real calibration batches; random data yields a garbage model)."
            "\nTip: pass --batch-path <batches.json>, or drop --quantize-vlm."
        )

    # Build final namespace with all PARAMS present (fill remaining with builtin).
    final = argparse.Namespace()
    for p in PARAMS:
        setattr(final, p.dest, merged.get(p.dest, p.default))
    # carry meta flags for caller
    final.config = config_path
    final.profile = ns.profile
    final.save_as = ns.save_as
    final.force = ns.force

    # Optional explicit save-as.
    if ns.save_as:
        config.setdefault("profiles", {})[ns.save_as] = _strip_for_persist(_clean_run_params(vars(final)))
        save_config(config_path, config)
        print(f"✓ Profile '{ns.save_as}' saved to {config_path}")

    return ResolvedConfig(args=final, sources=sources, config_path=config_path, profile=ns.profile)


def print_effective(resolved: ResolvedConfig) -> None:
    args = resolved.args
    src = resolved.sources
    print("[pi05_export] effective params (source):")
    order = [
        "policy_path",
        "exp_dir",
        "output_dir",
        "runtime_save_dir",
        "dtype",
        "device",
        "soc_version",
        "quantize_vlm",
        "quantize_ae",
        "batch_path",
        "calib_dir",
        "num_calib",
        "amp_num",
        "verify",
        "task",
        "log_level",
    ]
    for dest in order:
        val = getattr(args, dest, None)
        if val in (None, "", False):
            continue
        origin = src.get(dest, SRC_BUILTIN)
        arrow = "  → " if origin.startswith("derived") else "  "
        print(f"{arrow}{dest:18s}= {val}   ({origin})")
    print(f"  config: {resolved.config_path}")


# Run params that are transient per-invocation flow switches — useful to pass
# on the CLI but meaningless (and surprising) to persist into _last/profiles.
# Quant flow flags belong here too: "I quantized last time" must not silently
# re-trigger quantization (and its --batch-path requirement) on a later run.
_TRANSIENT_KEYS = {"verify", "quantize", "quantize_vlm", "quantize_ae"}


def _strip_for_persist(run_params: dict[str, Any]) -> dict[str, Any]:
    """Prepare run params for writing into ``_last``/profiles.

    Drops:
    - derived ``output_dir`` / ``runtime_save_dir`` when an ``exp_dir`` exists
      (they are *computed*, so persisting them would make a later ``--exp-dir``
      change silently ineffective);
    - transient flow flags such as ``verify`` (remembering "I verified last
      time" would wrongly re-trigger verification — and its --task requirement —
      on a later plain export run).
    """
    out = {k: v for k, v in run_params.items() if k not in _TRANSIENT_KEYS}
    if out.get("exp_dir"):
        for k in ("output_dir", "runtime_save_dir"):
            out.pop(k, None)
    return out


def write_last(resolved: ResolvedConfig) -> None:
    """Persist this run's params as _last (replaces a separate cache)."""
    config = load_config(resolved.config_path)
    run_params = _strip_for_persist(_clean_run_params(vars(resolved.args)))
    if resolved.profile:
        run_params["profile"] = resolved.profile
    config["_last"] = run_params
    try:
        save_config(resolved.config_path, config)
    except OSError as exc:
        # Persisting _last is a convenience, not critical; never fail the run.
        print(f"(warning) failed to write _last: {exc}")
