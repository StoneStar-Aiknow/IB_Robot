"""Config / wizard / derived-path layer for ``loss_compare``.

This module keeps ``loss_compare.py`` itself focused on inference + metrics by
moving all the *ergonomics* here:

- **Param metadata** (one source of truth): name, help, example, default, type.
  Reused by ``--help``, the interactive wizard, and config validation, so the
  meaning of each flag is documented in exactly one place.
- **YAML config** at ``~/.config/model_utils/loss_compare.yaml`` with three
  sections: ``defaults`` (shared), ``profiles`` (named param groups), and
  ``_last`` (auto-written after each run — this replaces a separate "remember
  last args" cache).
- **Merge precedence** (high -> low):
  ``CLI > --profile > defaults > _last (only when no --profile) > builtin``.
- **Derived paths**: a single ``--exp-dir`` expands to ``target.json`` /
  ``target_raw.json`` / ``noises/`` so the three long, error-prone paths
  collapse into one directory.
- **Wizard**: on first use (no config / no _last) or ``--init`` it prompts each
  field with its meaning + example + default, then offers to save a profile.

Inference selection uses a named deployment from ``inference_manifest.json``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

CONFIG_ENV = "LOSS_COMPARE_CONFIG"
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/model_utils/loss_compare.yaml")

# Derived filenames under --exp-dir (kept in the user's preferred style).
DERIVED_TARGET = "target.json"
DERIVED_RAW_TARGET = "target_raw.json"
DERIVED_NOISE_DIR = "noises"

# Provenance labels (shown by print_effective so the user knows where each
# value came from).  Kept as constants because some control flow checks them.
SRC_CLI = "cli"
SRC_PROFILE = "profile"  # used as f"{SRC_PROFILE}:{name}"
SRC_DEFAULTS = "defaults"
SRC_LAST = "last"
SRC_BUILTIN = "builtin"
SRC_WIZARD = "wizard"
SRC_DERIVED = "derived(exp-dir)"

# Transient values are never written back; new diagnostics are also ignored in config files.
_TRANSIENT_KEYS = {"generate_target", "metrics_json", "schedule_override_path", "curvature_log_path"}
_CLI_ONLY_KEYS = {"metrics_json", "schedule_override_path", "curvature_log_path"}


@dataclass
class Param:
    """Single source of truth for one CLI/config field.

    ``cli`` is the argparse flag (e.g. ``--policy_path``); ``dest`` is the
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
        dest="deployment",
        cli="--deployment",
        meaning="Named deployment from the policy bundle inference_manifest.json",
        example="ascend",
        default="cpu",
    ),
    Param(
        dest="policy_path",
        cli="--policy_path",
        meaning="Policy bundle directory: must contain config.json and inference_manifest.json for compiled runtimes",
        required_for_run=True,
    ),
    Param(
        dest="exp_dir",
        cli="--exp-dir",
        meaning="Experiment directory: auto-derives target.json / target_raw.json / noises/ to avoid typing 3 long paths",
    ),
    Param(
        dest="batch_path",
        cli="--batch_path",
        meaning="Raw observation batch path (.safetensors; legacy .json supported)",
        required_for_run=True,
    ),
    Param(
        dest="task",
        cli="--task",
        meaning="Natural language task prompt for VLA policies (pi05). Must match between generate-target and compute-loss",
        example='"pick up the cube"',
        default="",
    ),
    Param(
        dest="model_dtype",
        cli="--model_dtype",
        meaning="Optional model dtype requested from the selected deployment; unsupported deployments reject it",
        example="native",
        default="native",
        choices=["native", "fp16", "bf16", "fp32"],
        in_wizard=False,
    ),
    Param(
        dest="seed",
        cli="--seed",
        meaning="Random seed for reproducible diffusion/flow-matching noise",
        example="42",
        default=42,
        type=int,
    ),
    # The three long paths below are normally derived from --exp-dir; kept as
    # explicit overrides for non-standard layouts.
    Param(
        dest="target_path",
        cli="--target_path",
        meaning="Baseline inference output JSON (normally derived from --exp-dir)",
        example="<exp-dir>/target.json",
        in_wizard=False,
    ),
    Param(
        dest="raw_target_path",
        cli="--raw-target-path",
        meaning="Normalized-space (pre-postprocessor) action JSON (normally derived from --exp-dir)",
        example="<exp-dir>/target_raw.json",
        in_wizard=False,
    ),
    Param(
        dest="noise_dir",
        cli="--noise-dir",
        meaning="Noise file directory for cross-machine deterministic comparison (normally derived from --exp-dir)",
        example="<exp-dir>/noises/",
        in_wizard=False,
    ),
    Param(
        dest="generate_target",
        cli="--generate-target",
        meaning="Enter generate-target mode (else compute-loss mode)",
        default=False,
        is_flag=True,
        in_wizard=False,
    ),
    Param(
        dest="metrics_json",
        cli="--metrics-json",
        meaning="Write aggregate comparison metrics as machine-readable JSON",
        in_wizard=False,
    ),
    Param(
        dest="schedule_override_path",
        cli="--schedule-override-path",
        meaning="Transient strict PI0.5 schedule override passed to the selected runtime",
        in_wizard=False,
    ),
    Param(
        dest="curvature_log_path",
        cli="--curvature-log-path",
        meaning="Transient PI0.5 per-inference curvature JSONL output path",
        in_wizard=False,
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


def _clean_config_run_params(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in _clean_run_params(values).items() if key not in _CLI_ONLY_KEYS}


# ---------------------------------------------------------------------------
# argparse construction (help text comes from the param table)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model Loss Comparison — profile / wizard / derived-path support (see README)",
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
        help="Allow overwriting existing derived/target files during generate-target (prevents baseline clobber)",
    )
    return parser


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def apply_exp_dir_derivation(values: dict[str, Any], sources: dict[str, str]) -> None:
    """Fill target/raw/noise from exp_dir unless explicitly provided."""
    exp_dir = values.get("exp_dir")
    if not exp_dir:
        return
    derivations = {
        "target_path": os.path.join(exp_dir, DERIVED_TARGET),
        "raw_target_path": os.path.join(exp_dir, DERIVED_RAW_TARGET),
        "noise_dir": os.path.join(exp_dir, DERIVED_NOISE_DIR),
    }
    for dest, derived in derivations.items():
        if values.get(dest) is None:
            values[dest] = derived
            sources[dest] = SRC_DERIVED


def check_overwrite_guard(values: dict[str, Any], force: bool) -> None:
    """In generate-target mode, refuse to clobber existing baseline files."""
    if not values.get("generate_target") or force:
        return
    clashes = []
    for dest in ("target_path", "raw_target_path"):
        path = values.get(dest)
        if path and os.path.exists(path):
            clashes.append(path)
    noise_dir = values.get("noise_dir")
    if noise_dir and os.path.isdir(noise_dir) and os.listdir(noise_dir):
        clashes.append(noise_dir + "/ (non-empty)")
    if clashes:
        listing = "\n  - ".join(clashes)
        raise SystemExit(
            "Refusing to overwrite existing baseline files (generate-target mode):\n  - "
            f"{listing}\n"
            "Change --exp-dir to a new directory, or add --force to overwrite."
        )


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
    print(" loss_compare interactive setup wizard")
    print(" Press Enter to use default (shown in brackets)")
    print(" Defaults are taken from last run or profile")
    print("=" * 60)
    collected: dict[str, Any] = {}
    for p in PARAMS:
        if not p.in_wizard:
            continue
        cur = seed_values.get(p.dest, p.default)
        collected[p.dest] = _prompt(p, cur)

    # generate-target is a flow choice, ask explicitly (not a normal param).
    gen = input("\nGenerate baseline mode? (generate=y / compute-loss=N) [N]: ").strip().lower()
    collected["generate_target"] = gen in ("y", "yes")
    return _clean_run_params(collected)


def maybe_save_profile(config: dict[str, Any], config_path: str, run_params: dict[str, Any]) -> None:
    ans = input("\nSave as profile for reuse? (y/N): ").strip().lower()
    if ans not in ("y", "yes"):
        return
    name = input("Profile name [default]: ").strip() or "default"
    config.setdefault("profiles", {})[name] = _strip_for_persist(_clean_run_params(run_params))
    save_config(config_path, config)
    print(f"✓ Profile '{name}' saved to {config_path}")
    print(f"  Next time run: loss_compare --profile {name} --exp-dir <dir>")


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
                print(f"  - {name}: {_clean_config_run_params(vals)}")
        raise SystemExit(0)

    # CLI-provided run params (non-None means user passed it).
    cli_values = {p.dest: getattr(ns, p.dest) for p in PARAMS}
    cli_given = {k: v for k, v in cli_values.items() if v is not None}

    defaults = _clean_config_run_params(config.get("defaults", {}))
    last = _clean_config_run_params(config.get("_last", {}))
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
    else:
        profile_values: dict[str, Any] = {}
        if ns.profile:
            if ns.profile not in profiles:
                raise SystemExit(f"Profile '{ns.profile}' not found (config: {config_path}). Use --list-profiles.")
            profile_values = _clean_config_run_params(profiles[ns.profile])

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

    # Derive target/raw/noise from exp-dir (unless explicitly set).
    apply_exp_dir_derivation(merged, sources)

    # Validate required-for-run params.
    missing = [p.cli for p in PARAMS if p.required_for_run and not merged.get(p.dest)]
    # target_path is required to run but may be derived; check post-derivation.
    if not merged.get("target_path"):
        missing.append("--target_path or --exp-dir")
    if missing:
        raise SystemExit(
            "Missing required params: "
            + ", ".join(missing)
            + "\nTip: use --exp-dir to simplify, --init for the wizard, or --profile to reuse."
        )

    for dest in ("metrics_json", "schedule_override_path", "curvature_log_path"):
        value = merged.get(dest)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SystemExit(f"{PARAMS_BY_DEST[dest].cli} must be a non-empty string")
    if merged.get("generate_target") and any(
        merged.get(dest) is not None for dest in ("metrics_json", "schedule_override_path", "curvature_log_path")
    ):
        raise SystemExit("Diagnostic/tuning options are compute-only and cannot be used with --generate-target")

    # Overwrite guard for generate-target.
    check_overwrite_guard(merged, ns.force)

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
    print("[loss_compare] effective params (source):")
    order = [
        "deployment",
        "policy_path",
        "exp_dir",
        "target_path",
        "raw_target_path",
        "noise_dir",
        "batch_path",
        "task",
        "seed",
        "model_dtype",
        "metrics_json",
        "schedule_override_path",
        "curvature_log_path",
        "generate_target",
    ]
    for dest in order:
        val = getattr(args, dest, None)
        if val in (None, ""):
            continue
        origin = src.get(dest, SRC_BUILTIN)
        arrow = "  → " if origin.startswith("derived") else "  "
        print(f"{arrow}{dest:16s}= {val}   ({origin})")
    print(f"  config: {resolved.config_path}")


def _strip_for_persist(run_params: dict[str, Any]) -> dict[str, Any]:
    """Prepare run params for writing into ``_last``/profiles.

    Drops:
    - derived ``target/raw/noise`` paths when an ``exp_dir`` exists (they are
      *computed*, so persisting them would make a later ``--exp-dir`` change
      silently ineffective);
    - transient flow flags such as ``generate_target`` (remembering "I was in
      generate mode last time" would wrongly flip a later compute-loss run into
      overwrite-guard territory).
    """
    out = {k: v for k, v in run_params.items() if k not in _TRANSIENT_KEYS}
    if out.get("exp_dir"):
        for k in ("target_path", "raw_target_path", "noise_dir"):
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
        print(f"(warning) failed to write _last: {exc}")
