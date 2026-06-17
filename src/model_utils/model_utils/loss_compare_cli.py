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

CONFIG_ENV = "LOSS_COMPARE_CONFIG"
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/model_utils/loss_compare.yaml")

# Derived filenames under --exp-dir (kept in the user's preferred style).
DERIVED_TARGET = "target.json"
DERIVED_RAW_TARGET = "target_raw.json"
DERIVED_NOISE_DIR = "noises"


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
            parts.append(f"可选: {', '.join(self.choices)}")
        if self.example:
            parts.append(f"样例: {self.example}")
        if self.help_extra:
            parts.append(self.help_extra)
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Param table — order matters for wizard prompt sequence.
# ---------------------------------------------------------------------------
PARAMS: list[Param] = [
    Param(
        dest="policy_type",
        cli="--policy_type",
        meaning="模型类型 (会自动检测, 仅作回退)",
        example="pi05",
        default="act",
        choices=["act", "pi05"],
    ),
    Param(
        dest="device",
        cli="--device",
        meaning="推理后端: cpu/cuda/npu=torch, ascend_om=昇腾OM离线模型",
        example="ascend_om",
        default="cpu",
        choices=["cpu", "cuda", "npu", "ascend_om", "ascend_om_3403", "rknn"],
    ),
    Param(
        dest="policy_path",
        cli="--policy_path",
        meaning="策略模型目录 (含 config.json; OM 还需 config.om.json + .om)",
        example="/root/.../pi05/019200/",
        required_for_run=True,
    ),
    Param(
        dest="exp_dir",
        cli="--exp-dir",
        meaning="实验目录: target/raw/noise 自动派生到此 (target.json/target_raw.json/noises/)",
        example="/root/.../loss_compute_batches/0612",
    ),
    Param(
        dest="batch_path",
        cli="--batch_path",
        meaning="输入 batch JSON 文件",
        example="/root/.../batches_480_640_first_batch.json",
        required_for_run=True,
    ),
    Param(
        dest="task",
        cli="--task",
        meaning="VLA 策略 (pi05) 的自然语言任务提示词, 两端须一致",
        example='"pick up the cube"',
        default="",
    ),
    Param(
        dest="model_dtype",
        cli="--model_dtype",
        meaning="仅 torch 后端: 强制模型 dtype (编译后端忽略)",
        example="native",
        default="native",
        choices=["native", "fp16", "bf16", "fp32"],
        in_wizard=False,
    ),
    Param(
        dest="seed",
        cli="--seed",
        meaning="随机种子, 固定扩散/flow-matching 噪声",
        example="42",
        default=42,
        type=int,
    ),
    # The three long paths below are normally derived from --exp-dir; kept as
    # explicit overrides for backward compatibility / non-standard layouts.
    Param(
        dest="target_path",
        cli="--target_path",
        meaning="基准输出 JSON (默认由 --exp-dir 派生)",
        example="<exp-dir>/target.json",
        in_wizard=False,
    ),
    Param(
        dest="raw_target_path",
        cli="--raw-target-path",
        meaning="归一化空间(后处理前)动作 JSON (默认由 --exp-dir 派生)",
        example="<exp-dir>/target_raw.json",
        in_wizard=False,
    ),
    Param(
        dest="noise_dir",
        cli="--noise-dir",
        meaning="噪声文件目录, 跨机器确定性对比 (默认由 --exp-dir 派生)",
        example="<exp-dir>/noises/",
        in_wizard=False,
    ),
    Param(
        dest="generate_target",
        cli="--generate-target",
        meaning="进入基准数据生成模式 (否则为计算损失模式)",
        default=False,
        is_flag=True,
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
        raise ValueError(f"配置文件格式错误 (期望 mapping): {path}")
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
        description="Model Loss Comparison — 支持 profile / 向导 / 派生路径 (详见 README)",
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
    meta = parser.add_argument_group("配置 / profile / 向导")
    meta.add_argument(
        "--config", default=None, help=f"配置文件路径 (默认 {DEFAULT_CONFIG_PATH}, 或环境变量 {CONFIG_ENV})"
    )
    meta.add_argument("--profile", default=None, help="使用命名 profile 的参数组")
    meta.add_argument("--save-as", dest="save_as", default=None, help="把本次最终参数另存为指定名字的 profile")
    meta.add_argument("--init", action="store_true", help="强制进入交互向导 (首次设置)")
    meta.add_argument("--list-profiles", dest="list_profiles", action="store_true", help="列出已有 profile 后退出")
    meta.add_argument(
        "--force",
        action="store_true",
        help="generate-target 时允许覆盖已存在的派生/目标文件 (防误覆盖基准)",
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
            sources[dest] = "派生(exp-dir)"


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
        clashes.append(noise_dir + "/ (非空)")
    if clashes:
        listing = "\n  - ".join(clashes)
        raise SystemExit(
            "拒绝覆盖已存在的基准文件 (generate-target):\n  - "
            f"{listing}\n"
            "请更换 --exp-dir 指向新目录, 或显式加 --force 覆盖。"
        )


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------
def _prompt(param: Param, current_default: Any) -> Any:
    default_repr = "" if current_default in (None, "") else str(current_default)
    print(f"\n{param.dest}  ({param.cli})")
    print(f"  含义: {param.meaning}")
    if param.choices:
        print(f"  可选: {', '.join(param.choices)}")
    if param.example:
        print(f"  样例: {param.example}")
    suffix = f" [{default_repr}]" if default_repr else ""
    raw = input(f"  › 输入{suffix}: ").strip()
    if raw == "":
        return current_default
    if param.type is int:
        try:
            return int(raw)
        except ValueError:
            print("  (无法解析为整数, 使用默认值)")
            return current_default
    return raw


def run_wizard(seed_values: dict[str, Any]) -> dict[str, Any]:
    """Prompt each wizard param; return collected run params."""
    print("=" * 60)
    print(" loss_compare 交互式设置向导")
    print(" 回车=使用默认值 (方括号内); 默认值取自上次/profile")
    print("=" * 60)
    collected: dict[str, Any] = {}
    for p in PARAMS:
        if not p.in_wizard:
            continue
        cur = seed_values.get(p.dest, p.default)
        collected[p.dest] = _prompt(p, cur)

    # generate-target is a flow choice, ask explicitly (not a normal param).
    gen = input("\n生成基准模式? (生成基准=y / 计算损失=N) [N]: ").strip().lower()
    collected["generate_target"] = gen in ("y", "yes")
    return _clean_run_params(collected)


def maybe_save_profile(config: dict[str, Any], config_path: str, run_params: dict[str, Any]) -> None:
    ans = input("\n保存为 profile 以便复用? (y/N): ").strip().lower()
    if ans not in ("y", "yes"):
        return
    name = input("profile 名字 [default]: ").strip() or "default"
    config.setdefault("profiles", {})[name] = _strip_for_persist(_clean_run_params(run_params))
    save_config(config_path, config)
    print(f"✓ 已保存 profile '{name}' 到 {config_path}")
    print(f"  下次可直接运行: loss_compare --profile {name} --exp-dir <目录>")


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
            print(f"(无 profile) 配置文件: {config_path}")
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
            sources[k] = "向导"
        maybe_save_profile(config, config_path, run_params)
        merged = {**defaults, **run_params}
        for k in defaults:
            sources.setdefault(k, "defaults")
    else:
        profile_values: dict[str, Any] = {}
        if ns.profile:
            if ns.profile not in profiles:
                raise SystemExit(f"未找到 profile '{ns.profile}' (配置: {config_path}). 用 --list-profiles 查看。")
            profile_values = _clean_run_params(profiles[ns.profile])

        # Precedence: builtin < _last(only w/o profile) < defaults < profile < CLI
        merged = {}
        # builtin defaults
        for p in PARAMS:
            if p.default is not None:
                merged[p.dest] = p.default
                sources[p.dest] = "内置默认"
        # _last only applies when no explicit profile chosen
        if not ns.profile:
            for k, v in last.items():
                merged[k] = v
                sources[k] = "上次(_last)"
        for k, v in defaults.items():
            merged[k] = v
            sources[k] = "defaults"
        for k, v in profile_values.items():
            merged[k] = v
            sources[k] = f"profile:{ns.profile}"
        for k, v in cli_given.items():
            merged[k] = v
            sources[k] = "命令行"

    # Derive target/raw/noise from exp-dir (unless explicitly set).
    apply_exp_dir_derivation(merged, sources)

    # Validate required-for-run params.
    missing = [p.cli for p in PARAMS if p.required_for_run and not merged.get(p.dest)]
    # target_path is required to run but may be derived; check post-derivation.
    if not merged.get("target_path"):
        missing.append("--target_path 或 --exp-dir")
    if missing:
        raise SystemExit(
            "缺少必要参数: " + ", ".join(missing) + "\n提示: 用 --exp-dir 简化, 或 --init 走向导, 或 --profile 复用。"
        )

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
        print(f"✓ 已保存 profile '{ns.save_as}' 到 {config_path}")

    return ResolvedConfig(args=final, sources=sources, config_path=config_path, profile=ns.profile)


def print_effective(resolved: ResolvedConfig) -> None:
    args = resolved.args
    src = resolved.sources
    print("[loss_compare] 生效参数 (来源):")
    order = [
        "device",
        "policy_type",
        "policy_path",
        "exp_dir",
        "target_path",
        "raw_target_path",
        "noise_dir",
        "batch_path",
        "task",
        "seed",
        "model_dtype",
        "generate_target",
    ]
    for dest in order:
        val = getattr(args, dest, None)
        if val in (None, ""):
            continue
        origin = src.get(dest, "内置默认")
        arrow = "  → " if origin.startswith("派生") else "  "
        print(f"{arrow}{dest:16s}= {val}   ({origin})")
    print(f"  config: {resolved.config_path}")


# Run params that are transient per-invocation flow switches — useful to pass
# on the CLI but meaningless (and surprising) to persist into _last/profiles.
_TRANSIENT_KEYS = {"generate_target"}


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
        print(f"(警告) 无法写入 _last: {exc}")
