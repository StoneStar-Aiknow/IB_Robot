"""Parse, plan and prepare DDS configuration without changing the shell."""

import os
from pathlib import Path

from . import core, cyclonedds, environment_plan, fastdds
from .platform_facts import FactsError, collect_facts

ROOT = Path(__file__).parent
BACKENDS = {"fastdds": fastdds, "cyclonedds": cyclonedds}


class ConfigurationError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def read_configuration(source):
    try:
        frozen = core.snapshot(source)
    except (core.ConfigError, OSError) as exc:
        raise ConfigurationError(str(exc), 6) from exc
    if frozen["content"] is None:
        raise ConfigurationError(f"配置文件不存在：{source}", 2)
    try:
        config = core.parse_config(frozen["content"].decode("utf-8"))
    except (core.ConfigError, UnicodeError) as exc:
        raise ConfigurationError(str(exc), 2) from exc
    return frozen, config


def plan(source, fact_reader=None):
    frozen, config = read_configuration(source)
    return plan_configuration(config, frozen, fact_reader)


def plan_configuration(config, frozen=None, fact_reader=None):
    backend = config["backend"]
    network = config["network"]
    try:
        runtime, binding = (fact_reader or collect_facts)(backend, network)
    except FactsError as exc:
        raise ConfigurationError(str(exc), 3) from exc
    discovery_domain = None
    if backend == "fastdds" and network["discovery"] != "lan_multicast":
        raw_domain = os.environ.get("ROS_DOMAIN_ID", "0")
        if not raw_domain.isascii() or not raw_domain.isdecimal() or not 0 <= int(raw_domain) <= 232:
            raise ConfigurationError("ROS_DOMAIN_ID 不在 Fast DDS 发现端口支持范围 0..232 内", 2)
        discovery_domain = str(int(raw_domain))
        xml = fastdds.render(network, binding, domain_id=int(discovery_domain))
    else:
        xml = BACKENDS[backend].render(network, binding)
    # Fingerprint generation code, not UI/CLI edits that cannot change XML.
    versions = {
        name: core.digest((ROOT / name).read_bytes())
        for name in (f"{backend}.py", "core.py", "platform_facts.py", "configuration.py")
    }
    identity = {
        "input": core.digest(
            core.json_bytes(
                {
                    "backend": backend,
                    "network": network,
                    "profile": config["backends"][backend]["profile"],
                    "sources": versions,
                }
            )
        ),
        "xml": core.digest(xml),
        "bindings": binding,
        "runtime": runtime,
    }
    return {
        "source": frozen,
        "config": config,
        "xml": xml,
        "identity": identity,
        "binding": binding,
        "runtime": runtime,
        "discovery_domain": discovery_domain,
    }


def generation_status(home, target):
    path = core.state_root(str(home)) / "generated" / core.digest(core.json_bytes(target["identity"]))
    prepared = path.exists() or path.is_symlink()
    if prepared:
        try:
            manifest = core.verify_generation(path)
            if manifest["identity"] != target["identity"] or (path / "profile.xml").read_bytes() != target["xml"]:
                raise core.ConfigError("生成配置与当前目标不一致")
        except (core.ConfigError, OSError) as exc:
            raise ConfigurationError(str(exc), 5) from exc
    return {"prepared": prepared, "generation": str(path) if prepared else None, "binding": target["binding"]}


def prepare(home, target, fact_reader=None):
    def recheck():
        if plan(target["source"]["path"], fact_reader) != target:
            raise ConfigurationError("配置或本机环境在准备期间改变；请重试。", 6)

    recheck()
    try:
        core.publish_generation(core.state_root(str(home)), target["identity"], target["xml"])
    except (core.ConfigError, OSError) as exc:
        raise ConfigurationError(str(exc), 6) from exc
    recheck()
    return generation_status(home, target)


def preview_environment(home, target, inherited):
    generation = core.state_root(str(home)) / "generated" / core.digest(core.json_bytes(target["identity"]))
    return environment_plan.preview(
        target["config"]["backend"], generation / "profile.xml", target["config"]["network"]["discovery"], inherited
    )


def check_environment(preview):
    if preview["conflicts"]:
        names = ", ".join(item["variable"] for item in preview["conflicts"])
        raise ConfigurationError(f"DDS 环境冲突：{names}；当前环境未修改。", 4)
