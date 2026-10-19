"""Read-only verification of the actually applied environment and generation."""

import json
from pathlib import Path

from . import configuration, core, shell_bridge


def inspect_current(home, values):
    raw = values.get("IBROBOT_DDS_STATE")
    if raw is None:
        return None
    result = {"backend": None, "temporary": None, "generation": None, "status": "not ready"}
    try:
        shell_bridge.check_owned(values, str(home))
        ledger = json.loads(raw)
        backend = ledger.get("backend")
        if (
            not isinstance(backend, str)
            or backend not in configuration.BACKENDS
            or type(ledger.get("temporary")) is not bool
        ):
            raise configuration.ConfigurationError("接管记录中的后端或临时状态无效", 4)
        generation = ledger.get("generation")
        if not isinstance(generation, str) or not Path(generation).is_absolute():
            raise configuration.ConfigurationError("接管记录中的产物路径无效", 4)
        result.update(backend=backend, temporary=ledger["temporary"], generation=generation)
        snapshot = ledger.get("configuration")
        if snapshot is None:
            raise configuration.ConfigurationError(
                "旧接管记录缺少配置快照；请重新 set，或 source .shrc_local 应用默认。", 5
            )
        # Recompute against current code/runtime/interface facts, then compare.
        config = core.parse_config(core.json_bytes(ledger["configuration"]).decode())
        if config["backend"] != backend:
            raise configuration.ConfigurationError("配置快照与接管后端不一致", 4)
        result["configuration"] = config
        plan = configuration.plan_configuration(config)
        status = configuration.generation_status(home, plan)
        if not status["prepared"] or status["generation"] != generation:
            raise configuration.ConfigurationError("当前终端配置已过期或产物缺失；请重新 set。", 5)
        result["binding"] = status["binding"]
        external = {key: value for key, value in values.items() if key not in shell_bridge.KEYS}
        preview = configuration.preview_environment(home, plan, external)
        configuration.check_environment(preview)
        expected = {key: preview["set"].get(key) for key in ledger["values"]}
        if expected != ledger["values"]:
            raise configuration.ConfigurationError("接管记录与当前 XML 环境映射不一致", 4)
        if values.get("IBROBOT_DDS_STATUS") != "ready":
            raise configuration.ConfigurationError("上次环境应用未成功；请修复后重新 set 或 source。", 5)
        result["status"] = "ready"
    except configuration.ConfigurationError as exc:
        result["diagnostic"] = {"code": exc.code, "message": str(exc)}
    except (core.ConfigError, OSError, ValueError, KeyError, TypeError) as exc:
        result["diagnostic"] = {"code": 5, "message": f"当前环境无法确认：{exc}"}
    return result
