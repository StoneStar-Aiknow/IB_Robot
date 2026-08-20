# Hermes 与 IB-Robot 同步

此目录是 IB-Robot Hermes 集成的唯一发布源，包含推荐 `SOUL.md`、TTS hook 和手动同步入口。
同步会安装当前 `ibrobot-control` Skill，绑定当前 workspace、robot config 和 ROS Domain，并清理已退役的
`ibrobot-robot-control` Plugin 残留副本（即时执行改由 `robot-skill` 的 `confirm-plan` + `execute-plan` 承担）。

## 前置条件

- 已安装 Hermes Agent 0.16.0 或更新版本。
- 已构建 `robot_skill_cli`、`robot_config`、`ibrobot_msgs` 和 TTS 相关包。
- 当前终端已 source 目标 IB-Robot workspace。
- 启用语音时，目标 robot config 的 `voice_tts.enabled` 必须为 `true`。
- 真机运动仍必须由操作员在启动 pipeline 时显式设置 `authorize_motion:=true`。

## 同步

从仓库根目录执行：

```bash
source .shrc_local
export ROS_DOMAIN_ID=56

hermes-robot-configure \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --restart-gateway
```

`.shrc_local` 是 ROS+venv+install overlay 的 SSOT 入口（见 `AGENTS.md` 环境初始化章节），
无需重复 `source install/setup.bash`。

也可以使用此目录中的 shell 入口：

```bash
bash src/robot_skill_cli/resource/hermes/sync_hermes.sh \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --restart-gateway
```

先预览而不写文件：

```bash
hermes-robot-configure --config-name so101_single_arm --dry-run
```

`--soul-mode replace` 会先备份现有 `SOUL.md`，再安装仓库中的完整推荐版本。使用
`--soul-mode merge` 会在现有文件中维护一个 `IBROBOT-MANAGED` 区块；`--soul-mode skip` 不修改 SOUL。

## 同步产物

同步在当前 Hermes profile 下生成以下受管文件（带 managed 标记，重跑同步即覆盖，不要手改）：

- `skills/ibrobot-control/SKILL.md`：当前 `ibrobot-control` Agent Skill 副本。
- `ibrobot/bin/robot-skill`：绑定 `ROBOT_CONFIG`、`ROS_DOMAIN_ID` 并 source workspace
  `.shrc_local` 的 wrapper；进入 Hermes 会话后用它而非裸 `robot-skill`，且不得再传
  `--config-name` / `--config-path`。
- `ibrobot/ibrobot-env.sh`：写入 `terminal.shell_init_files` 的环境文件，并把
  `auto_source_bashrc` 置 `false`，使受管环境优先于用户 bashrc；它同时把 `ibrobot/bin`
  加入 `PATH`。排查「Hermes 不再 source 我的 bashrc」时先看此文件与该开关。
- `hooks/ibrobot-speak`：`post_llm_call` speech hook wrapper，source `.shrc_local` 后
  `exec python3 -m robot_skill_cli.hermes_tts_hook`；TTS 服务名与超时来自 `robot_config` SSOT。

`--accept-hooks` 先 `hermes hooks revoke` 清理旧 mtime，再用
`hermes --accept-hooks hooks doctor` 重新批准；首次安装无既有审批时，revoke 的非零退出经
`hermes hooks list` 确认未注册后跳过，不会阻断审批。

## 生效与验证

本地 CLI：

```bash
hermes-robot --config-name so101_single_arm -- --cli
```

飞书使用长期运行的 Hermes Gateway。同步配置或 Skill 后必须重启，并在飞书中发送 `/new` 创建新会话：

```bash
hermes gateway restart
hermes gateway status
hermes hooks list
hermes hooks doctor
```

动作请求的预期流程为：展示并 flush 冻结计划，内部调用 `confirm-plan` 绑定 exact tuple，然后立即
`execute-plan`，不等待用户再次回复“确认”。这不会绕过 `authorize_motion`、Gateway 校验或停止能力。

最终自然语言回复通过 `post_llm_call` 自动调用 `/voice_tts/synthesize` 和 `/voice_tts/play`，本地扬声器
播放结果。诊断日志位于 `/tmp/hermes-speak.log`。

## 升级

每次更新并重新构建 IB-Robot 后再次执行同一同步命令。同步是幂等的，不会重复添加 hook、Skill 或
SOUL 托管区块。配置变化后必须重启 Gateway；仅使用本地 CLI 时重新启动 CLI 会话即可。
