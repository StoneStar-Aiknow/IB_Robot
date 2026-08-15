---
name: ibrobot-docker-verify
description: "Execute setup.sh + build.sh in a clean Ubuntu 22.04 ROS desktop-full Docker container with a host pip cache mount and stage timing. Use when the user explicitly asks for Docker/setup/build verification, or when author-side PR creation/update workflows trigger the dependency/setup verification gate. Do not use automatically during PR review; review should check developer-provided Verification in the PR description."
---

# IB-Robot Docker Verification Skill

Full end-to-end validation of `setup.sh` and `build.sh` inside a clean Ubuntu
22.04 container based on ROS 2 Humble desktop-full. The default flow copies the
current host workspace so uncommitted changes can be tested, while removing
host-built artifacts before creating a fresh venv and colcon workspace. A
clean remote clone is available only when the user explicitly requests it.

The ROS-ready image deliberately skips ROS first-install testing. Changes to
`scripts/install_ros.sh`, ROS repository setup, or ROS GPG-key handling require
the [bootstrap variant](references/bootstrap-variant.md) based on plain `ubuntu:22.04`.

## When to Use

- User explicitly requests "Docker 验证" / "container test" / "实际验证 setup/build".
- An author-side PR creation/update workflow (`atomgit-pr` or `ibrobot-git-flow`)
  triggers the dependency/setup verification gate and needs real results for the
  PR description.
- The current task is to validate local changes to `scripts/setup.sh`,
  `scripts/setup/platforms/*.sh`, `scripts/setup/verify_env.sh`,
  `scripts/install_ros.sh`, or pip/apt dependency resolution.
- Do not infer this skill from PR review alone.

## Review Boundary

- During PR review, do **not** run this skill automatically just because a PR
  touches `package.xml`, `setup.py`, setup scripts, or build files.
- A user asking "review this PR" or "check whether this PR is OK" is not an
  explicit request to run Docker verification.
- PR review should inspect the developer-provided Verification section in the
  PR description. If required Ubuntu verification is missing or incomplete,
  raise a blocking review issue asking the developer to provide it.
- Only run this skill in a review session when the user explicitly asks the
  agent to perform the actual Ubuntu Docker setup/build verification.

## Prerequisites

- Docker CLI installed on the host (check with `command -v docker`).
- The current user has permission to run containers.
- The IB-Robot workspace has uncommitted or committed changes to validate.
- If the user explicitly requests a remote commit or branch, its repository is
  reachable from the container.
- The host pip cache directory exists at `${PIP_CACHE_DIR:-$HOME/.cache/pip}`.
- The container user must use the host's actual `id -u` and `id -g`. Never
  hard-code `1000:1000`; doing so can make the bind-mounted pip cache
  inaccessible even when the directory exists.
- Network access to Aliyun apt mirror, TUNA ROS 2 repo, Huawei pip mirror,
  and `gitcode.com` / `atomgit.com` for lerobot submodule fetch.
- **NVIDIA GPU + CUDA toolkit on the host (optional)**: GraspGen's
  `pointnet2_ops` CUDA extension is installed by default when a CUDA toolkit
  (`nvcc`) is available. If the host has no CUDA toolkit, `setup.sh` will
  warn and skip GraspGen install gracefully; the verification continues
  without the grasp smoke test.
  - Host CUDA toolkit with `nvcc` on PATH or at `/usr/local/cuda/bin/nvcc`.
    The toolkit directory is mounted read-only into the container at the same
    path, and `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` are set accordingly.
    System directories (`/`, `/usr`, `/usr/local`) are rejected to prevent
    accidental bind-mount of the entire system.
  - `--gpus` passthrough is NOT required for setup+build verification:
    `pointnet2_ops` compilation uses `TORCH_CUDA_ARCH_LIST` (fixed, not
    GPU-detected) and the smoke test only does `importlib.util.find_spec`.

## Core Principle

> **setup.sh 和 build.sh 是唯一合法的软件安装途径。**

- 禁止手动 `apt install` / `pip install` / `sed` patch 脚本 / 手动配置 ROS 仓库。
- 允许配置 apt 镜像源、创建用户、通过环境变量传递镜像 URL、挂载 pip cache。
- 完整的禁止/允许事项清单和论述见 [references/discipline.md](references/discipline.md)。

## Error Classification

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告；rosdep keys 未解析；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。

## Container Naming Convention

| Variable      | Value                        |
|---------------|------------------------------|
| Container name | `verify-ubuntu2204`          |
| User           | `testuser`                   |
| Workspace      | `/home/testuser/IB_Robot`    |
| Image          | `osrf/ros:humble-desktop-full-jammy` |
| Pip cache      | `/var/cache/ibrobot-pip`      |

## Procedure Overview

Each phase's complete bash commands and detailed rationale live in
[references/procedure.md](references/procedure.md). The summary below is for
routing and mental model only; execute the actual commands by reading the
corresponding phase section.

| Phase | Purpose | Key Output |
|-------|---------|------------|
| **0** | Check host prerequisites (docker, uid, pip cache) | `PROJECT_ROOT`, `HOST_UID`, `HOST_PIP_CACHE` |
| **1** | Pull `osrf/ros:humble-desktop-full-jammy` image | `IMAGE_SECONDS` |
| **2** | Create container, configure mirrors, create `testuser`, verify cache write | `CONTAINER_SECONDS` |
| **3** | Prepare source workspace — **choose one mode** | `SOURCE_MODE`, `SOURCE_DETAILS` |
| **4** | Run `setup.sh --yes`, capture log and timing | `/tmp/setup.log`, `SETUP_ELAPSED_SECONDS` |
| **5** | Run `build.sh`, capture log and timing | `/tmp/build.log`, `BUILD_ELAPSED_SECONDS` |
| **6** | Collect ERROR lines, report timing, clean up container | Final report |

### Phase 3 Source Modes

- **Default — local workspace copy**: `docker cp` the current host workspace
  into the container, then `rm -rf venv build install log`. Preserves
  uncommitted and untracked changes.
- **Optional — remote clean clone**: only when the user explicitly asks to
  validate a pushed commit/branch. Replace placeholders with exact remote and
  branch; never default silently.

Record the selected mode and source details in the verification result and in
the PR description's Verification section.

## Variants

- **ROS installation changes** (modify `install_ros.sh`, ROS repo, GPG key):
  use the [bootstrap variant](references/bootstrap-variant.md) with
  `ubuntu:22.04` base image.
- **Iterative local testing** (re-run without full Phase 0-2):
  use the [quick-run one-liner](references/quick-run.md).

## Troubleshooting

Common issues (sudo tty, rosdep permission, pip cache disabled, TUNA 404, etc.)
are documented in [references/pitfalls.md](references/pitfalls.md). Consult
that table when a verification run fails with an unfamiliar error.

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch in container |
| `DEBIAN_FRONTEND` | `noninteractive` | Prevent tzdata etc. from blocking |
| `PIP_CACHE_DIR` | `/var/cache/ibrobot-pip` | Reuse host pip downloads without reusing the venv |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |
