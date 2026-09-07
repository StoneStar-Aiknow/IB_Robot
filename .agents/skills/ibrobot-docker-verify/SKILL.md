---
name: ibrobot-docker-verify
description: "Execute setup.sh + build.sh in a clean Ubuntu 22.04 ROS desktop-full Docker container with a host pip cache mount and stage timing. Use when the user explicitly asks for Docker/setup/build verification, or after an author-side PR gate asks WIP vs review and the user confirms review-ready. Skip author-side [WIP] PRs; do not run automatically during PR review."
---

# IB-Robot Docker Verification Skill

Full end-to-end validation of `setup.sh` and `build.sh` inside a clean Ubuntu
22.04 container based on ROS 2 Humble desktop-full. Local testing may copy the
current workspace, while PR evidence must use an isolated snapshot of one
committed Git tree. Both modes remove host-built artifacts before creating a
fresh venv and colcon workspace.

The ROS-ready image deliberately skips ROS first-install testing. Changes to
`scripts/install_ros.sh`, ROS repository setup, or ROS GPG-key handling require
the [bootstrap variant](references/bootstrap-variant.md) based on plain `ubuntu:22.04`.
When such a change **switches the ROS package source** (repo URL, mirror, or
GPG key) and verification must reuse a container that already has ROS packages
installed (desktop-full image or a quick-run container), remove ALL ROS
packages first per Phase 3.5 in
[references/procedure.md](references/procedure.md) — pre-installed packages
belong to the old source and would be silently reused by `setup.sh`'s ROS
detection, making the new-source install path untested.

## When to Use

- User explicitly requests "Docker 验证" / "container test" / "实际验证 setup/build".
- An author-side PR creation/update workflow (`atomgit-pr` or `ibrobot-git-flow`)
  triggers the dependency/setup verification gate, the user confirms the PR is
  ready for reviewer inspection, and real results are needed for the description.
- The current task is to validate local changes to `scripts/setup.sh`,
  `scripts/setup/platforms/*.sh`, `scripts/setup/verify_env.sh`,
  `scripts/install_ros.sh`, or pip/apt dependency resolution.
- Do not infer this skill from PR review alone.
- Before an author-side gate invokes this skill, ask whether the PR is WIP or
  review-ready — via the interactive ask-user tool (the `question` tool in
  opencode), not a prose question and not an assumed default. A `[WIP]` PR skips both Docker skills until promotion; explicit
  standalone Docker requests still run normally.

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

## Tree Binding for PR Verification

When an author-side PR gate triggers both Docker skills, verification evidence
must describe one exact committed tree:

1. Resolve the target commit and record
   `VERIFIED_TREE="$(git rev-parse "${VERIFIED_COMMIT}^{tree}")"` before the
   first platform. The user's current worktree may contain unrelated changes.
2. Materialize an isolated standalone snapshot of `VERIFIED_COMMIT`; Ubuntu and
   openEuler must both test snapshots whose tree equals `VERIFIED_TREE`.
3. Include `Verified tree: <full SHA>` in this skill's result. The PR workflow
   assembles it into the structured `## Docker Verification` block.
4. At PR creation/update, compare the field with the remote head commit's tree.
   Re-run only when that tree changes. Commit-message, author, or trailer-only
   rewrites retain the same tree and do not invalidate the result.
5. Directly copying a dirty workspace remains valid for local diagnosis, but
   that result must not be used as PR verification evidence.

## Prerequisites

- Docker CLI installed on the host (check with `command -v docker`).
- Git LFS installed on the host for PR-evidence snapshots; the procedure runs
  `git lfs pull` and `git lfs fsck` so a clean tree cannot hide pointer-only files.
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
| **3.5** | ROS source-switch pre-clean (conditional) — remove all `ros-*` packages when the change switches the ROS source | No ROS packages left |
| **4** | Run `setup.sh --yes`, capture log and timing | `/tmp/setup.log`, `SETUP_ELAPSED_SECONDS` |
| **5** | Run `build.sh`, capture log and timing | `/tmp/build.log`, `BUILD_ELAPSED_SECONDS` |
| **6** | Collect ERROR lines, report timing, clean up container | Final report |

### Phase 3 Source Modes

- **PR evidence — isolated committed snapshot**: create a standalone snapshot
  of the target commit outside the user's worktree, verify its tree SHA, then
  `docker cp` it into the container. Required for author-side PR gates.
- **Local diagnosis — current workspace copy**: preserves uncommitted and
  untracked changes. It cannot produce reusable PR evidence.
- **Explicit remote clone**: use only when the user specifically requests a
  remote commit or branch; resolve and report the exact commit and tree.

Record the selected mode and source details in the verification result and in
the PR description's Verification section.

## Variants

- **ROS installation changes** (modify `install_ros.sh`, ROS repo, GPG key):
  use the [bootstrap variant](references/bootstrap-variant.md) with
  `ubuntu:22.04` base image. If the change switches the ROS source and you
  keep the desktop-full image instead, the Phase 3.5 ROS pre-clean in
  [references/procedure.md](references/procedure.md) is **mandatory**.
- **Iterative local testing** (re-run without full Phase 0-2):
  use the [quick-run one-liner](references/quick-run.md). If the ROS source
  changed between runs, re-run the Phase 3.5 ROS pre-clean before `setup.sh`.

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
