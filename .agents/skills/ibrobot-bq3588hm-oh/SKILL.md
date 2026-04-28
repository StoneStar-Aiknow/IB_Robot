---
name: ibrobot-bq3588hm-oh
description: "Captures the verified OpenHarmony runtime facts for the Bearkey BQ3588HM board. Use when user wants to bootstrap ROS on-device, source ros2ohos.env, check Python, inspect /data/install or /data/out, or debug the read-only-root workaround on the board."
---

# IB-Robot BQ3588HM OpenHarmony Board Skill

This skill captures the current verified runtime state of the Bearkey BQ3588HM OpenHarmony board.

Use `ibrobot-hdc` for connectivity and file transfer. Use this skill for board-local runtime facts, ROS bootstrap, Python availability, and known platform quirks.

## Current Verified Board State

- Board: Bearkey BQ3588HM
- OpenHarmony access path: HDC over TCP to `192.168.136.111:8710`
- HDC binary: `/home/xqw/Research/IB_Robot_dev_worktree/tmp/openharmony/sdk/toolchains/hdc`
- Official runtime archives currently on the board:
  - `/data/ohos-humble-build-aarch64-20260115100449.tar.gz`
  - `/data/ohos-18-sysdeps-aarch64-20260115.tar.gz`
- Extracted ROS runtime roots:
  - `/data/install`
  - `/data/out`
- ROS environment file:
  - `/data/ros2ohos.env`
- Local board patch already applied:
  - `/data/sysdeps.env` now inserts `mount -o remount,rw /` before the SSH setup `mkdir`

## Core Runtime Bootstrap

The official documentation is directionally correct: stay in `/data` and source `./ros2ohos.env`.
On this board, the local `/data/sysdeps.env` has already been patched so that it remounts `/` read-write before the SSH setup branch creates directories on the root filesystem.

### Preferred Verified Pattern

```sh
cd /data
. ./ros2ohos.env
ros2 topic list
python3 --version
python3 -m pip --version
mount -o remount,ro /
```

Keep all commands in the same shell session. Remount `/` back to read-only after the commands that depend on the sourced environment have completed.

### Fallback Pattern

If remounting `/` is undesired in a specific session, the SSH setup branch can still be bypassed by temporarily moving the sysdeps `sshd_config` file:

```sh
cd /data
mv out/etc/sshd_config out/etc/sshd_config.disabled
. ./ros2ohos.env
ros2 topic list
python3 --version
python3 -m pip --version
mv out/etc/sshd_config.disabled out/etc/sshd_config
```

## Why Remounting Matters

The originally installed `/data/sysdeps.env` did **not** contain a `mount -o remount,rw /` line before its SSH setup branch. The board-local file has since been patched so that the relevant part is now:

```sh
if [ -f "${OHOS_ROS2_SYSDEPS}/etc/sshd_config" ]; then
    mount -o remount,rw /
    mkdir -p /var/empty /var/run /root/.ssh /libexec
    ...
fi
```

On this board, the root filesystem starts as read-only for those paths, so the `mkdir` operations would abort the environment setup before ROS and Python are fully prepared unless `/` is remounted read-write first.

The author-suggested remount path is valid in practice: `mount -o remount,rw /` works on this board, and after patching `sysdeps.env` the plain `source ./ros2ohos.env` path succeeds without the `sshd_config` workaround.

Temporarily moving `/data/out/etc/sshd_config` out of the way remains a fallback that avoids modifying the official scripts when remounting `/` is not desired.

## Python Environment on the Board

The board does **not** provide a ready-to-use global `python`, `python3`, or `pip` before the ROS/sysdeps environment is loaded.

After sourcing the environment successfully:

- `python3` resolves to `/data/out/bin/python3`
- Real interpreter: `/data/out/bin/python3.12`
- Verified version: `Python 3.12.12`
- `pip` is available via:

```sh
python3 -m pip --version
```

Verified output:

```text
pip 25.1.1 from /data/out/lib/python3.12/site-packages/pip (python 3.12)
```

## Verified ROS Runtime Signals

After sourcing with the workaround, `ros2 topic list` succeeds on the board and currently shows at least:

- `/joint_states`
- `/parameter_events`
- `/robot_status/ee_pose`
- `/rosout`
- `/tf_static`

## Scope Boundary

Use this skill when the user wants to:

- bootstrap ROS on the BQ3588HM board
- source `ros2ohos.env`
- check on-device Python or pip
- inspect `/data/install` or `/data/out`
- understand the read-only-root workaround
- confirm what is already installed on this board

Do **not** use this skill for:

- HDC transport and reconnect logic (`ibrobot-hdc`)
- local workspace builds (`ibrobot-build`)
- local Ubuntu ROS environment setup (`ibrobot-env`)

## Practical Consequence for IB_Robot

The board now has:

- official OpenHarmony ROS runtime unpacked
- working on-device `ros2`
- working on-device Python 3.12 from sysdeps

The next major step is **not** rebuilding the board image again.
It is cross-building and deploying the minimum IB_Robot ROS packages plus Python dependencies that must run on top of this board runtime.
