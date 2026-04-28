---
name: ibrobot-hdc
description: "Handles stable OpenHarmony HDC-over-TCP access to the Bearkey BQ3588HM board. Use when user wants to 'hdc shell', 'connect board', 'send file to OH', 'pull file from OH', 'OpenHarmony device', 'BQ3588HM', '网络调试', '推送文件', '拉取文件', or run commands on the flashed board."
---

# IB-Robot OpenHarmony HDC Skill

This skill standardizes all host-to-board interactions through the local SDK `hdc` binary and the fixed TCP target for the current BQ3588HM board.

For board-local runtime facts such as ROS bootstrap, Python 3.12 availability, or the read-only-root workaround around `ros2ohos.env`, use `ibrobot-bq3588hm-oh`.

## Fixed Endpoint (Current Lab Setup)

Use these exact values unless the user says the board or SDK path changed:

```bash
HDC_BIN=/home/xqw/Research/IB_Robot_dev_worktree/tmp/openharmony/sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710
```

Always call `"$HDC_BIN"` explicitly instead of relying on `$PATH`.

## Core Rule: Prefer TCP, Not USB

For this board, USB HDC sessions are known to become unstable after large transfers and can leave stale server-side sessions. Prefer the network target `192.168.136.111:8710` for all automation, shells, and file transfers.

## Standard Execution Patterns

### Check or establish the connection

```bash
"$HDC_BIN" list targets
"$HDC_BIN" tconn "$HDC_TARGET"
"$HDC_BIN" list targets
```

TCP targets require `tconn`; USB targets appear automatically.

### Run a one-off remote command

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell pwd
"$HDC_BIN" -t "$HDC_TARGET" shell 'ls -1 /data'
"$HDC_BIN" -t "$HDC_TARGET" shell 'cd /data && tar -zxpvf ohos-humble-build-aarch64-20260115100449.tar.gz'
```

### Open an interactive shell

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell
```

### Send files to the board

```bash
"$HDC_BIN" -t "$HDC_TARGET" file send <local_path> <remote_path>
"$HDC_BIN" -t "$HDC_TARGET" file send ./artifact.tar.gz /data/artifact.tar.gz
```

### Receive files from the board

```bash
"$HDC_BIN" -t "$HDC_TARGET" file recv <remote_path> <local_path>
"$HDC_BIN" -t "$HDC_TARGET" file recv /data/result.log ./result.log
```

### Stream device logs

```bash
"$HDC_BIN" -t "$HDC_TARGET" hilog
```

### Reboot or change daemon mode

```bash
"$HDC_BIN" -t "$HDC_TARGET" target boot
"$HDC_BIN" -t "$HDC_TARGET" tmode port 8710
"$HDC_BIN" -t "$HDC_TARGET" tmode port close
```

Note: `tmode port 8710` is usually performed once over USB to move the daemon into TCP mode. After that, prefer Ethernet or Wi-Fi plus `tconn`.

## Recommended Automation Pattern

For reproducible agent actions, wrap every command with the fixed binary and target:

```bash
HDC_BIN=/home/xqw/Research/IB_Robot_dev_worktree/tmp/openharmony/sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710
"$HDC_BIN" -t "$HDC_TARGET" shell '<command>'
```

Examples:

```bash
HDC_BIN=/home/xqw/Research/IB_Robot_dev_worktree/tmp/openharmony/sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710
"$HDC_BIN" -t "$HDC_TARGET" shell 'cd /data && ls'
"$HDC_BIN" -t "$HDC_TARGET" file send ./local.tar.gz /data/local.tar.gz
```

## Stability Recovery

If TCP automation stops responding:

```bash
"$HDC_BIN" kill -r
"$HDC_BIN" tconn "$HDC_TARGET"
"$HDC_BIN" list targets -v
```

If the device is still reachable but the session is stale, remove and reconnect the TCP target:

```bash
"$HDC_BIN" tconn "$HDC_TARGET" -remove
"$HDC_BIN" tconn "$HDC_TARGET"
```

If USB is available for recovery, use it only to re-enable TCP mode:

```bash
"$HDC_BIN" shell ifconfig
"$HDC_BIN" tmode port 8710
"$HDC_BIN" tconn "$HDC_TARGET"
```

## Board-Specific Notes

- Current board: Bearkey BQ3588HM
- Current target: `192.168.136.111:8710`
- Current local SDK tool: `/home/xqw/Research/IB_Robot_dev_worktree/tmp/openharmony/sdk/toolchains/hdc`
- Verified remote paths of interest: `/data`, `/system`, `/vendor`
- Current `/data` already contains:
  - `ohos-18-sysdeps-aarch64-20260115.tar.gz`
  - `ohos-humble-build-aarch64-20260115100449.tar.gz`

## When to Use This Skill

Invoke this skill when the user wants to:
- connect to the flashed OpenHarmony board
- run `hdc shell`
- push or pull files with `hdc`
- inspect `/data`, logs, or services on the board
- prepare the board for ROS 2 or IB-Robot runtime validation

Do NOT use this skill for:
- local workspace builds (`ibrobot-build`)
- local ROS 2 environment setup (`ibrobot-env`)
- launching the Ubuntu-side robot stack (`ibrobot-launch`)

## Quick Reference

| Task | Command |
|------|---------|
| List targets | `"$HDC_BIN" list targets` |
| Connect TCP target | `"$HDC_BIN" tconn "$HDC_TARGET"` |
| Remote shell command | `"$HDC_BIN" -t "$HDC_TARGET" shell '<cmd>'` |
| Interactive shell | `"$HDC_BIN" -t "$HDC_TARGET" shell` |
| Send file | `"$HDC_BIN" -t "$HDC_TARGET" file send <local> <remote>` |
| Receive file | `"$HDC_BIN" -t "$HDC_TARGET" file recv <remote> <local>` |
| Show logs | `"$HDC_BIN" -t "$HDC_TARGET" hilog` |
