# Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Couldn't resolve host name` | rootfs missing `/etc/resolv.conf` | Phase 2.3 copies from host container |
| `Config error: File exists: /var/log` | `/var/log` symlink target missing | Phase 2.4 creates `/var/volatile/log` |
| `/dev/stdout: No such file or directory` | 命令依赖了 chroot 中不存在的设备伪文件 | 将验证输出重定向到 rootfs 内普通文件，不要挂载宿主 `/dev` |
| `mount: command not found` / `mountpoint: command not found` | `:env` 镜像外层不提供挂载工具 | setup/build 验证不需要挂载；删除相关基础设施命令 |
| `dubious ownership in repository` | UID mismatch after `docker cp` | Phase 2.5 adds `safe.directory` |
| `gpg.errors.GPGMEError` during `rosdep install` | qemu-aarch64 emulation bug with Python `gpg` | setup.sh 自动禁用 `gpgcheck` |
| `git-lfs was not found` post-checkout hook | No git-lfs in rootfs | `lerobot_patches.sh` auto-removes hook when git-lfs missing |
| `ERROR: file:///root/IB_Robot/libs/lerobot does not appear to be a Python project` | Copied a linked worktree or an uninitialized submodule tree into the container | Use a standalone clone and run `git submodule update --init --recursive` before `docker cp` |
| `pip3 not found, cannot install colcon` | `platform_install_python_bootstrap` not called before `ensure_colcon` | `install_system_deps` calls bootstrap first |
| `python%{python3_pkgversion}-scipy` not found | `ROS_OS_OVERRIDE=rhel:8` uses RHEL naming; openEuler dnf can't match macro | Platform script skips `python3-scipy` in rosdep, installs via explicit `dnf install` |
| `rosdep install failed` for missing packages | Some ROS packages not in openEuler repos (e.g. `robot_localization`) | Platform script uses non-fatal rosdep + skip-keys |
| dnf outputs config dump instead of installing | Running dnf without `--nogpgcheck --setopt=strict=0` in chroot | Always use `dnf install -y --nogpgcheck` |
| `which git` returns nothing despite git being installed | `which` binary in rootfs behaves incorrectly under qemu-user | Use `type git` or absolute path `/usr/bin/git` instead |
| `ReadTimeoutError: files.pythonhosted.org` during pip install | Network instability under qemu-user emulation | Set `export PIP_DEFAULT_TIMEOUT=120` before running setup.sh |
