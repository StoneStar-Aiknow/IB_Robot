# Verification Discipline

> **核心原则：setup.sh 和 build.sh 是唯一合法的软件安装途径。**

## 禁止事项

- **禁止手动安装软件包**：不允许在容器中手动执行 `dnf install`、`pip install` 来安装 ROS、Python 包、系统依赖等。所有软件安装必须通过 `setup.sh` 完成。手动安装会绕过脚本逻辑，使验证结果失去意义。
- **禁止手动 patch 脚本**：不允许用 `sed`、`cp` 等方式修改容器内的 `install_ros.sh`、`setup.sh`、平台脚本或 `lerobot_patches.sh`。如果脚本存在网络适配缺陷（如 pip 镜像未传递），应修复脚本本身并提交到代码仓库，而不是在验证时临时 patch。
- **禁止手动配置 ROS 仓库或 GPG key**：ROS 安装必须由 `install_ros.sh` 完成。

## 允许事项

- **修复 chroot 基础设施**：允许修复 DNS（`resolv.conf`）、`/proc`、`/sys`、`/var/log`、`git safe.directory` 等 qemu-user chroot 环境缺陷。这属于容器环境初始化，不是软件安装。
- **通过环境变量传递镜像 URL**：允许通过 `PIP_INDEX_URL`、`PIP_TRUSTED_HOST`、`ROS_GPG_KEY` 等环境变量让脚本使用镜像源。这不是"手动安装"，而是让脚本的 `${VAR:-default}` 机制正确工作。**但必须**：(1) 在验证报告中明确列出每个设置的环境变量及其值；(2) 在 PR 描述的 Verification 小节中提及这些环境变量，让审阅者知道容器网络环境与默认值的差异。
- **修复原始镜像已损坏的包状态（RPM DB 与磁盘内容不一致）**：当 `:env` 镜像自带状态不一致的基础包（RPM 数据库记录版本 X，磁盘实际为版本 Y）时，允许在 chroot 中用 `dnf reinstall -y --nogpgcheck <pkg> [<pkg>-devel]` 重装受影响包以恢复一致性。这属于环境修复，不是"安装新软件"——包已经在 RPM DB 中声明，`reinstall` 只是把磁盘内容重新对齐到数据库记录，不引入新的依赖、不改变 RPM DB 状态。典型场景：`lz4` 的 RPM DB 标称 `1.9.4` 但磁盘实际为 `1.10.0`，导致 setup.sh 安装 `lz4-devel` 时创建指向不存在的 `liblz4.so.1.9.4` 的符号链接。完整流程见 [procedure.md](procedure.md) 的 Phase 3.5。检测必须在 setup.sh 启动前完成，否则后续 dnf 事务可能因基础层不一致而失败。

## 错误分类与报告

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告（如 numpy 版本冲突）；rosdep keys 未解析（自定义包不在 rosdistro）；updmap 字体映射缺失；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。
