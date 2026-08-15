# Verification Discipline

> **核心原则：setup.sh 和 build.sh 是唯一合法的软件安装途径。**

## 禁止事项

- **禁止手动安装软件包**：不允许在容器中手动执行 `apt install`、`pip install` 来安装 ROS、Python 包、系统依赖等。所有软件安装必须通过 `setup.sh` 完成。手动安装会绕过脚本逻辑，使验证结果失去意义。
- **禁止手动 patch 脚本**：不允许用 `sed`、`cp` 等方式修改容器内的 `install_ros.sh`、`setup.sh` 或平台脚本。如果脚本存在网络适配缺陷（如 GPG key URL 不可达、pip 镜像未传递），应修复脚本本身并提交到代码仓库，而不是在验证时临时 patch。
- **禁止手动配置 ROS 仓库或 GPG key**：ROS 安装必须由 `install_ros.sh` 完成。

## 允许事项

- **配置 apt 镜像源**：允许在容器创建阶段将 Ubuntu 源切换到 Aliyun、
  ROS 2 源切换到 TUNA。这属于容器环境初始化。
- **创建用户和权限**：允许创建 `testuser`、配置 `NOPASSWD` sudo。
- **配置 locale 和时区**。
- **通过环境变量传递镜像 URL**：允许通过 `PIP_INDEX_URL`、`ROS_GPG_KEY` 等环境变量让脚本使用镜像源。这不是"手动安装"，而是让脚本的 `${VAR:-default}` 机制正确工作。
- **挂载宿主机 pip cache**：只复用下载缓存，禁止挂载宿主机 `venv`、
  `build`、`install` 或整个源码工作区。默认流程使用 `docker cp` 复制源码，
  不是 bind mount，并在容器内删除宿主机构建产物。
- **挂载宿主机 CUDA toolkit（可选）**：GraspGen 的 `pointnet2_ops` CUDA 扩展
  需要 `nvcc` 编译，CUDA toolkit 目录以只读方式挂载到容器中同一路径（不是
  源码工作区的一部分，属于构建工具），并设置 `CUDA_HOME` / `PATH` /
  `LD_LIBRARY_PATH` 环境变量。系统目录（`/`、`/usr`、`/usr/local`）被拒绝挂载。
  无 CUDA toolkit 时 setup.sh 优雅跳过 grasp 安装，验证继续。
- **缓存权限必须以目标用户验证**：root 身份执行 `test -w` 没有意义。必须以
  `testuser` 实际在 `PIP_CACHE_DIR` 创建并删除探针文件，失败时立即终止验证。

## 错误分类与报告

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告（如 numpy 版本冲突）；rosdep keys 未解析（自定义包不在 rosdistro）；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。
