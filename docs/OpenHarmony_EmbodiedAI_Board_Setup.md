# OpenHarmony EmbodiedAI 1.0.1 开发板烧录与调试全指南

## 第一阶段：HDC 调试工具准备
HDC (Hardware Device Connector) 是与 OpenHarmony 设备交互的核心工具。在烧录前，建议先在主机上准备好该工具。

1.  **下载全量 SDK：**
    * 访问每日构建 (DailyBuild) 页面：[OpenHarmony DailyBuild](https://dcp.openharmony.cn/workbench/cicd/dailybuild/detail/component)。
    * **项目选择**：`openharmony`；**下载包**：`ohos-sdk-full`。
2.  **获取 HDC 工具：**
    * 解压下载的 SDK 后，从其中的 `toolchains` 目录获取 HDC。
    * Windows 主机通常使用 `windows/toolchains/hdc.exe`。
    * Linux / Ubuntu 主机通常使用 `linux/toolchains/hdc`，或解压后的 `<sdk-root>/toolchains/hdc`。
    * 本仓库当前实验环境验证过的 Linux 路径示例：`<hdc_bin_path>`。
3.  **运行方式 (二选一)：**
    * **方案 A (推荐)：** 配置全局环境变量。把 SDK 的 `toolchains` 目录加入 `PATH`，并写入 shell 启动文件持久化。
      * Bash 示例：`echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.bashrc && source ~/.bashrc`
      * Zsh 示例：`echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.zshrc && source ~/.zshrc`
      * 完成后应能直接执行：`hdc list targets`
    * **方案 B (快捷)：** 无需配置环境。直接在 `toolchains` 文件夹空白处，按住 `Shift` 并右键，选择“在此处打开 PowerShell/终端”即可就地使用。

## 第二阶段：高可靠网络调试 (TCP 模式)
为避免大文件传输导致的 USB 僵尸会话，建议切换到局域网 TCP 调试。

1.  **开启监听：** 在 USB 连接状态下执行 `hdc tmode port 8710`。
2.  **获取 IP：** 执行 `hdc shell ifconfig` 查看开发板当前局域网 IP。
3.  **远端连接：** 执行 `hdc tconn <board-ip>:8710`。
4.  **操作习惯：** 建议使用 `hdc -t <board-ip>:8710 shell` 明确指定目标设备，确保自动化交互的稳定性。

## 第三阶段：SSH 登录与公钥配置

在当前实验环境中，开发板已经验证过可通过 SSH 登录；推荐先用 HDC/TCP 完成初始化，再切换到 SSH 做日常操作。

### 1. 连接前提

1. 先确认开发板已经联网，并能通过 HDC/TCP 访问。
2. 通过 HDC 推送并执行 SSH 配置脚本：

   ```sh
   # 主机侧
   hdc -t <board-ip>:8710 file send scripts/setup_sshd.sh /data/setup_sshd.sh
   hdc -t <board-ip>:8710 shell 'sh /data/setup_sshd.sh'
   ```

   脚本会自动完成 remount rw、拷贝 sshd 配置、生成 host key、启动 sshd。

   > **注意**：OpenHarmony 板端无 systemd，sshd 无法开机自启。每次重启后需先通过 HDC shell 执行一次 `sh /data/setup_sshd.sh` 重新启动 sshd。公钥只需首次配置一次，重启后仍然有效。
3. 获取开发板局域网 IP 后，从主机登录：

   ```sh
   ssh root@<board-ip>
   ```

### 2. 公钥配置（可选）

如需免密登录，通过 HDC 上传公钥：

```sh
hdc -t <board-ip>:8710 file send ~/.ssh/id_ed25519.pub /data/local/id_ed25519.pub
hdc -t <board-ip>:8710 shell 'mkdir -p /root/.ssh && cat /data/local/id_ed25519.pub >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys'
```

### 4. 建议的使用习惯

- **大文件传输 / 自动化脚本**：优先使用 HDC/TCP。
- **日常命令行操作 / 多终端调试**：优先使用 SSH。
- 如果 SSH 临时失效，可先回到 HDC 执行：

  ```sh
  cd /data
  . ./robooh_1.0.1.env
  ```

  然后再重新尝试 `ssh root@<board-ip>`。
