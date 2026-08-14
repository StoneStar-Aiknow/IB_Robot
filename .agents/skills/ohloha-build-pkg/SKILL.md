---
name: ohloha-build-pkg
description: "Cross-compile third-party packages for OpenHarmony aarch64 using tools_ohloha_pkgs (ohloha package manager). Use when user mentions `tools_ohloha_pkgs`, `builder.sh`, `ohloha`, `pkgs-create.sh`, `编译 bash/zsh/vim 到板端`, `交叉编译第三方库到 OH`, `OHOS_SDK`, `ohloha_pkgs BUILD`, `dist.aarch64`, or needs to build non-ROS packages (bash, zsh, vim, ncurses, etc.) for the OpenHarmony board. Triggers for 'BQ3588HM', 'RoboPi'."
---

# ohloha 包交叉编译 (tools_ohloha_pkgs)

使用 `tools_ohloha_pkgs` 框架为 OpenHarmony aarch64 交叉编译第三方包（bash、zsh、vim、ncurses 等）。

## 适用场景

- 用 `builder.sh` 编译 `tools_ohloha_pkgs` 仓库中的包
- 用 `pkgs-create.sh` 创建新的包构建规格
- 交叉编译 shell / 编辑器 / 库到板端
- 交叉编译 Python C 扩展（regex, numpy 等）到板端
- 交叉编译 Rust/Python 混合包（tokenizers 等）到板端
- 部署编译产物到 BQ3588HM 板子

## 不适用

- IB_Robot 自有包（用 `oh-build-roboframe`）
- 第三方 ROS 2 包（用 `oh-cross-build-ros-pkg`）
- 内核编译（用 `oh-rebuild-kernel`）

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| 常见踩坑（config.h/autoheader/模块禁用/yodl/Python C 扩展/Rust+Python）及参考 BUILD | `references/pitfalls.md` |
| Rust/Python 混合包的 maturin/PyO3 交叉编译 BUILD 模板 | `references/rust-python-template.md` |

Do not expose these references as separate skills.

## ⚠️ 构建前必做检查：先查 ohloha 源码是否已支持

**在编译构建任何包之前，必须先拉取最新的 `tools_ohloha_pkgs` 源码，检查仓库中是否已经支持目标包。** 避免重复造轮子。

### 检查步骤

1. **拉取最新代码**：

   ```bash
   cd <ohloha_pkgs_dir>
   git pull
   ```

2. **搜索目标包是否已存在**：

   ```bash
   # 按包名搜索 BUILD 文件
   ls */BUILD 2>/dev/null | grep -i <包名>
   # 或搜索目录
   find . -maxdepth 2 -iname "*<包名>*" -type d
   ```

3. **如果已存在**：直接用 `builder.sh <包名>/BUILD` 构建，无需新建。检查已有 BUILD 文件的版本是否满足需求，如需升级版本，在原 BUILD 上修改而非新建。

4. **如果不存在**：需要新增包。新增包时**必须遵循 ohloha 源码仓库中的规范**：
   - 阅读 ohloha 仓库根目录的 `README.md`，了解包的命名约定、BUILD 文件格式、目录结构要求
   - 阅读 ohloha 仓库 `.agents/skills/` 目录下的 skill 文档，这些 skill 专门描述了如何为 ohloha 新增包、BUILD 字段语义、hook 编写规范等
   - 使用 `./pkgs-create.sh <包名>` 生成初始 BUILD 模板，再按规范填充字段

### ohloha 源码仓库中的 skill 位置

```
tools_ohloha_pkgs/
├── README.md                    # 仓库说明和快速上手
├── .agents/skills/              # ohloha 专属 skill 文档
│   └── ...                      # 新增包、BUILD 编写、hook 使用等规范
├── pkgs-create.sh               # 创建新包脚手架
├── builder.sh                   # 构建入口
└── <已有包>/
    └── BUILD                    # 已有包的构建规格（参考示例）
```

新增包前，优先参考仓库中**已有的同类包的 BUILD 文件**作为模板（例如新增 shell 类包参考 `bash/BUILD` 或 `zsh/BUILD`），再查阅 `.agents/skills/` 中的规范文档补充细节。

## ⚠️ 构建前置条件

### 1. OHOS_SDK 环境变量

必须设置 `OHOS_SDK` 指向 OpenHarmony SDK 根目录（含 `native/llvm/bin/clang` 和 `native/sysroot`）：

```bash
# 已知可用的 SDK 路径（选一个）:
export OHOS_SDK=/data/oh_build/prebuilts/ohos-sdk/linux/18
# 或
export OHOS_SDK=<oh_build_root>/custom_build_root/ohos-robot-toolchain/18
```

### 2. GNU diff 优先（关键！）

OH SDK toolchains 自带的 `diff`（在 `$OHOS_SDK/toolchains/diff` 或 `<ohos_sdk_dir>/toolchains/diff`）**对所有输入返回 0**，会导致 autoconf 的 `config.status` 误判文件"unchanged"而不创建 `config.h`，编译失败。

**修复**：构建前必须让 GNU diff 优先：

```bash
export PATH=/usr/bin:/bin:$PATH
```

验证：`diff --version` 应显示 `GNU diffutils`，而不是 OH SDK 的版本。

### 3. 代理设置（如需下载源码）

GitHub 源码下载可能需要绕过代理：

```bash
curl -sSL --noproxy '*' -o /tmp/pkg.zip "https://github.com/..."
```

## 构建流程

### 仓库位置

```bash
cd <ohloha_pkgs_dir>
```

### 创建新包

```bash
export OHOS_SDK=<SDK路径>
./pkgs-create.sh <包名>
# 编辑 <包名>/BUILD 填写版本、依赖、URL、构建参数
```

### 构建包

```bash
export OHOS_SDK=<SDK路径>
export PATH=/usr/bin:/bin:$PATH
./builder.sh <包名>/BUILD
# 产物在 dist.aarch64.<包名>/
```

### 构建顺序（依赖拓扑序）

`builder.sh` 不管理依赖图，需手动按拓扑序指定：

```bash
./builder.sh libncursesw/BUILD bash/BUILD   # 先编依赖再编目标
```

## BUILD 文件关键字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `pkg_version` | 版本号 | `"5.9"` |
| `pkg_name` | 包名 | `"zsh"` |
| `pkg_deps` | 运行依赖 | `"libncursesw>=6.0"` |
| `pkg_build_deps` | 构建依赖 | `"libncursesw>=6.0"` |
| `pkg_source_url` | 源码 URL | `"https://github.com/zsh-users/zsh/archive/refs/tags/zsh-5.9.tar.gz"` |
| `pkg_build_type` | 构建系统 | `"autotools"` / `"cmake"` / `"meson"` / `"pure-python"` / `"custom"` |
| `pkg_build_autotools_extra_configure_flags` | configure 参数 | `"--disable-gdbm --enable-multibyte"` |
| `pkg_build_autotools_make_install_target` | make install 目标 | `"install.bin install.modules install.fns"` |

## Hooks（按执行顺序）

| Hook | 时机 | 用途 |
|------|------|------|
| `native_env_hook` | 最先 | 设置 PATH 等环境（如 GNU diff 优先） |
| `prebuilt_patch_once_hook` | 下载后（仅一次） | 打补丁 |
| `prebuilt_patch_hook` | 下载后（每次） | 打补丁 |
| `custom_build` | 构建前 | 自定义构建流程（设 `_custom_build_continue=false` 可跳过默认流程） |
| `postbuilt_hook` | 构建后 | 后处理 |

## 已知踩坑

7 个常见坑及其修复方案详见 `references/pitfalls.md`，覆盖：

1. OH SDK `diff` 损坏 → config.h 不生成
2. `autoheader` 在 make 阶段删除 config.h
3. 交叉编译下模块/功能被禁用
4. GitHub 源码归档不含预生成 configure
5. install.man / install.runhelp 需要 yodl（zsh）
6. Python C 扩展包构建
7. Rust/Python 混合包构建（详见 `references/rust-python-template.md`）

## 部署到板子

Python 包（纯 Python 或 C/Rust 扩展）部署到板端 pysite 目录：

```bash
# 打包
tar -czpf /tmp/<pkg>.tar.gz \
    -C dist.aarch64.python3-<pkg>/lib/python${PY_VERSION}/site-packages \
    <pkg_module> <pkg_module>-<version>.dist-info

# 推送（HDC）
hdc -t <board> file send /tmp/<pkg>.tar.gz /data/<pkg>.tar.gz
hdc -t <board> shell 'cd /data/roboframe/pysite && tar -zxpf /data/<pkg>.tar.gz && rm /data/<pkg>.tar.gz'
```

C/C++ 二进制包部署：

```bash
# 二进制
ssh root@<board> 'cat > /data/<pkg>/bin/<binary>' < dist.aarch64.<pkg>/bin/<binary>
ssh root@<board> 'chmod +x /data/<pkg>/bin/<binary>'

# 目录（functions、lib 等）
tar czf /tmp/<pkg>-extras.tar.gz -C dist.aarch64.<pkg> share
ssh root@<board> 'cat > /data/<pkg>-extras.tar.gz' < /tmp/<pkg>-extras.tar.gz
ssh root@<board> 'cd /data/<pkg> && tar xzf /data/<pkg>-extras.tar.gz && rm /data/<pkg>-extras.tar.gz'
```

## 已构建的包清单

| 包 | 版本 | 类型 | 依赖 | PR |
|----|------|------|------|-----|
| libncursesw | 6.5 | autotools | - | 仓库已有 |
| bash | 5.2 | autotools | libncursesw | [#2](https://gitcode.com/openharmony-robot/tools_ohloha_pkgs/pull/2) |
| vim | 9.1.1989 | autotools | libncursesw, libreadline, libgettext, python3 | 仓库已有（VIMRUNTIMEDIR 修复见 [#3](https://gitcode.com/openharmony-robot/tools_ohloha_pkgs/pull/3)） |
| zsh | 5.9 | autotools | libncursesw | [#4](https://gitcode.com/openharmony-robot/tools_ohloha_pkgs/pull/4) |
| python3-regex | 2026.6.28 | custom (C ext) | python3 | [#6](https://gitcode.com/openharmony-robot/tools_ohloha_pkgs/pull/6) |
| python3-tokenizers | 0.22.2 | custom (Rust) | python3 | 待提交 |
