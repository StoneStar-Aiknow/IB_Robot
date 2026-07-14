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

## 已知踩坑与修复

### 坑 1: OH SDK `diff` 损坏 → config.h 不生成

**症状**：configure 说 `config.h is unchanged` 但 config.h 不存在，make 报 `No such file`。

**根因**：OH SDK 的 `diff` 对所有输入返回 0，config.status 误判"unchanged"跳过创建。

**修复**：`native_env_hook` 里 `export PATH=/usr/bin:/bin:$PATH`。

### 坑 2: `autoheader` 在 make 阶段删除 config.h

**症状**：configure 成功创建 config.h，但 make 运行 `autoheader` 后 config.h 消失。

**根因**：zsh/vim 等 autotools 项目的 Makefile 在 `headers` target 中调用 `autoheader`，交叉编译下重建 config.h 失败导致被删。

**修复**：configure 前 sed 修改 Makefile.in 禁用 autoheader：

```bash
sed -i 's/cd \$(sdir) && autoheader/cd $(sdir) \&\& true/' Makefile.in
```

### 坑 3: 交叉编译下模块/功能被禁用

**症状**：zsh 的 `zsh/regex`、`zsh/system` 模块 `zmodload` 失败；configure 检测不到某些功能。

**根因**：交叉编译下 configure 无法运行 run-test（dlopen、功能探测等），相关模块被设为 `link=no` 或功能被 `#undef`。

**修复**：configure 后手动修改生成文件：
- zsh: `sed -i 's/link=no/link=static/' config.modules`（把需要的模块改为 static 编进二进制）
- 其他: 手动编辑 `config.h` 把 `#undef` 改为 `#define`

### 坑 4: GitHub 源码归档不含预生成 configure

**症状**：`builder.sh` 报 `no executable configure file in this project`。

**根因**：GitHub archive 只有 `configure.ac`，没有预生成的 `configure`。

**修复**：`custom_build` 里先 `autoreconf -fi` 生成 configure。

### 坑 5: install.man / install.runhelp 需要 yodl（zsh）

**症状**：`make install` 在 `install.man` 阶段失败。

**根因**：man 页需要 yodl 文档工具生成，主机未安装。

**修复**：设 `pkg_build_autotools_make_install_target="install.bin install.modules install.fns"`（跳过 man/runhelp）。

### 坑 6: Python C 扩展包构建

**症状**：需要交叉编译 `regex`、`numpy` 等 Python C 扩展到 OH musl/aarch64。

**适用模式**：`pkg_build_type="custom"` + `setup_pycrossenv` + `pip install --no-binary :all: .`

**关键点**：
- `pkg_build_deps` 至少需要 `python3>3.7`
- `setup_pycrossenv` 会设置 OH 交叉编译环境（CC/CXX/CFLAGS/LDFLAGS + 目标 Python 头文件）
- `pip install --no-binary :all: .` 在 pycrossenv 中从源码编译 C 扩展
- 产物在 `${HOST_SITE_PKGS}/` 下，`postbuilt_hook` 需复制到 `${target_root_with_pkgname}/${OHOS_LIBDIR}/python${PY_VERSION}/site-packages/`

**参考 BUILD**：`python3-netifaces/BUILD`（简单 C 扩展）、`python3-regex/BUILD`

### 坑 7: Rust/Python 混合包构建（maturin/PyO3）

**症状**：需要交叉编译 `tokenizers`（HuggingFace）等 Rust+Python 混合包。

**核心挑战**：OH SDK 用 musl+libc++（不是 glibc+libstdc++），Rust 的 `aarch64-unknown-linux-musl` target 与 OH SDK 的 `aarch64-linux-ohos` 存在 C++ ABI 和链接库差异。

**前置条件**：主机需要安装 Rust 工具链：
```bash
rustup target add aarch64-unknown-linux-musl
pip install maturin
```

**BUILD 模板**（`pkg_build_type="custom"`）：

`custom_build` hook 中需要设置以下环境变量：

```bash
custom_build() {
    export PATH="${HOME}/.cargo/bin:${PATH}"
    rustup target list --installed | grep -qx aarch64-unknown-linux-musl || \
        rustup target add aarch64-unknown-linux-musl
    command -v maturin >/dev/null 2>&1 || python3 -m pip install maturin

    setup_pycrossenv

    local ohos_sysroot="${OHOS_SDK}/native/sysroot"
    local ohos_cc="${OHOS_SDK}/native/llvm/bin/clang"
    local ohos_cxx_inc="${OHOS_SDK}/native/llvm/include/libcxx-ohos/include/c++/v1"
    local ohos_libdir="${ohos_sysroot}/usr/lib/aarch64-linux-ohos"

    # Rust musl expects libgcc_s/libstdc++, while OH uses libunwind/libc++.
    # Create empty compatibility archives only when the SDK does not provide
    # them; never overwrite existing toolchain libraries.
    mkdir -p "${ohos_libdir}"
    if [[ ! -e "${ohos_libdir}/libgcc_s.a" ]]; then
        "${OHOS_SDK}/native/llvm/bin/llvm-ar" \
            rcs "${ohos_libdir}/libgcc_s.a"
    fi
    if [[ ! -e "${ohos_libdir}/libstdc++.a" ]]; then
        "${OHOS_SDK}/native/llvm/bin/llvm-ar" \
            rcs "${ohos_libdir}/libstdc++.a"
    fi

    # --- Rust linker 配置 ---
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER="${ohos_cc}"
    export RUSTFLAGS="-C linker=${ohos_cc} \
        -C link-arg=--target=aarch64-linux-ohos \
        -C link-arg=--sysroot=${ohos_sysroot} \
        -C link-arg=-lpython${PY_VERSION}"

    # --- cc-rs 配置（Rust 的 C/C++ 编译 wrapper） ---
    # 1. 用 target-specific 变量覆盖编译器，避免 cc-rs 自动加 --target=<rust_target>
    export CC_aarch64_unknown_linux_musl="${ohos_cc}"
    export CXX_aarch64_unknown_linux_musl="${OHOS_SDK}/native/llvm/bin/clang++"
    export CRATE_CC_NO_DEFAULTS=1
    # 2. C++ 头文件：OH SDK 有两套，libcxx-ohos 目录下才有 __config_site
    export CFLAGS_aarch64_unknown_linux_musl="--target=aarch64-linux-ohos --sysroot=${ohos_sysroot} -I${ohos_cxx_inc}"
    export CXXFLAGS_aarch64_unknown_linux_musl="--target=aarch64-linux-ohos --sysroot=${ohos_sysroot} -I${ohos_cxx_inc} -std=c++17"
    # 3. OH 用 libc++ 不是 libstdc++，cc-rs 默认链接 stdc++，需覆盖
    export CXXSTDLIB="c++"

    # --- PyO3 交叉编译 ---
    local host_python=$(which python3)
    export PYO3_PYTHON="${host_python}"
    export PYO3_CROSS_PYTHON_VERSION="${PY_VERSION}"
    export PYO3_CROSS_LIB_DIR="${HOST_PYTHON_DIST}/lib"
    export PYO3_CROSS_INCLUDE_DIR="${HOST_PYTHON_DIST}/include/python${PY_VERSION}"

    pushd ${current_source_root}
    # maturin build：用主机 Python 做 metadata，Rust 交叉编译
    # --skip-auditwheel 跳过共享库打包检查（OH 的 libpython/libc++ 不在主机上）
    maturin build --interpreter "${host_python}" --target aarch64-unknown-linux-musl --release --skip-auditwheel

    # 主机 pip 无法安装 musllinux wheel，手动解压
    local wheel_file=$(find ${current_source_root} -name "*.whl" -path "*/wheels/*" 2>/dev/null | head -1)
    local extract_dir="${current_source_root}/wheel_extracted"
    rm -rf "${extract_dir}" && mkdir -p "${extract_dir}"
    python3 -m zipfile -e "${wheel_file}" "${extract_dir}"
    cp -r "${extract_dir}/<pkg_name>" "${HOST_SITE_PKGS}/"
    cp -r "${extract_dir}/<pkg_name>-<version>.dist-info" "${HOST_SITE_PKGS}/"
    popd

    destroy_pycrossenv
    _custom_build_continue=false
}
```

**`libgcc_s` 缺失**：Rust musl target 链接时需要 `-lgcc_s`，但 OH 用
`libunwind` 替代。模板在 `ohos_sysroot` 定义后创建空兼容 archive，并用
文件存在检查保护 SDK 中已有的 toolchain library，重复构建不会覆盖它们。

**版本兼容性**：编译的包版本必须满足板上其他 Python 包的版本约束（如 `transformers` 要求 `tokenizers>=0.22.0,<=0.23.0`）。PyPI 某些版本可能没有 sdist（只有 wheel），需要检查 `pip download <pkg>==<ver> --no-binary :all:` 是否可用。

**参考 BUILD**：`python3-tokenizers/BUILD`

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

## 参考示例

- **bash/BUILD**: 简单 autotools 包，自带 readline，`--with-curses --without-bash-malloc --disable-nls`
- **zsh/BUILD**: 复杂 autotools 包，需 autoreconf、禁用 autoheader、改 config.modules 把 regex/system 改 static
- **vim/BUILD**: autotools 包，需 patch CFLAGS 和 Makefile，`VIMRUNTIMEDIR` 必须设为板端实际路径
- **libreadline/BUILD**: 简单 autotools 包，`--with-curses`，可作为依赖构建的参考
- **python3-netifaces/BUILD**: Python C 扩展包，`setup_pycrossenv` + `pip install --no-binary :all: .`
- **python3-regex/BUILD**: Python C 扩展包（regex），同 netifaces 模式
- **python3-tokenizers/BUILD**: Rust/Python 混合包（tokenizers），maturin + PyO3 交叉编译，见坑 7
