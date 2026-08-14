# 已知踩坑与修复

## When to Read

- 构建过程中遇到 config.h 不生成、模块被禁用、man install 失败等问题时
- 交叉编译 zsh/vim 等 autotools 项目前

## 坑 1: OH SDK `diff` 损坏 → config.h 不生成

**症状**：configure 说 `config.h is unchanged` 但 config.h 不存在，make 报 `No such file`。

**根因**：OH SDK 的 `diff` 对所有输入返回 0，config.status 误判"unchanged"跳过创建。

**修复**：`native_env_hook` 里 `export PATH=/usr/bin:/bin:$PATH`。

## 坑 2: `autoheader` 在 make 阶段删除 config.h

**症状**：configure 成功创建 config.h，但 make 运行 `autoheader` 后 config.h 消失。

**根因**：zsh/vim 等 autotools 项目的 Makefile 在 `headers` target 中调用 `autoheader`，交叉编译下重建 config.h 失败导致被删。

**修复**：configure 前 sed 修改 Makefile.in 禁用 autoheader：

```bash
sed -i 's/cd \$(sdir) && autoheader/cd $(sdir) \&\& true/' Makefile.in
```

## 坑 3: 交叉编译下模块/功能被禁用

**症状**：zsh 的 `zsh/regex`、`zsh/system` 模块 `zmodload` 失败；configure 检测不到某些功能。

**根因**：交叉编译下 configure 无法运行 run-test（dlopen、功能探测等），相关模块被设为 `link=no` 或功能被 `#undef`。

**修复**：configure 后手动修改生成文件：
- zsh: `sed -i 's/link=no/link=static/' config.modules`（把需要的模块改为 static 编进二进制）
- 其他: 手动编辑 `config.h` 把 `#undef` 改为 `#define`

## 坑 4: GitHub 源码归档不含预生成 configure

**症状**：`builder.sh` 报 `no executable configure file in this project`。

**根因**：GitHub archive 只有 `configure.ac`，没有预生成的 `configure`。

**修复**：`custom_build` 里先 `autoreconf -fi` 生成 configure。

## 坑 5: install.man / install.runhelp 需要 yodl（zsh）

**症状**：`make install` 在 `install.man` 阶段失败。

**根因**：man 页需要 yodl 文档工具生成，主机未安装。

**修复**：设 `pkg_build_autotools_make_install_target="install.bin install.modules install.fns"`（跳过 man/runhelp）。

## 坑 6: Python C 扩展包构建

**症状**：需要交叉编译 `regex`、`numpy` 等 Python C 扩展到 OH musl/aarch64。

**适用模式**：`pkg_build_type="custom"` + `setup_pycrossenv` + `pip install --no-binary :all: .`

**关键点**：
- `pkg_build_deps` 至少需要 `python3>3.7`
- `setup_pycrossenv` 会设置 OH 交叉编译环境（CC/CXX/CFLAGS/LDFLAGS + 目标 Python 头文件）
- `pip install --no-binary :all: .` 在 pycrossenv 中从源码编译 C 扩展
- 产物在 `${HOST_SITE_PKGS}/` 下，`postbuilt_hook` 需复制到 `${target_root_with_pkgname}/${OHOS_LIBDIR}/python${PY_VERSION}/site-packages/`

**参考 BUILD**：`python3-netifaces/BUILD`（简单 C 扩展）、`python3-regex/BUILD`

## 坑 7: Rust/Python 混合包构建（maturin/PyO3）

**症状**：需要交叉编译 `tokenizers`（HuggingFace）等 Rust+Python 混合包。

**核心挑战**：OH SDK 用 musl+libc++（不是 glibc+libstdc++），Rust 的 `aarch64-unknown-linux-musl` target 与 OH SDK 的 `aarch64-linux-ohos` 存在 C++ ABI 和链接库差异。

详见 `references/rust-python-template.md`。

## 参考示例

- **bash/BUILD**: 简单 autotools 包，自带 readline，`--with-curses --without-bash-malloc --disable-nls`
- **zsh/BUILD**: 复杂 autotools 包，需 autoreconf、禁用 autoheader、改 config.modules 把 regex/system 改 static
- **vim/BUILD**: autotools 包，需 patch CFLAGS 和 Makefile，`VIMRUNTIMEDIR` 必须设为板端实际路径
- **libreadline/BUILD**: 简单 autotools 包，`--with-curses`，可作为依赖构建的参考
- **python3-netifaces/BUILD**: Python C 扩展包，`setup_pycrossenv` + `pip install --no-binary :all: .`
- **python3-regex/BUILD**: Python C 扩展包（regex），同 netifaces 模式
- **python3-tokenizers/BUILD**: Rust/Python 混合包（tokenizers），maturin + PyO3 交叉编译，见 `references/rust-python-template.md`
