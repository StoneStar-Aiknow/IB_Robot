# Rust/Python 混合包构建模板（maturin/PyO3）

## When to Read

- 交叉编译 Rust+Python 混合包（如 `tokenizers`）
- 遇到坑 7 的情况
- 需要为 OH musl/aarch64 构建 PyO3 wheel

## 前置条件

主机需要安装 Rust 工具链：

```bash
rustup target add aarch64-unknown-linux-musl
pip install maturin
```

## BUILD 模板

`pkg_build_type="custom"`，`custom_build` hook 中设置以下环境变量：

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

## 关键说明

**`libgcc_s` 缺失**：Rust musl target 链接时需要 `-lgcc_s`，但 OH 用
`libunwind` 替代。模板在 `ohos_sysroot` 定义后创建空兼容 archive，并用
文件存在检查保护 SDK 中已有的 toolchain library，重复构建不会覆盖它们。

**版本兼容性**：编译的包版本必须满足板上其他 Python 包的版本约束（如 `transformers` 要求 `tokenizers>=0.22.0,<=0.23.0`）。PyPI 某些版本可能没有 sdist（只有 wheel），需要检查 `pip download <pkg>==<ver> --no-binary :all:` 是否可用。

**参考 BUILD**：`python3-tokenizers/BUILD`
