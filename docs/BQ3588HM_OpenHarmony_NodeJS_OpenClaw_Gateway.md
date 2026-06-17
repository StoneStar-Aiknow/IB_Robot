# BQ3588HM OpenHarmony Node.js 与 OpenClaw Gateway 部署记录

本文记录在 Bearkey BQ3588HM OpenHarmony RK3588 开发板上部署 Node.js v22.19.0，并在其基础上离线部署完整 OpenClaw Gateway 的实际操作过程。目标是让其他开发者或 AI agent 可以沿着同一路径继续复现、排障或扩展。

## 1. 目标与最终状态

### 1.1 目标

在 BQ3588HM OpenHarmony 板端完成：

- 交叉编译 Node.js v22.19.0 arm64/musl 版本。
- 将 Node.js、npm、npx、corepack 部署到板端。
- 在板端部署完整 OpenClaw Gateway，而不是仅部署 RosClaw 插件。
- 启动 Gateway 并确认其进入 `ready` 状态。

### 1.2 最终板端状态

| 项目 | 状态 |
|------|------|
| Node.js | `v22.19.0` |
| npm | `10.9.3` |
| npx | `10.9.3` |
| corepack | 可运行 |
| Node.js 部署目录 | `/data/local/nodejs` |
| Node.js 环境脚本 | `/data/local/nodejs/nodejs.env` |
| OpenClaw 版本 | `2026.6.8 (844f405)` |
| OpenClaw 部署目录 | `/data/local/openclaw` |
| OpenClaw 状态目录 | `/data/local/openclaw-home/.openclaw` |
| OpenClaw 环境脚本 | `/data/local/openclaw/openclaw.env` |
| Gateway 日志 | `/data/local/tmp/openclaw-gateway.log` |
| Gateway 监听 | `127.0.0.1:18789`, `::1:18789` |
| Gateway 状态 | 日志出现 `[gateway] ready` |

当前 OpenClaw Gateway 使用最小验证配置：

```sh
openclaw gateway run \
  --dev \
  --allow-unconfigured \
  --auth none \
  --bind loopback \
  --port 18789 \
  --verbose
```

该配置只适合板端本地验证。对外开放前必须改为 token/password 鉴权，并将 bind 从 `loopback` 调整为合适的模式。

## 2. 环境信息

### 2.1 主机环境

本次实际使用的主机路径：

```sh
IB_Robot=/home/xqw/Research/IB_Robot
HDC_BIN=/home/xqw/Research/oh_sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710
OHOS_SDK_TARBALL=/home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz
WORKDIR=/tmp/opencode
```

主机需要：

- `gcc-12`
- `g++-12`
- `python3`
- `make`
- `tar`
- `npm`
- 足够磁盘空间，实际中 `/tmp/opencode/ohos-ndk/18/native` 提取后约 2.9 GB，Node.js 源码与构建产物还会占用更多空间。

检查命令：

```sh
gcc-12 --version
g++-12 --version
nproc
```

### 2.2 板端连接

使用 HDC TCP 连接：

```sh
HDC_BIN=/home/xqw/Research/oh_sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710

"$HDC_BIN" tconn "$HDC_TARGET"
"$HDC_BIN" list targets
```

执行板端命令的固定模式：

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '<command>'
```

推送文件的固定模式：

```sh
"$HDC_BIN" -t "$HDC_TARGET" file send <local_path> <remote_path>
```

### 2.3 板端基础检查

先确认板端架构和当前是否已有 Node.js：

```sh
HDC_BIN=/home/xqw/Research/oh_sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710

"$HDC_BIN" -t "$HDC_TARGET" shell '
uname -a
which node 2>/dev/null || true
which npm 2>/dev/null || true
ls /data/local/skh-run/bin/node 2>/dev/null || true
ls /system/bin/node 2>/dev/null || true
ls /lib/ld-musl-aarch64.so.1 2>/dev/null || true
ls /system/lib64/libc++_shared.so 2>/dev/null || true
'
```

本次实际结果：

- 板端为 `aarch64`。
- 部署前未发现 `node` 或 `npm`。
- `/lib/ld-musl-aarch64.so.1` 存在。
- `/system/lib64/libc++_shared.so` 存在。

## 3. 交叉编译方案概述

Node.js v22.19.0 的交叉编译核心思路：

- Node.js v22.19.0 源码地址：`https://nodejs.org/dist/v22.19.0/node-v22.19.0.tar.gz`
- 主机编译 Node.js 22 需要 `gcc-12` 和 `g++-12`（用于编译构建期辅助工具）。
- 使用 OpenHarmony NDK 工具链交叉编译，目标三元组为 `aarch64-linux-ohos`。
- 配置时使用：

```sh
./configure \
  --dest-cpu=arm64 \
  --dest-os=linux \
  --cross-compiling \
  --openssl-no-asm \
  --prefix=<install_dir>
```

Node.js 上游已有 OpenHarmony 相关 PR（在 `--dest-os` 中添加 `openharmony` 支持），但该修改的完整性尚不确定，因此这里仍使用 `--dest-os=linux`。

上游 PR 链接：

```text
https://github.com/nodejs/node/commit/215587feca
```

### 3.1 替代方案：lycium 一键编译

除了本文档详细记录的手动交叉编译方式外，OpenHarmony 社区还提供了一个集成编译框架 `ttyd_openharmony`，可以一键编译 Node.js：

```sh
sudo apt update
sudo apt install gcc-12 g++-12

git clone https://gitee.com/OpenHarmony_rk_equipment_transplantation/ttyd_openharmony.git
cd ttyd_openharmony/lycium

# 设置 OpenHarmony NDK 环境变量
export OHOS_SDK=<your-ohos-sdk-path>

# 编译 nodejs v22.19.0 arm64-v8a 版本
./build.sh nodejs_22_19_0
```

编译产物在 `ttyd_openharmony/lycium/usr/nodejs_22_19_0` 目录下。

该方案适合快速验证；如果需要对编译过程有更细粒度的控制（如处理 CRC32/AES intrinsic、`libatomic`/`getservbyport_r` 缺失等问题），建议使用本文档的手动交叉编译方式。

## 4. 提取 OpenHarmony NDK

### 4.1 找到 SDK tarball

本次使用：

```sh
/home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz
```

检查 tarball 中是否包含 NDK：

```sh
tar tzf /home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz \
  | grep -E '18/native/llvm/bin/clang$|18/native/llvm/bin/llvm-ar$'
```

预期可见：

```text
18/native/llvm/bin/clang
18/native/llvm/bin/llvm-ar
```

### 4.2 提取必要目录

提取到 `/tmp/opencode/ohos-ndk`：

```sh
mkdir -p /tmp/opencode/ohos-ndk

tar xzf /home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz \
  -C /tmp/opencode/ohos-ndk \
  '18/native/llvm/bin/'

tar xzf /home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz \
  -C /tmp/opencode/ohos-ndk \
  '18/native/llvm/lib/'

tar xzf /home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz \
  -C /tmp/opencode/ohos-ndk \
  '18/native/llvm/include/'

tar xzf /home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz \
  -C /tmp/opencode/ohos-ndk \
  '18/native/sysroot/usr/include/'

tar xzf /home/xqw/Research/oh_sdk/ohos-ros-sdk-build/20260115/ohos-sdk-18-linux-x86_64-20260115.tar.gz \
  -C /tmp/opencode/ohos-ndk \
  '18/native/sysroot/usr/lib/aarch64-linux-ohos/'
```

注意：必须提取 `18/native/llvm/include/`，否则 Node.js 编译时会找不到 C++ 标准库头文件，如：

```text
fatal error: 'string' file not found
fatal error: 'atomic' file not found
fatal error: 'version' file not found
```

检查提取结果：

```sh
ls /tmp/opencode/ohos-ndk/18/native/llvm/bin/clang
ls /tmp/opencode/ohos-ndk/18/native/llvm/bin/clang++
ls /tmp/opencode/ohos-ndk/18/native/llvm/include/c++/v1/string
ls /tmp/opencode/ohos-ndk/18/native/sysroot/usr/include/stdio.h
ls /tmp/opencode/ohos-ndk/18/native/sysroot/usr/lib/aarch64-linux-ohos/libc.so
du -sh /tmp/opencode/ohos-ndk/18/native
```

本次提取后的 `18/native` 约 2.9 GB。

### 4.3 验证交叉编译器

用 C 程序测试：

```sh
bash -c '
OHOS_SDK=/tmp/opencode/ohos-ndk/18/native
CC="${OHOS_SDK}/llvm/bin/clang --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"

cat > /tmp/opencode/test_hello.c << EOF
#include <stdio.h>
int main() { printf("hello ohos aarch64\\n"); return 0; }
EOF

$CC -fPIC -D__MUSL__=1 -o /tmp/opencode/test_hello /tmp/opencode/test_hello.c
file /tmp/opencode/test_hello
'
```

预期输出包含：

```text
ELF 64-bit LSB pie executable, ARM aarch64
interpreter /lib/ld-musl-aarch64.so.1
```

用 C++ 程序测试：

```sh
bash -c '
OHOS_SDK=/tmp/opencode/ohos-ndk/18/native
CXX="${OHOS_SDK}/llvm/bin/clang++ --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"

cat > /tmp/opencode/test_cxx.cpp << EOF
#include <string>
#include <atomic>
int main() { std::string s = "ok"; return 0; }
EOF

$CXX -fPIC -D__MUSL__=1 -std=gnu++20 -o /tmp/opencode/test_cxx /tmp/opencode/test_cxx.cpp
file /tmp/opencode/test_cxx
'
```

## 5. 交叉编译 Node.js v22.19.0

### 5.1 下载源码

```sh
cd /tmp/opencode

if [ ! -f node-v22.19.0.tar.gz ]; then
  wget https://nodejs.org/dist/v22.19.0/node-v22.19.0.tar.gz
fi

tar xzf node-v22.19.0.tar.gz
```

### 5.2 首次 configure

环境变量：

```sh
OHOS_SDK=/tmp/opencode/ohos-ndk/18/native
NODE_SRC=/tmp/opencode/node-v22.19.0
INSTALL_DIR=/tmp/opencode/nodejs-ohos-install
JOBS=$(nproc)

export CC_host="gcc-12"
export CXX_host="g++-12"
export AR_host="ar"
export RANLIB_host="ranlib"
export LINK_host="g++-12"

export AS="${OHOS_SDK}/llvm/bin/llvm-as"
export CC="${OHOS_SDK}/llvm/bin/clang --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"
export CXX="${OHOS_SDK}/llvm/bin/clang++ --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"
export LD="${OHOS_SDK}/llvm/bin/ld.lld"
export STRIP="${OHOS_SDK}/llvm/bin/llvm-strip"
export RANLIB="${OHOS_SDK}/llvm/bin/llvm-ranlib"
export OBJDUMP="${OHOS_SDK}/llvm/bin/llvm-objdump"
export OBJCOPY="${OHOS_SDK}/llvm/bin/llvm-objcopy"
export NM="${OHOS_SDK}/llvm/bin/llvm-nm"
export AR="${OHOS_SDK}/llvm/bin/llvm-ar"
```

关键 CFLAGS/CXXFLAGS：

```sh
export CFLAGS="-fPIC -D__MUSL__=1 -march=armv8-a+crc+aes"
export CXXFLAGS="-fPIC -D__MUSL__=1 -march=armv8-a+crc+aes"
```

`-march=armv8-a+crc+aes` 是本次编译成功的关键。原因见第 5.5 节。

执行 configure：

```sh
cd "$NODE_SRC"

./configure \
  --dest-cpu=arm64 \
  --dest-os=linux \
  --cross-compiling \
  --openssl-no-asm \
  --prefix="$INSTALL_DIR" \
  > /tmp/opencode/nodejs-configure.log 2>&1
```

### 5.3 创建缺失库与缺失符号 stub

OpenHarmony musl sysroot 中没有独立 `libatomic`，而 Node.js 链接阶段会需要 `-latomic`。同时本次还遇到了缺失 `getservbyport_r`：

```text
ld.lld: error: unable to find library -latomic
ld.lld: error: undefined symbol: getservbyport_r
```

创建 `getservbyport_r` stub：

```sh
mkdir -p /tmp/opencode/stubs

cat > /tmp/opencode/stubs/getservbyport_r.c << 'EOF'
#include <netdb.h>
#include <string.h>
#include <errno.h>

int getservbyport_r(int port, const char *proto, struct servent *result_buf,
                    char *buf, size_t buflen, struct servent **result)
{
    struct servent *se = getservbyport(port, proto);
    if (!se) {
        *result = NULL;
        return -1;
    }
    *result_buf = *se;
    *result = result_buf;
    return 0;
}
EOF
```

编译 stub 并打包为 `libatomic.a`：

```sh
bash -c '
OHOS_SDK=/tmp/opencode/ohos-ndk/18/native
CC="${OHOS_SDK}/llvm/bin/clang --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"
AR="${OHOS_SDK}/llvm/bin/llvm-ar"
LIB_DIR="${OHOS_SDK}/sysroot/usr/lib/aarch64-linux-ohos"

echo "" > /tmp/opencode/empty.c
$CC -fPIC -D__MUSL__=1 -c /tmp/opencode/empty.c -o /tmp/opencode/empty.o
$CC -fPIC -D__MUSL__=1 -c /tmp/opencode/stubs/getservbyport_r.c -o /tmp/opencode/stubs/getservbyport_r.o

$AR rcs "${LIB_DIR}/libatomic.a" /tmp/opencode/stubs/getservbyport_r.o /tmp/opencode/empty.o
"${OHOS_SDK}/llvm/bin/llvm-nm" "${LIB_DIR}/libatomic.a" | grep getservbyport
'
```

预期可以看到：

```text
U getservbyport
T getservbyport_r
```

说明 `getservbyport_r` 已由 stub 提供，而 `getservbyport` 会继续从 libc 中解析。

### 5.4 编译与安装

完整编译命令：

```sh
bash -c '
OHOS_SDK=/tmp/opencode/ohos-ndk/18/native
NODE_SRC=/tmp/opencode/node-v22.19.0
INSTALL_DIR=/tmp/opencode/nodejs-ohos-install
JOBS=$(nproc)

export CC_host="gcc-12"
export CXX_host="g++-12"
export AR_host="ar"
export RANLIB_host="ranlib"
export LINK_host="g++-12"

export AS="${OHOS_SDK}/llvm/bin/llvm-as"
export CC="${OHOS_SDK}/llvm/bin/clang --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"
export CXX="${OHOS_SDK}/llvm/bin/clang++ --target=aarch64-linux-ohos --sysroot=${OHOS_SDK}/sysroot"
export LD="${OHOS_SDK}/llvm/bin/ld.lld"
export STRIP="${OHOS_SDK}/llvm/bin/llvm-strip"
export RANLIB="${OHOS_SDK}/llvm/bin/llvm-ranlib"
export OBJDUMP="${OHOS_SDK}/llvm/bin/llvm-objdump"
export OBJCOPY="${OHOS_SDK}/llvm/bin/llvm-objcopy"
export NM="${OHOS_SDK}/llvm/bin/llvm-nm"
export AR="${OHOS_SDK}/llvm/bin/llvm-ar"
export CFLAGS="-fPIC -D__MUSL__=1 -march=armv8-a+crc+aes"
export CXXFLAGS="-fPIC -D__MUSL__=1 -march=armv8-a+crc+aes"

cd "$NODE_SRC"
rm -rf out/

./configure \
  --dest-cpu=arm64 \
  --dest-os=linux \
  --cross-compiling \
  --openssl-no-asm \
  --prefix="$INSTALL_DIR" \
  > /tmp/opencode/nodejs-configure.log 2>&1

make -j"$JOBS" > /tmp/opencode/nodejs-build.log 2>&1
make install > /tmp/opencode/nodejs-install.log 2>&1
'
```

安装树检查：

```sh
ls -la /tmp/opencode/nodejs-ohos-install
ls /tmp/opencode/nodejs-ohos-install/bin
file /tmp/opencode/nodejs-ohos-install/bin/node
du -sh /tmp/opencode/nodejs-ohos-install
```

本次结果：

```text
bin/corepack
bin/node
bin/npm
bin/npx
```

`node` 为：

```text
ELF 64-bit LSB pie executable, ARM aarch64, dynamically linked, interpreter /lib/ld-musl-aarch64.so.1
```

### 5.5 Node.js 编译过程中的关键坑

#### 5.5.1 缺 C++ 标准库头文件

错误：

```text
fatal error: 'string' file not found
fatal error: 'atomic' file not found
fatal error: 'version' file not found
```

原因：只提取 `llvm/bin`、`llvm/lib` 和 sysroot 不够，还需要提取：

```text
18/native/llvm/include/
```

其中包含：

```text
18/native/llvm/include/c++/v1/string
18/native/llvm/include/c++/v1/atomic
18/native/llvm/include/c++/v1/version
```

#### 5.5.2 zlib CRC32/AES intrinsic 编译失败

错误一：

```text
fatal error: error in backend: Cannot select: intrinsic %llvm.aarch64.crc32b
```

错误二：

```text
../deps/zlib/crc32_simd.c:461:27: error: instruction requires: aes
```

原因：Node.js bundled zlib 的 `crc32_simd.c` 使用：

```c
__attribute__((target("arch=armv8-a+aes+crc")))
```

OHOS clang 15.0.4 会提示未知 architecture 并忽略该 target attribute，导致 CRC/AES intrinsic 无法正确选中。

解决：全局加：

```sh
-march=armv8-a+crc+aes
```

也就是：

```sh
export CFLAGS="-fPIC -D__MUSL__=1 -march=armv8-a+crc+aes"
export CXXFLAGS="-fPIC -D__MUSL__=1 -march=armv8-a+crc+aes"
```

#### 5.5.3 `-latomic` 和 `getservbyport_r` 缺失

错误：

```text
ld.lld: error: unable to find library -latomic
ld.lld: error: undefined symbol: getservbyport_r
```

解决：创建 `libatomic.a` stub，并把 `getservbyport_r` 包装实现放进去。详见第 5.3 节。

## 6. 部署 Node.js 到板端

### 6.1 strip 与打包

```sh
/tmp/opencode/ohos-ndk/18/native/llvm/bin/llvm-strip \
  /tmp/opencode/nodejs-ohos-install/bin/node

cd /tmp/opencode
tar czf nodejs-ohos-v22.19.0.tar.gz nodejs-ohos-install/
ls -lh nodejs-ohos-v22.19.0.tar.gz
```

本次 tarball 约 43 MB。

### 6.2 推送并解压

```sh
HDC_BIN=/home/xqw/Research/oh_sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  /tmp/opencode/nodejs-ohos-v22.19.0.tar.gz \
  /data/local/nodejs-ohos-v22.19.0.tar.gz

"$HDC_BIN" -t "$HDC_TARGET" shell '
cd /data/local
rm -rf nodejs nodejs-ohos-install
tar xzf nodejs-ohos-v22.19.0.tar.gz
mv nodejs-ohos-install nodejs
ls /data/local/nodejs/bin
'
```

### 6.3 修复 npm/npx/corepack shebang 问题

Node.js install 生成的 `npm` / `npx` 默认是符号链接，目标 JS 文件的 shebang 为：

```sh
#!/usr/bin/env node
```

但 BQ3588HM OpenHarmony 板端没有 `/usr/bin/env`，只有 `/bin/env` 或 `/system/bin/env`。因此直接执行 `npm` 会出现：

```text
/bin/sh: /data/local/nodejs/bin/npm: No such file or directory
```

解决方式：删除 symlink，替换为使用绝对 Node 路径的 shell wrapper。

```sh
HDC_BIN=/home/xqw/Research/oh_sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710

"$HDC_BIN" -t "$HDC_TARGET" shell '
NODE_HOME=/data/local/nodejs
NODE_BIN=${NODE_HOME}/bin/node

rm -f ${NODE_HOME}/bin/npm ${NODE_HOME}/bin/npx ${NODE_HOME}/bin/corepack

cat > ${NODE_HOME}/bin/npm << EOF
#!/bin/sh
${NODE_BIN} ${NODE_HOME}/lib/node_modules/npm/bin/npm-cli.js "\$@"
EOF

cat > ${NODE_HOME}/bin/npx << EOF
#!/bin/sh
${NODE_BIN} ${NODE_HOME}/lib/node_modules/npm/bin/npx-cli.js "\$@"
EOF

cat > ${NODE_HOME}/bin/corepack << EOF
#!/bin/sh
${NODE_BIN} ${NODE_HOME}/lib/node_modules/corepack/dist/corepack.js "\$@"
EOF

chmod 755 ${NODE_HOME}/bin/npm ${NODE_HOME}/bin/npx ${NODE_HOME}/bin/corepack
'
```

### 6.4 创建 Node.js 环境脚本

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '
cat > /data/local/nodejs/nodejs.env << '\''EOF'\''
export NODE_HOME=/data/local/nodejs
export PATH=${NODE_HOME}/bin:${PATH}
EOF
'
```

以后使用：

```sh
. /data/local/nodejs/nodejs.env
node --version
npm --version
npx --version
```

### 6.5 验证 Node.js

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '
. /data/local/nodejs/nodejs.env
node --version
npm --version
npx --version
node -e "console.log(JSON.stringify({version: process.version, arch: process.arch, platform: process.platform}))"
node -e "const http=require(\"http\"); const fs=require(\"fs\"); console.log(\"http/fs OK\")"
node -e "const crypto=require(\"crypto\"); console.log(crypto.createHash(\"sha256\").update(\"hello ohos\").digest(\"hex\"))"
'
```

本次实际验证：

```text
v22.19.0
10.9.3
10.9.3
{"version":"v22.19.0","arch":"arm64","platform":"linux"}
```

### 6.6 npm 外网访问问题

板端执行 npm install 外部包时出现：

```text
npm error errno EAI_AGAIN
npm error request to https://registry.npmjs.org/... failed, reason: getaddrinfo EAI_AGAIN registry.npmjs.org
```

诊断：

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '
cat /etc/resolv.conf 2>/dev/null || true
cat /system/etc/resolv.conf 2>/dev/null || true
ping -c 1 -W 3 1.1.1.1 2>&1 || true
ping -c 1 -W 3 registry.npmjs.org 2>&1 || true
. /data/local/nodejs/nodejs.env
npm config get registry
'
```

本次结果：

- DNS 文件中有 `114.114.114.114` 和 `8.8.8.8`。
- `ping 1.1.1.1` 不通。
- `registry.npmjs.org` 解析失败。

结论：板端无外网连通，不能直接 `npm install -g openclaw@latest`。因此 OpenClaw 改为主机侧离线准备 arm64/musl 安装目录，再推送到板端。

## 7. 准备 OpenClaw 离线安装目录

### 7.1 确认 OpenClaw npm 包要求

主机侧查询：

```sh
npm view openclaw version dist.tarball bin engines dependencies optionalDependencies --json
```

本次关键结果：

```json
{
  "version": "2026.6.8",
  "bin": {
    "openclaw": "openclaw.mjs"
  },
  "engines": {
    "node": ">=22.19.0"
  }
}
```

板端 Node.js v22.19.0 正好满足最低要求。

### 7.2 主机侧按 linux/arm64/musl 安装 OpenClaw

因为板端无外网，所以在主机上生成目标平台安装前缀：

```sh
rm -rf /tmp/opencode/openclaw-board-prefix
mkdir -p /tmp/opencode/openclaw-board-prefix

npm install -g \
  --prefix /tmp/opencode/openclaw-board-prefix \
  openclaw@2026.6.8 \
  --omit=dev \
  --os=linux \
  --cpu=arm64 \
  --libc=musl
```

本次安装输出：

```text
added 294 packages
```

检查：

```sh
ls -la /tmp/opencode/openclaw-board-prefix/bin
du -sh /tmp/opencode/openclaw-board-prefix
node /tmp/opencode/openclaw-board-prefix/lib/node_modules/openclaw/openclaw.mjs --version
```

本次结果：

```text
OpenClaw 2026.6.8 (844f405)
```

安装目录约 349 MB，打包后约 54 MB。

### 7.3 检查 native 依赖是否包含 arm64 产物

```sh
find /tmp/opencode/openclaw-board-prefix -name '*.node' -o -type f -perm -111 \
  | while read p; do file "$p"; done \
  | head -80
```

本次确认存在 arm64 native 产物，例如：

```text
@lydell/node-pty-linux-arm64/prebuilds/linux-arm64/pty.node: ELF 64-bit LSB shared object, ARM aarch64
tree-sitter-bash/prebuilds/linux-arm64/tree-sitter-bash.node: ELF 64-bit LSB shared object, ARM aarch64
sqlite-vec-linux-arm64
```

注意：目录中也可能包含 x64/win32/darwin 等其他平台预构建物，这是依赖包自己的布局，不代表实际会加载它们。关键是必须有 linux arm64 版本。

## 8. 部署 OpenClaw 到板端

### 8.1 打包和推送

```sh
cd /tmp/opencode
tar czf /tmp/opencode/openclaw-board-prefix.tar.gz -C /tmp/opencode openclaw-board-prefix
ls -lh /tmp/opencode/openclaw-board-prefix.tar.gz
```

推送并解压：

```sh
HDC_BIN=/home/xqw/Research/oh_sdk/toolchains/hdc
HDC_TARGET=192.168.136.111:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  /tmp/opencode/openclaw-board-prefix.tar.gz \
  /data/local/openclaw-board-prefix.tar.gz

"$HDC_BIN" -t "$HDC_TARGET" shell '
cd /data/local
rm -rf openclaw openclaw-board-prefix
tar xzf openclaw-board-prefix.tar.gz
mv openclaw-board-prefix openclaw
ls /data/local/openclaw/bin
du -sh /data/local/openclaw
'
```

本次部署后 `/data/local/openclaw` 约 286 MB。

### 8.2 修复 OpenClaw CLI shebang 问题

OpenClaw CLI 入口 `openclaw.mjs` 的 shebang 是：

```sh
#!/usr/bin/env node
```

板端没有 `/usr/bin/env`，所以不能直接使用原始 symlink。

特别注意：`/data/local/openclaw/bin/openclaw` 初始是 symlink：

```text
openclaw -> ../lib/node_modules/openclaw/openclaw.mjs
```

如果直接 `cat > /data/local/openclaw/bin/openclaw`，会跟随 symlink 覆盖真正的 `openclaw.mjs`，导致 CLI 报语法错误。正确做法是：先删除 symlink，再创建 wrapper 文件。

如果误覆盖了 `openclaw.mjs`，从主机恢复：

```sh
"$HDC_BIN" -t "$HDC_TARGET" file send \
  /tmp/opencode/openclaw-board-prefix/lib/node_modules/openclaw/openclaw.mjs \
  /data/local/openclaw/lib/node_modules/openclaw/openclaw.mjs
```

创建 wrapper：

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '
rm -f /data/local/openclaw/bin/openclaw

cat > /data/local/openclaw/bin/openclaw << '\''EOF'\''
#!/bin/sh
NODE=/data/local/nodejs/bin/node
OPENCLAW_HOME=${OPENCLAW_HOME:-/data/local/openclaw-home}
export OPENCLAW_HOME
exec "$NODE" /data/local/openclaw/lib/node_modules/openclaw/openclaw.mjs "$@"
EOF

chmod 755 /data/local/openclaw/bin/openclaw
head -1 /data/local/openclaw/lib/node_modules/openclaw/openclaw.mjs
ls -la /data/local/openclaw/bin/openclaw
'
```

### 8.3 创建 OpenClaw 环境脚本

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '
cat > /data/local/openclaw/openclaw.env << '\''EOF'\''
. /data/local/nodejs/nodejs.env
export OPENCLAW_PREFIX=/data/local/openclaw
export OPENCLAW_HOME=${OPENCLAW_HOME:-/data/local/openclaw-home}
export PATH=${OPENCLAW_PREFIX}/bin:${PATH}
EOF
'
```

验证：

```sh
"$HDC_BIN" -t "$HDC_TARGET" shell '
. /data/local/openclaw/openclaw.env
which openclaw
openclaw --version
openclaw gateway --help | head -20
'
```

预期：

```text
/data/local/openclaw/bin/openclaw
OpenClaw 2026.6.8 (844f405)
```

## 9. 启动 OpenClaw Gateway

### 9.1 最小启动命令

开发验证阶段使用：

```sh
. /data/local/openclaw/openclaw.env
mkdir -p /data/local/openclaw-home

openclaw gateway run \
  --dev \
  --allow-unconfigured \
  --auth none \
  --bind loopback \
  --port 18789 \
  --verbose \
  > /data/local/tmp/openclaw-gateway.log 2>&1 &
```

几秒后检查：

```sh
tail -80 /data/local/tmp/openclaw-gateway.log
netstat -an 2>/dev/null | grep 18789 || true
```

本次成功日志包含：

```text
Dev config ready: $OPENCLAW_HOME/.openclaw/openclaw.json
Dev workspace ready: $OPENCLAW_HOME/.openclaw/workspace-dev
[gateway] loading configuration…
[gateway] resolving authentication…
[gateway] auth mode=none explicitly configured; all gateway connections are unauthenticated.
[gateway] starting...
[gateway] starting HTTP server...
[plugins] loaded 7 plugin(s)
[gateway] http server listening (7 plugins: browser, canvas, device-pair, file-transfer, memory-core, phone-control, talk-voice; ...)
[gateway] ready
```

监听检查：

```text
tcp  0  0 127.0.0.1:18789  0.0.0.0:*  LISTEN
tcp6 0  0 ::1:18789        :::*       LISTEN
```

### 9.2 生成的配置目录

启动后会生成：

```text
/data/local/openclaw-home/.openclaw/
├── identity/
│   └── device.json
├── logs/
├── openclaw.json
├── openclaw.json.last-good
├── state/
│   ├── openclaw.sqlite
│   ├── openclaw.sqlite-shm
│   └── openclaw.sqlite-wal
└── workspace-dev/
    ├── AGENTS.md
    ├── IDENTITY.md
    ├── SOUL.md
    ├── TOOLS.md
    └── USER.md
```

本次 dev 配置示例：

```json
{
  "gateway": {
    "mode": "local",
    "bind": "loopback"
  },
  "agents": {
    "defaults": {
      "workspace": "/data/local/openclaw-home/.openclaw/workspace-dev",
      "skipBootstrap": true
    },
    "list": [
      {
        "id": "dev",
        "default": true,
        "workspace": "/data/local/openclaw-home/.openclaw/workspace-dev",
        "identity": {
          "name": "C3-PO",
          "theme": "protocol droid",
          "emoji": "🤖"
        }
      }
    ]
  }
}
```

### 9.3 CLI probe 现象说明

`openclaw gateway status` 可能显示：

```text
Connectivity probe: failed
device identity required
```

这并不表示 Gateway 没启动。实际日志中 Gateway 已经 `ready`，端口也在监听。该 probe 失败是因为 OpenClaw Gateway WebSocket 握手需要 device identity/credentials。

本次判断 Gateway 成功的依据：

- 进程存在。
- `127.0.0.1:18789` 监听。
- 日志出现 `[gateway] ready`。
- 插件加载成功。

## 10. 板端管理脚本

OpenHarmony 板端没有 systemd，因此不要依赖：

```sh
openclaw gateway install
openclaw gateway start
openclaw gateway stop
```

本次创建了三个简单管理脚本。

### 10.1 `/data/local/openclaw/start-gateway.sh`

```sh
#!/bin/sh
set -eu
. /data/local/openclaw/openclaw.env
PORT=${OPENCLAW_PORT:-18789}
BIND=${OPENCLAW_BIND:-loopback}
AUTH=${OPENCLAW_AUTH:-none}
LOG=${OPENCLAW_LOG:-/data/local/tmp/openclaw-gateway.log}
PIDFILE=/data/local/openclaw/gateway.pid
mkdir -p /data/local/tmp "$OPENCLAW_HOME"
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "OpenClaw gateway already running: pid=$PID"
    exit 0
  fi
  rm -f "$PIDFILE"
fi
if netstat -an 2>/dev/null | grep ":$PORT " | grep LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already listening; refusing to start another gateway"
  exit 1
fi
openclaw gateway run --dev --allow-unconfigured --auth "$AUTH" --bind "$BIND" --port "$PORT" --verbose > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "OpenClaw gateway started: pid=$(cat "$PIDFILE"), bind=$BIND, port=$PORT, log=$LOG"
```

### 10.2 `/data/local/openclaw/stop-gateway.sh`

```sh
#!/bin/sh
PIDFILE=/data/local/openclaw/gateway.pid
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$PID" ]; then
    kill "$PID" 2>/dev/null || true
    sleep 2
    kill -9 "$PID" 2>/dev/null || true
  fi
fi
rm -f "$PIDFILE"
echo "OpenClaw gateway stopped"
```

### 10.3 `/data/local/openclaw/status-gateway.sh`

```sh
#!/bin/sh
. /data/local/openclaw/openclaw.env
PIDFILE=/data/local/openclaw/gateway.pid
echo "PID file:"
cat "$PIDFILE" 2>/dev/null || echo "none"
echo
echo "Process:"
PID=$(cat "$PIDFILE" 2>/dev/null || true)
if [ -n "$PID" ]; then ps -ef | grep " $PID " | grep -v grep || true; fi
echo
echo "Listeners:"
netstat -an 2>/dev/null | grep "${OPENCLAW_PORT:-18789}" || true
echo
echo "Last log lines:"
tail -40 "${OPENCLAW_LOG:-/data/local/tmp/openclaw-gateway.log}" 2>/dev/null || true
```

赋权：

```sh
chmod 755 \
  /data/local/openclaw/start-gateway.sh \
  /data/local/openclaw/stop-gateway.sh \
  /data/local/openclaw/status-gateway.sh
```

使用：

```sh
/data/local/openclaw/start-gateway.sh
/data/local/openclaw/status-gateway.sh
/data/local/openclaw/stop-gateway.sh
```

### 10.4 脚本适配注意事项

OpenHarmony 板端缺少部分常见 Linux 工具。本次遇到：

- `awk` 不存在。
- `getprop` 不存在。
- `ps -ef` 中 OpenClaw 进程名显示为 `openclaw`，不显示完整 `node ... openclaw.mjs gateway ...` 命令。

因此脚本最后避免使用 `awk`，也不再用模糊 `ps | grep openclaw.mjs gateway` 作为唯一判断，而是用 pidfile + port 检测。

## 11. 从 loopback 切换到 LAN 访问

当前为了安全验证使用：

```sh
--bind loopback --auth none
```

如果需要局域网访问 Gateway，应至少：

1. 设置鉴权 token 或 password。
2. 改为 `--bind lan`。
3. 明确端口开放范围。

示例：

```sh
export OPENCLAW_BIND=lan
export OPENCLAW_AUTH=token
export OPENCLAW_GATEWAY_TOKEN='<replace-with-strong-token>'
/data/local/openclaw/stop-gateway.sh
/data/local/openclaw/start-gateway.sh
```

注意：当前 `start-gateway.sh` 使用 `AUTH=${OPENCLAW_AUTH:-none}`，如果要支持 token，还需确保 OpenClaw CLI 能从环境读取 `OPENCLAW_GATEWAY_TOKEN`，或修改脚本显式传入：

```sh
--token "$OPENCLAW_GATEWAY_TOKEN"
```

## 12. 当前限制与后续工作

### 12.1 当前限制

- 板端无外网，npm registry、OpenClaw update check、模型 API 访问都会受影响。
- 当前 Gateway 是 dev 配置，适合验证，不适合长期安全运行。
- 当前 Gateway 只绑定 loopback，外部机器无法直接访问。
- 未配置模型 provider 凭据，不能完成真实 AI 对话。
- 未配置 Telegram/WhatsApp/Slack 等 channel。
- 未接入 RosClaw 插件，本文只覆盖完整 OpenClaw Gateway 本体部署。

### 12.2 推荐下一步

1. 修复板端网络或配置代理，使其能访问模型 API 和必要外部服务。
2. 改用 token/password 鉴权。
3. 将 bind 从 `loopback` 切换到 `lan`，并验证局域网客户端连接。
4. 配置 `OPENCLAW_HOME/.openclaw/openclaw.json` 中的模型 provider。
5. 接入所需聊天渠道。
6. 后续再部署 RosClaw plugin，使 OpenClaw Gateway 能通过 rosbridge 控制 ROS 2 机器人。

## 13. 快速复现清单

主机侧：

```sh
# 1. 提取 OHOS NDK
mkdir -p /tmp/opencode/ohos-ndk
tar xzf $OHOS_SDK_TARBALL -C /tmp/opencode/ohos-ndk '18/native/llvm/bin/'
tar xzf $OHOS_SDK_TARBALL -C /tmp/opencode/ohos-ndk '18/native/llvm/lib/'
tar xzf $OHOS_SDK_TARBALL -C /tmp/opencode/ohos-ndk '18/native/llvm/include/'
tar xzf $OHOS_SDK_TARBALL -C /tmp/opencode/ohos-ndk '18/native/sysroot/usr/include/'
tar xzf $OHOS_SDK_TARBALL -C /tmp/opencode/ohos-ndk '18/native/sysroot/usr/lib/aarch64-linux-ohos/'

# 2. 下载 Node.js
cd /tmp/opencode
wget https://nodejs.org/dist/v22.19.0/node-v22.19.0.tar.gz
tar xzf node-v22.19.0.tar.gz

# 3. 创建 libatomic/getservbyport_r stub
# 参考第 5.3 节

# 4. 编译 Node.js
# 参考第 5.4 节

# 5. 打包 Node.js
/tmp/opencode/ohos-ndk/18/native/llvm/bin/llvm-strip /tmp/opencode/nodejs-ohos-install/bin/node
tar czf /tmp/opencode/nodejs-ohos-v22.19.0.tar.gz -C /tmp/opencode nodejs-ohos-install

# 6. 主机侧准备 OpenClaw 离线包
rm -rf /tmp/opencode/openclaw-board-prefix
mkdir -p /tmp/opencode/openclaw-board-prefix
npm install -g --prefix /tmp/opencode/openclaw-board-prefix openclaw@2026.6.8 --omit=dev --os=linux --cpu=arm64 --libc=musl
tar czf /tmp/opencode/openclaw-board-prefix.tar.gz -C /tmp/opencode openclaw-board-prefix
```

板端：

```sh
# 1. 部署 Node.js
cd /data/local
tar xzf nodejs-ohos-v22.19.0.tar.gz
mv nodejs-ohos-install nodejs

# 2. 创建 npm/npx/corepack wrappers 和 nodejs.env
# 参考第 6.3、6.4 节

# 3. 部署 OpenClaw
cd /data/local
tar xzf openclaw-board-prefix.tar.gz
mv openclaw-board-prefix openclaw

# 4. 创建 OpenClaw wrapper 和 openclaw.env
# 参考第 8.2、8.3 节

# 5. 启动 Gateway
/data/local/openclaw/start-gateway.sh
/data/local/openclaw/status-gateway.sh
```

验证标准：

```text
node --version => v22.19.0
npm --version => 10.9.3
openclaw --version => OpenClaw 2026.6.8 (844f405)
netstat -an | grep 18789 => LISTEN
openclaw-gateway.log => [gateway] ready
```

## 14. 替代部署方式：系统集成（device_tools）

本文档前文使用的是 `/data/local` 目录下的用户态部署方式，适合快速验证。如果需要将 Node.js 永久集成到 OpenHarmony 系统镜像中，可以采用以下 `device_tools` 系统集成方案。

### 14.1 构建产物目录结构

准备一个 `device_tools` 目录，用于存放环境变量、二进制文件和构建脚本：

```text
device_tools/
├── bundle.json
├── BUILD.gn
├── libc++_shared.so
└── node-v22.19.0-ohos-install.tar.gz
```

其中 `libc++_shared.so` 从 OHOS NDK 中提取（参见第 4 节），`node-v22.19.0-ohos-install.tar.gz` 为第 5 节编译并打包后的 Node.js 安装产物。

### 14.2 bundle.json

`bundle.json` 用于 OpenHarmony 编译构建系统识别该组件：

```json
{
    "name": "device_tools",
    "description": "Node.js and other binary tools for OpenHarmony device",
    "version": "1.0",
    "license": "Apache-2.0",
    "publishAs": "code-segment",
    "segment": {
        "destPath": "device_tools"
    },
    "dirs": {},
    "scripts": {},
    "component": {
        "name": "device_tools",
        "subsystem": "",
        "syscap": [],
        "features": [],
        "rom": "",
        "ram": "",
        "deps": {
            "components": [],
            "third_party": []
        },
        "build": {
            "sub_component": [
                "//device_tools:device_tools"
            ],
            "inner_kits": [],
            "test": []
        }
    }
}
```

### 14.3 BUILD.gn

`BUILD.gn` 用于将二进制文件和资源安装到系统指定路径：

```python
import("//build/ohos.gni")

ohos_prebuilt_etc("libcxx_shared") {
    source = "libc++_shared.so"
    relative_install_dir = "lib"
    subsystem = "device_tools"
    part_name = "device_tools"
}

ohos_prebuilt_etc("nodejs_install") {
    source = "node-v22.19.0-ohos-install.tar.gz"
    relative_install_dir = "bin"
    subsystem = "device_tools"
    part_name = "device_tools"
}

group("device_tools") {
    deps = [
        ":libcxx_shared",
        ":nodejs_install",
    ]
}
```

上述配置将 `libc++_shared.so` 安装到 `/lib`，将 Node.js 安装包放到 `/bin`。

### 14.4 系统初始化脚本

在系统启动初始化脚本中插入以下命令，完成 Node.js 的展开和路径配置：

```sh
# 拷贝 C++ 运行时库
cp /bin/libc++_shared.so /lib

# 解压 Node.js 安装包
cd /bin
tar xvf /bin/node-v22.19.0-ohos-install.tar.gz
cd node-v22.19.0-ohos

# 将 node 二进制和库安装到系统路径
cp bin/* /bin/ -rf
cp lib/* /lib/ -rf

# 创建 /usr 目录并建立符号链接
mkdir -p /usr
ln -sf /bin /usr/bin

# 拷贝共享资源（man pages 等）
cp -rf share /usr/
```

执行完毕后，`node`、`npm`、`npx` 将在系统全局 `PATH` 中可用。

注意：该方式需要系统根目录可写。OpenHarmony 标准系统的根目录默认只读，需要通过修改 `init` 配置或使用 `mount -o remount,rw /` 临时解除。生产环境中应通过编译系统将上述文件打包进系统镜像。

### 14.5 与本文档 /data/local 方式的对比

| 维度 | /data/local 用户态部署（本文档主方案） | device_tools 系统集成 |
|------|---------------------------------------|----------------------|
| 写入位置 | `/data/local/nodejs` | `/bin`、`/lib`、`/usr` |
| 需要修改系统镜像 | 否 | 是 |
| 需要 root remount | 否 | 是（临时）或编译期集成 |
| 重启后保持 | 是（/data 持久化） | 取决于镜像是否包含 |
| 适合场景 | 开发调试、快速验证 | 生产部署、出厂镜像 |

## 15. 其他二进制工具部署

原始参考文档中还提到了以下工具可以按类似方式部署到 OpenHarmony 板端：

| 工具 | 用途 | 部署方式 |
|------|------|---------|
| `tcpdump` | 网络抓包分析 | 交叉编译或使用 OpenHarmony 预编译版本 |
| `ffmpeg` | 音视频处理 | 参考 thirdparty_pytorch 中的 FFmpeg SDK 扩展层 |
| `v4l2-ctl` | V4L2 视频设备控制 | 交叉编译 v4l-utils |

这些工具的交叉编译不在本文档范围内，但部署方式与 Node.js 类似：交叉编译得到 aarch64 ELF 二进制后，推送到板端 `/data/local` 或通过 `device_tools` 集成到系统镜像。

## 16. 在 toybox 中增加 vi 命令

OpenHarmony 使用 `toybox` 作为默认的工具箱。如果需要 `vi` 编辑器，可以通过 patch toybox 源码实现。

toybox `vi` patch 文件：`81e39c5.diff.zip`

使用方式：

```sh
# 获取 toybox 源码
git clone https://gitee.com/openharmony/third_party_toybox.git
cd third_party_toybox

# 应用 vi 补丁
unzip 81e39c5.diff.zip
git apply 81e39c5.diff

# 重新编译 toybox 并推送到板端
# 具体编译方式取决于 OpenHarmony 版本和构建系统
```

应用补丁后，`vi` 命令将出现在 toybox 的命令列表中。

## 17. 参考链接

### Node.js 交叉编译

- Node.js v22.19.0 源码：`https://nodejs.org/dist/v22.19.0/node-v22.19.0.tar.gz`
- Node.js OpenHarmony 支持上游 PR：`https://github.com/nodejs/node/commit/215587feca`
- lycium 集成编译框架：`https://gitee.com/OpenHarmony_rk_equipment_transplantation/ttyd_openharmony`

### OpenHarmony SDK

- OpenHarmony NDK 下载：`https://www.openharmony.cn/maincontribution/harmonyos-sdk`
- BQ3588HM 板端使用文档：`docs/BQ3588HM_board_usage.md`
- BQ3588HM OpenHarmony ROS 文档：`docs/BQ3588HM_OpenHarmony_ROS.md`

### OpenClaw

- OpenClaw 官方仓库：`https://github.com/openclaw/openclaw`
- OpenClaw 文档：`https://docs.openclaw.ai`
- OpenClaw 快速入门：`https://docs.openclaw.ai/start/getting-started`

### 其他

- PyYAML 官方仓库：`https://github.com/yaml/pyyaml`
- toybox 项目：`https://github.com/landley/toybox`
