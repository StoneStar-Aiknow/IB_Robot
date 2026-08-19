# mHandPro SDK 3.0.20 制品清单

本目录记录 IB-Robot mHandPro 真机接入所依赖的厂商运行时 SDK。该 SDK 是预编译的
Linux x86-64 ELF 动态库，不是源码、ROS 包或可由本仓库重建的 Python 依赖。

预期制品路径：

```text
linux-x86_64/libVDMocapSDK_mHandPro.so
```

SDK 通过其导出的 `GetVersionInfo` ABI 报告产品名 `VDMocapSDK_mHandPro`、版本
`3.0.20`。真实手套模式通过 `ctypes.CDLL` 加载该文件；mock 模式不需要它。完整技术
身份、依赖和校验值见 `manifest.yaml`。

二进制本体不进入仓库或发布包。本 PR 只记录外部运行时的技术身份和校验值，不携带厂商
二进制；真机用户必须在仓库外取得合法副本，并通过 `MHANDPRO_SDK_LIB` 指向绝对路径。
后续如需把制品加入提交或发布包，必须另行完成供应商授权和许可证审查，不得用 IB-Robot
的 Apache-2.0 许可证替代厂商条款。

一键启动脚本要求设置 `MHANDPRO_SDK_LIB` 或 `--sdk`，并根据 `SHA256SUMS` 校验已验证的
3.0.20 制品。直接调用校准 CLI 时通过 `--lib-path` 指定该文件。

## Artifact form

The runtime is a vendor-built, closed-source Linux x86-64 shared library. It is loaded through its
exported C ABI and is required only for real-glove operation. It is not stored or packaged by this
repository; this PR keeps it external and records only its technical identity.
