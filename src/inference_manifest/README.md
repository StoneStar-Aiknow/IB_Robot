# inference_manifest

`inference_manifest` 定义 IB-Robot 统一推理 bundle 的硬件无关契约。Schema v2 使用稳定 UUID、自动
revision 和轻量结构摘要，并允许 compiled artifact 声明内容 SHA-256。普通 runtime 不强制读取模型文件
计算 hash；启用 inference scheduler 时，`robot_config` 要求并流式校验 selected compiled artifacts 的 SHA-256。

## Schema v2

```json
{
  "schema_version": 2,
  "bundle": {
    "uuid": "6a7e8c42-1031-41f3-8cde-c1bf8738ca31",
    "revision": 3,
    "name": "pi05-policy",
    "files": [
      {"path": "config.json"},
      {"path": "policy_preprocessor.json"}
    ],
    "digest": {
      "algorithm": "sha256",
      "scope": "structure",
      "value": "<lightweight structural digest>"
    }
  },
  "deployments": {
    "ascend-fp16": {
      "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
      "revision": 8,
      "backend": "ascend",
      "target": {"soc": "Ascend310P3", "runtime": "acl"},
      "artifacts": {
        "vlm": {
          "path": "artifacts/ascend/ascend-fp16/generations/<uuid>/vlm.om",
          "format": "om",
          "sha256": "<64 lowercase hex>"
        },
        "action_expert": {
          "path": "artifacts/ascend/ascend-fp16/generations/<uuid>/action_expert.om",
          "format": "om",
          "sha256": "<64 lowercase hex>"
        }
      },
      "execution": ["vlm", "action_expert"],
      "bindings": {},
      "device_links": []
    }
  }
}
```

实际 compiled deployment 必须提供完整 `bindings`；示例省略了 tensor 内容。
张量 `shape` 使用 ONNX/ACL 语义：空数组 `[]` 表示 rank-0 scalar，`-1` 表示动态维度。

## Identity

| 字段 | 含义 | 变化规则 |
|---|---|---|
| `bundle.uuid` | 策略 bundle lineage | 首次创建时生成，后续保持稳定 |
| `bundle.revision` | 语义资产发布版本 | 通过 bundle refresh 发布时递增 |
| `deployment.uuid` | named deployment lineage | 首次创建时生成，后续保持稳定 |
| `deployment.revision` | 执行定义发布版本 | deployment 结构变化时自动递增 |
| bundle digest | bundle 声明的紧凑摘要 | UUID、revision、名称或文件路径集合变化时变化 |
| deployment fingerprint | selected deployment 的紧凑摘要 | bundle digest、deployment identity 或执行结构变化时变化 |

Bundle digest 对以下 canonical JSON 计算 SHA-256：

```json
{
  "format": "ibrobot.bundle-structure-v2",
  "uuid": "<bundle uuid>",
  "revision": 3,
  "name": "pi05-policy",
  "files": ["config.json", "policy_preprocessor.json"]
}
```

Deployment fingerprint 对 schema version、bundle digest、deployment 名称和完整 typed deployment
声明计算 SHA-256。两者只处理几 KB Manifest JSON，不读取 OM、RKNN、HMM、safetensors 或 tokenizer
文件内容。

这些 identity 值提供版本标识和分布式一致性检查，不提供发布者认证。Scheduler 开启时会通过 artifact
`sha256` 发现原地内容变化；正式发布仍必须经过 packager，并把发布目录设为只读。需要防篡改时应在安装阶段
使用签名、fs-verity、dm-verity 或只读镜像。

## Revision Lifecycle

Exporter/packager 负责 identity，调用者不得手工计算：

1. 新 bundle 和 deployment 自动获得 UUID，revision 从 `1` 开始。
2. Deployment upsert 在 bundle-local lock 内比较去除 identity 后的 typed 结构。
3. 无变化时保留 UUID/revision，不重写 Manifest。
4. 结构变化时保留 UUID，deployment revision 自动加一。
5. Semantic assets 被正式替换后运行 `bump-inference-bundle-revision <bundle>`。
6. Artifact 写入唯一 generation 目录，禁止原地覆盖已发布 generation。
7. Manifest 使用临时文件、`fsync` 和 `os.replace` 原子更新。

PI0.5 单独重编 VLM 或 Action Expert 时，未变化的 counterpart artifact descriptor 会直接复用，不复制
大文件。

## Runtime Validation

`load_inference_manifest()` 在 backend SDK 初始化前执行：

1. 拒绝重复 JSON key、未知字段和非 schema-v2 Manifest。
2. 校验 UUID、正整数 revision 和轻量 bundle digest。
3. 校验 bundle-relative path，拒绝绝对路径、`..`、broken symlink 和 bundle escape。
4. Full loader 要求 bundle files 和 selected artifacts 存在且为普通文件。
5. Metadata-only loader 允许 distributed Edge 缺少 Cloud artifact。
6. 加载 LeRobot metadata，校验 required files、features、bindings、execution graph 和 device links。
7. 计算 selected deployment fingerprint。

普通 runtime 不读取模型文件计算内容 hash。Scheduler 开启时由 `robot_config` 在 backend SDK 初始化前校验
selected compiled artifact；之后 backend 仍会校验实际 runtime descriptor。

## RKNN Sharing

Schema v2 不再根据相同文件 SHA 猜测 RKNN session sharing。需要共享的 artifact 必须显式声明同一个
`share_group`，同时满足：

- 仅用于 RKNN deployment。
- 相同 group 引用同一个 artifact path。
- 相同 group 的 runtime input/output ABI 完全一致。
- 重复 artifact path 必须具有相同非空 group。

## V1 Compatibility

Schema v1 bundle 和所有由旧版 exporter 生成的 artifact 不受支持，也不会原地迁移。旧 Manifest 中
缺少稳定 UUID、revision 和显式 sharing 语义；自动转换可能把旧文件或资源行为错误地合法化为一个
新的 schema-v2 发布版本。

遇到 schema v1 时，必须使用当前 exporter 或 packager 重新生成完整 schema-v2 bundle。Runtime 会
明确拒绝旧 Manifest，不会猜测、迁移或复用旧 artifact。

## Public APIs

| API | 用途 |
|---|---|
| `load_inference_manifest()` | 完整本机运行验证 |
| `load_inference_manifest_metadata()` | Distributed Edge metadata 验证 |
| `canonical_bundle_digest()` | 计算轻量 bundle 结构摘要 |
| `deployment_fingerprint()` | 计算轻量 deployment 结构摘要 |
| `write_inference_manifest()` | Canonical atomic writer |
