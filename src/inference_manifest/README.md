# inference_manifest

`inference_manifest` 定义 IB-Robot 统一推理 bundle 的硬件无关契约，提供 JSON Schema、
Pydantic 模型、严格加载器、规范化写入器、完整性校验和安全路径处理。该包只依赖
Pydantic 与 jsonschema，不导入 Torch、ROS 2 或任何后端 SDK，因此可以在模型导出工具、
机器人边缘端和推理主机之间共享。

运行时通过策略 bundle 根目录中的唯一 `inference_manifest.json` 选择一个命名
deployment。Manifest 只描述部署身份、artifact 和 runtime ABI；LeRobot 策略语义仍由
`config.json`、processor 配置、processor state、tokenizer 和原生权重拥有。

## 职责边界

本包负责：

- 定义 `schema_version: 1` 的 manifest 数据结构。
- 严格拒绝未知字段、隐式类型转换、重复 JSON key 和不安全路径。
- 发现并校验 LeRobot 策略语义、必需本地资产和 feature shapes。
- 校验 bundle 文件、compiled artifact 的 SHA-256 和 canonical bundle digest。
- 校验 compiled deployment 的 execution graph、tensor bindings 和 device links。
- 生成稳定的 deployment fingerprint，供分布式 Edge/Cloud 身份握手使用。
- 以确定性 JSON 格式原子写入 manifest。

本包不负责：

- 加载或执行 Torch、Ascend、Hisilicon、RKNN、HMM 后端。
- 编译、复制或自动发现部署 artifact。
- 修改 LeRobot `config.json` 或 processor 文件。
- 根据目录内容、文件后缀或环境变量猜测 backend。

模型导出和打包通常应使用
[`model_utils`](../model_utils/model_utils/README.md)；不要手工维护 hashes、bindings 或
bundle digest。

## Bundle 结构

典型策略 bundle：

```text
policy_bundle/
├── config.json
├── model.safetensors                         # 存在 Torch deployment 时需要
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # PI0.5 / SmolVLA 按需存在
├── artifacts/
│   ├── ascend/<deployment>/...
│   ├── hisilicon/<deployment>/...
│   ├── rknn/<deployment>/...
│   └── hmm/<deployment>/...
└── inference_manifest.json
```

`bundle.files` 声明的是策略语义资产，不包含 compiled deployment artifacts。Artifact 在
对应 deployment 的 `artifacts` 中单独声明和校验。两类路径不能重复。

## Manifest 结构

Manifest 顶层包含三个字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | `1` | 当前唯一支持的 schema 版本 |
| `bundle` | object | 策略名称、语义文件清单和 canonical digest |
| `deployments` | object | 一个或多个命名部署；名称不是 backend 名称 |

顶层和所有子对象均禁止未知字段。Deployment 名称必须匹配
`^[A-Za-z0-9][A-Za-z0-9_.-]*$`。

### Bundle

```json
{
  "name": "so101_act",
  "files": [
    {
      "path": "config.json",
      "sha256": "<64 个小写十六进制字符>"
    }
  ],
  "digest": {
    "algorithm": "sha256",
    "value": "<canonical bundle digest>"
  }
}
```

每个 `files` 条目包含 bundle 相对路径和文件 SHA-256。Canonical bundle digest 对按路径
排序后的 `[{"path": ..., "sha256": ...}]` 紧凑 JSON 计算 SHA-256，因此不受
`bundle.files` 原始顺序影响。

### Torch Deployment

Torch deployment 直接使用 LeRobot 原生权重，不声明 compiled artifact 或 tensor
bindings：

```json
{
  "backend": "torch",
  "device": "cpu"
}
```

`device` 支持 `cpu`、`cuda`、`mps` 和 `npu`。Manifest 校验不检查当前主机是否真的具备
该设备；后端初始化时再执行设备准入检查。

### Compiled Deployment

Compiled deployment 的 `backend` 支持 `ascend`、`hisilicon`、`rknn` 和 `hmm`：

```json
{
  "backend": "rknn",
  "target": {
    "soc": "rk3588",
    "runtime": "rknn-lite2"
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/policy.rknn",
      "format": "rknn",
      "sha256": "<64 个小写十六进制字符>"
    }
  },
  "execution": ["policy"],
  "bindings": {
    "policy": {
      "inputs": [
        {
          "semantic": "observation.state",
          "runtime_name": "state",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 6]
        },
        {
          "semantic": "observation.images.top",
          "runtime_name": "image",
          "index": 1,
          "dtype": "float32",
          "shape": [1, 480, 640, 3],
          "layout": "NHWC"
        }
      ],
      "outputs": [
        {
          "semantic": "action",
          "runtime_name": "actions",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 100, 6]
        }
      ]
    }
  }
}
```

上例中的摘要为字段占位符，不是可加载 bundle。应由 exporter 或 packager 计算真实值。

Compiled deployment 字段：

| 字段 | 含义 |
| --- | --- |
| `target` | 目标 SoC 和 runtime 标识 |
| `artifacts` | execution role 到已固化 runtime artifact 的映射 |
| `execution` | role 的确定性执行顺序 |
| `bindings` | 每个 execution role 的完整输入/输出 ABI 映射 |
| `device_links` | 可选的跨 role 设备指针关系 |

`artifacts` 可以包含不参与 `execution` 的辅助 artifact，例如 Hisilicon worker
executable；这些 artifact 仍会进行存在性和 SHA-256 校验。`bindings` 的 role 集合必须与
`execution` 完全一致，而 `artifacts` 至少覆盖全部 execution roles。

### 核心模型

| 模型 | 用途 |
| --- | --- |
| `InferenceManifest` | 顶层 schema、bundle 和 deployments |
| `ManifestBundle` | bundle 名称、语义文件条目和 digest |
| `BundleFile` / `Digest` | 文件 SHA-256 条目和 canonical digest |
| `TorchDeployment` | 原生 Torch 设备选择 |
| `CompiledDeployment` | target、artifacts、execution、bindings 和 device links |
| `DeploymentTarget` | compiled deployment 的 SoC/runtime 标识 |
| `DeploymentArtifact` | 一个 bundle 内 runtime artifact 及其格式和 SHA-256 |
| `ArtifactBindings` / `TensorBinding` | 一个 role 的 inputs/outputs 及每个 tensor 的 ABI 语义 |
| `DeviceLink` | execution roles 之间的设备指针关系 |
| `PolicyMetadata` / `PolicyFeature` | 从 LeRobot 文件只读发现的策略语义 |

所有模型均为 strict、frozen 且 `extra="forbid"`：不会把字符串自动转换成数字，不允许
运行时修改字段，也不接受未声明的扩展字段。`Deployment` 是以 `backend` 为 discriminator
的 `TorchDeployment | CompiledDeployment` 联合类型，`ExecutionRole` 是经过命名规则校验
的字符串类型。

## Tensor Bindings

`TensorBinding` 将 runtime tensor 显式映射到策略语义：

| 字段 | 规则 |
| --- | --- |
| `semantic` | 例如 `observation.state`、`observation.images.top`、`action` 或 `internal.*` |
| `runtime_name` | runtime tensor 名称；与 `index` 至少提供一个 |
| `index` | 非负 runtime 位置；与 `runtime_name` 至少提供一个 |
| `dtype` | `bool`、整数类型、`float16`、`bfloat16`、`float32` 或 `float64` |
| `shape` | 至少一维；正整数表示静态维度，`-1` 表示动态维度 |
| `layout` | 图像语义必须为 `NCHW` 或 `NHWC`；非图像语义必须省略 |

同一个 role 的 inputs 或 outputs 内，semantic、已声明的 runtime name 和 index 均不得重复。
如果某个方向使用 index，则所有 binding 都必须声明 index。Input indices 必须从 `0`
开始连续；output indices 可以稀疏。Compiled deployment 至少需要一个 `action` output。

加载器还会将 `observation.*` 和 `action` binding shape 与 LeRobot `config.json` 中的
features 对照。PI0.5 与 SmolVLA 允许编译图对 STATE/ACTION 末维做 padding，也允许视觉
输入采用不同空间分辨率，但 channel 维仍须匹配。

## Device Links

多模块 compiled deployment 可以用匹配的 `internal.*` output/input 连接 role。需要表达
设备内存零拷贝、所有权和生命周期时，使用 `device_links`：

```json
{
  "semantic": "internal.kv_cache",
  "producer": "vlm",
  "consumer": "action_expert",
  "transport": "device_pointer",
  "owner": "producer"
}
```

规则包括：

- `semantic` 必须以 `internal.` 开头。
- producer 必须在 consumer 之前执行。
- semantic 必须匹配 producer binding 和 consumer input binding。
- `producer_binding` 默认为 `output`；设为 `input` 时 `owner` 必须为 `producer`。
- `transport` 当前只能是 `device_pointer`，`lifetime` 当前只能是 `inference`。
- 每个 `internal.*` input 必须有一个更早执行的 producer 或显式 device link。

## 加载 API

### 完整运行时校验

```python
from inference_manifest import load_inference_manifest

validated = load_inference_manifest("/path/to/policy_bundle", "rk3588")

print(validated.deployment.backend)
print(validated.policy.input_features)
print(validated.fingerprint)
```

`load_inference_manifest()` 在后端 SDK 初始化前完成：

1. 严格解析 JSON 并拒绝重复 key。
2. 检查 `schema_version`，执行 JSON Schema 和 Pydantic 类型校验。
3. 选择命名 deployment 并检查 bundle/artifact 路径唯一性。
4. 校验全部 `bundle.files` 的文件 SHA-256 和 canonical bundle digest。
5. 只读解析 LeRobot config、processor、state、tokenizer/VLM 引用和 feature metadata。
6. 检查 manifest 是否完整声明必需语义文件。
7. 对 compiled deployment 校验全部 artifact SHA-256 和 feature compatibility。
8. 计算 deployment fingerprint 并返回 `ValidatedManifest`。

如果 manifest 中存在任意 Torch deployment，完整加载会要求
`model.safetensors` 存在并纳入 bundle，即使本次选择的是 compiled deployment。

### Metadata-only 校验

```python
from inference_manifest import load_inference_manifest_metadata

metadata = load_inference_manifest_metadata("/path/to/policy_bundle", "rk3588")
```

`load_inference_manifest_metadata()` 用于分布式 Edge/Cloud 身份建立和 Edge 侧 processor
加载。它仍然校验 schema、类型、bundle digest 声明、必需语义文件及其 SHA-256、compiled
bindings 和 fingerprint，但不要求 cloud-only compiled artifacts 或
`model.safetensors` 存在，也不校验 artifact hashes。不能使用该 API 代替后端启动前的完整
校验。

### ValidatedManifest

| 属性 | 含义 |
| --- | --- |
| `bundle_root` | 已解析的 bundle 根目录 |
| `manifest_path` | 已解析的 `inference_manifest.json` 路径 |
| `manifest` | 完整的 `InferenceManifest` 类型化对象 |
| `deployment_name` | 已选择的 deployment 名称 |
| `deployment` | `TorchDeployment` 或 `CompiledDeployment` |
| `policy` | 从 LeRobot 文件发现的 `PolicyMetadata` |
| `fingerprint` | 当前 bundle digest、deployment 名称和定义的稳定身份 |

## Policy Metadata API

```python
from inference_manifest import load_policy_metadata

policy = load_policy_metadata(
    "/path/to/policy_bundle",
    require_native_weights=False,
)
```

`load_policy_metadata()` 不修改策略文件，返回：

- `policy_type`、`input_features` 和 `output_features`。
- 可选的 `nominal_chunk_size` 与 `max_action_dimension`。
- 必须纳入 manifest `bundle.files` 的 `required_files`。
- tokenizer 或 VLM 等尚未 vendor 到 bundle 的 `external_dependencies`。
- 本次发现是否要求原生权重的 `native_weights_required`。

统一 manifest 的最终导出流程要求语义依赖全部位于 bundle 内。`model_utils` 的共享
`upsert_deployment()` 会在发现 external dependencies 时拒绝完成 manifest，提示先 vendor
对应资产并更新 LeRobot metadata。

## 写入 API

`canonical_manifest_bytes()` 和 `write_inference_manifest()` 是 exporter、packager 与测试使用
的低层 API：

```python
from inference_manifest import canonical_manifest_bytes, write_inference_manifest

content = canonical_manifest_bytes(manifest)
path = write_inference_manifest(
    "/path/to/policy_bundle/inference_manifest.json",
    manifest,
)
```

两个 API 都会先执行 JSON Schema 和 Pydantic 校验。规范 JSON 使用 UTF-8、两空格缩进、
排序后的 object keys，省略 `None` 和默认值，并以换行结尾。写入器在目标目录创建临时
文件，完成 flush/fsync 后通过 `os.replace()` 原子替换目标，再同步目录。

常规模型转换不要直接调用该 API 拼装 hashes。共享导出器会收集策略资产、计算摘要、
合并 deployment、原子写入，并用生产 `load_inference_manifest()` 回读验证；失败时恢复原
manifest。具体入口见 [`model_utils`](../model_utils/model_utils/README.md)。

## 完整性与路径 API

公共辅助函数：

| API | 用途 |
| --- | --- |
| `sha256_file()` | 流式计算文件 SHA-256，不把大型 artifact 整体读入内存 |
| `verify_file_sha256()` | 安全解析 bundle 文件并比对摘要 |
| `canonical_bundle_digest()` | 计算与文件顺序无关的 bundle digest |
| `verify_bundle_digest()` | 比对 canonical bundle digest |
| `deployment_fingerprint()` | 计算 schema、bundle、deployment 名称与定义的稳定身份 |
| `normalize_bundle_path()` | 规范并验证 POSIX bundle 相对路径 |
| `resolve_bundle_path()` | 解析现有路径并证明其仍位于 bundle 根目录内 |
| `resolve_bundle_file()` | 在上述检查基础上要求目标是普通文件 |
| `manifest_schema()` | 读取包内安装的 JSON Schema |
| `validate_manifest_schema()` | 对原始 JSON 值执行 schema 校验 |

Bundle 路径必须使用 POSIX `/` 分隔符，且禁止空路径、绝对路径、Windows drive prefix、
NUL、空 segment、`.`、`..` 和反斜杠。Symlink 解析后仍必须位于 bundle 根目录；broken
symlink 和指向目录的文件声明都会被拒绝。

## 异常类型

所有公开异常都继承 `ManifestError` 和 `ValueError`：

| 异常 | 含义 |
| --- | --- |
| `ManifestValidationError` | JSON、schema、类型、策略语义或 execution graph 无效 |
| `ManifestPathError` | 路径不安全、缺失、越界或不是所需文件类型 |
| `ManifestIntegrityError` | 文件 SHA-256 或 canonical digest 不匹配 |

摘要不匹配时不要直接修改 manifest。应重新运行拥有该 artifact 的 exporter 或 packaging
workflow，确保 artifact、bindings 和所有身份摘要同步更新。

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `Unsupported schema_version` | 当前只支持版本 `1`；使用匹配代码重新导出 bundle |
| `duplicate JSON key` | JSON 对象存在重复字段；修复生成器，不要依赖后一个值覆盖前一个值 |
| `available deployments` 中没有目标 | 配置选择了不存在的 deployment；检查 robot YAML 和 manifest |
| `omits required LeRobot semantic files` | processor state、tokenizer 或其他本地语义资产未纳入 `bundle.files` |
| `unreferenced LeRobot semantic files` | Manifest 声明了 LeRobot 保留命名资产，但 config/processor 未引用它 |
| `SHA-256 mismatch` | 文件在导出后变化或摘要来源错误；重新运行 exporter/packager |
| `image semantic ... requires ... layout` | 图像 binding 缺少显式 `NCHW`/`NHWC` |
| `non-image semantic ... must omit layout` | 对 state/action 等非图像 tensor 错误声明了 layout |
| `artifact roles must contain every execution role` | `execution` 引用了无 artifact 的 role |
| `binding roles must exactly match execution roles` | `bindings` 缺少 execution role 或包含多余 role |
| `internal input ... has no declared producer` | 多模块内部 tensor 缺少更早的 output 或 device link |
| `bundle path escapes ... through a symlink` | Bundle 内 symlink 最终指向根目录之外；将资产实际 vendor 到 bundle |

## 开发验证

项目命令前先加载环境：

```bash
source .shrc_local
```

Manifest 核心契约测试位于 `inference_service`，metadata/exporter 集成测试位于
`inference_service` 和 `model_utils`：

```bash
python3 -m pytest \
    src/inference_service/tests/test_inference_manifest.py \
    src/inference_service/tests/test_bundle_metadata.py \
    src/model_utils/test/test_inference_manifest_export.py
```

仅检查本包 Python 文件：

```bash
ruff check src/inference_manifest/inference_manifest
ruff format --check src/inference_manifest/inference_manifest
```

## 相关文档

- [模型导出与 deployment 打包](../model_utils/model_utils/README.md)
- [统一推理运行时](../inference_service/README.md)
- [OpenHarmony RKNN 推理流程](../../docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md)
- [HMM 转换与打包](../../docs/Houmo_HMM_Conversion.md)
- [JSON Schema](inference_manifest/inference_manifest.schema.json)
