# semantic_mapping

`semantic_mapping` 基于固定安装在底盘上的 D435 同步 RGB-D 数据构建独立、持久化的 3D 语义目标地图。它只依赖
时间戳对应的 TF，不依赖 FAST-LIO、FAST-LIVO2 或其他 SLAM 的内部地图表示。

## Data Flow

1. 使用 `ApproximateTimeSynchronizer` 同步彩色图、对齐深度图和 `CameraInfo`。
2. 默认通过服务并发运行 SAM2 盲扫和 RAM++ 打标，再批量调用 SigLIP2 编码与标签匹配。
3. 将 mask 内有效深度像素反投影至相机光学坐标系，并过滤深度离群点。
4. 按 RGB 图像时间戳查询 `global_frame -> camera optical frame` TF。
5. 将物体点云、中心点和世界坐标轴包围盒转换至 `global_frame`。
6. 使用类别、世界坐标距离和 SigLIP 余弦相似度关联持久目标。
7. 将目标状态和 SigLIP 特征写入独立 SQLite 数据库。

## Public Interfaces

| Interface | Default | Description |
| --- | --- | --- |
| Topic | `/semantic_mapping/objects` | `SemanticObject3DArray` 持久目标快照，Reliable + Transient Local |
| Topic | `/semantic_mapping/object_cloud` | 当前处理帧中所有实例的世界坐标点云 |
| Service | `/semantic_mapping/get_objects` | 按 ID、类别、文本、状态、新鲜度和空间区域查询目标 |
| Service | `/semantic_mapping/resolve_target` | 返回相互区分的目标物体姿态和 Nav2 stand-off 姿态 |

数据库默认位于 `~/.ros/ibrobot/semantic_map.sqlite3`，不保存 SLAM 几何点云或 SLAM 状态。

## Configuration

新流水线配置由 `robot_config` YAML 顶层的 `robot.semantic_mapping` 管理，不依赖 embodied minimal closure。
相机引用必须指向一个启用 aligned depth、固定安装并具有 `base_link` 静态变换的 RealSense peripheral：

```yaml
semantic_mapping:
  enabled: true
  camera:
    peripheral: realsense
    mounting: fixed
    parent_frame: base_link
    rgb_topic: /camera/realsense/image_raw
    depth_topic: /camera/realsense/aligned_depth_to_color/image_raw
    camera_info_topic: /camera/realsense/camera_info
  slam:
    global_frame: map
    active_map_hash_topic: /slam/active_geometry_map_hash
    localization_ready_topic: /slam/localization_ready
    authoritative_map_odom_topic: /slam/authoritative_map_odom_ready
    geometry_map_hash: <active-map-hash>
    localization_session_id: <active-session-id>
    calibration_id: <d435-calibration-hash>
    urdf_hash: <active-urdf-hash>
    map_odom_authority: slam
  perception:
    mapping_backend: service
    semantic_roles:
      sam2_masks: semantic_sam2_masks
      ram_plus_tags: semantic_ram_plus_tags
      siglip2_image: semantic_siglip2_image
      siglip2_text: semantic_siglip2_text
      gdino_confirmation: semantic_gdino_confirmation
```

完整配置包含 persistence、mask/depth filtering、bounded queue/batch、lifecycle、target-watch 和 public interface
参数，参考 `robot_config/config/robots/lekiwi_mapping.yaml`。模型 endpoint 的唯一配置源是顶层
`perception_services.services`；每个 role 指向一个 enabled service ID，不再直接配置 backend、endpoint 或模型
identity。service 模式下 loader 从 schema-v2 bundle manifest 取得 semantic identity，验证精确 service type 和 required/optional
policy，并拒绝 SigLIP2 image/text embedding metadata 不兼容的配置。检查入库的 service entries 是 disabled
templates，因此没有 production model assets 时 YAML 仍可加载；启用建图时必须提供有效 bundle 和 named deployment。

在线和离线建图均有独立 launch 入口，不会启动 embodied runtime。在线入口按 role list 启动所有 enabled、
referenced generic model services；离线入口只启动 SAM2、RAM++ 和 SigLIP2 image construction services。两者把
同一正 63-bit configuration generation 和 `require_semantic_identity=true` 传给 generic hosts，并把 generation
及 manifest semantic identities 传给 mapping node：

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 launch semantic_mapping semantic_mapping.launch.py \
  config_path:=/path/to/enabled-lekiwi-mapping.yaml
```

离线入口额外要求 rosbag 路径；它从同一配置读取 camera topics、map/model identities、filtering 和 services：

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 launch semantic_mapping offline_mapping.launch.py \
  config_path:=/path/to/enabled-lekiwi-mapping.yaml \
  bag_path:=/path/to/rosbag
```

旧 embedded mapping 路径仍支持 Hugging Face 格式的本地 Grounding DINO，但只用于显式迁移兼容：

```yaml
detector_backend: huggingface
grounding_model_path: /data/Research/3D_semantic/models/grounding-dino-tiny
sam_checkpoint: /data/Research/3D_semantic/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
sam_config: configs/sam2.1/sam2.1_hiera_l.yaml
siglip_model_path: siglip2_so400m_patch14_384/assets/model
```

`semantic_mapping.migration` 明确记录旧入口状态：`grounded_sam2_node: compatibility` 继续为现有 grasp
流水线提供 `DetectSegment`，`grounded_sam2_snapshot: diagnostic_only` 仅用于诊断/fixture，
`perception_service_node: general_scene` 继续负责通用场景理解。若使用 embedded mapping，必须同时设置
`allow_legacy_embedded: true` 和 `embedded_mapping_backend: migration_only`；这些开关不会静默改变旧接口。
embedded 启动不会解析或启动 generic service bundles，也不要求 service semantic identities；它只保留显式迁移所需
的本地模型参数。service 模式仍严格要求 construction 和 text roles。

`global_frame` 必须是 SLAM 提供的持久固定坐标系。FAST-LIO 可使用 `camera_init`；若后续 SLAM 发布标准
`map` 坐标系，只需修改该参数。

## Model Assets

```bash
./scripts/setup.sh --with-perception
./scripts/download_perception_models.sh
```

下载脚本默认准备 semantic mapping 的四个 schema-v2 bundle：

```text
models/
├── sam2.1_hiera_tiny/{inference_manifest.json,assets/}
├── ram_plus_swin_large_14m/{inference_manifest.json,assets/,model_utils_work/}
├── siglip2_so400m_patch14_384/{inference_manifest.json,assets/,model_utils_work/}
└── grounded_sam2_swint_ogc/{inference_manifest.json,assets/,model_utils_work/}
```

RAM++ 还需要 `ram_models/recognize-anything/` 上游源码树供 runtime 导入 `ram` 包；源码与权重分开管理。
`model_utils_work/` 下的转换产物不是 production deployment。只有 manifest 声明并通过对应 conformance 的
named deployment 才能在 production SSOT 中启用。

映射清单使用各服务响应中经过规范化校验的 `ModelRuntimeInfo.semantic_identity_json`，SigLIP2 图像与文本查询还
必须具有相同的 `embedding_space_id`、维度、归一化和双端预处理契约。backend、named deployment、deployment
fingerprint 和 runtime version 仅作为 run/observation provenance 持久化，不参与地图兼容性判断。运行时模型均从
本地目录加载；缺少资产、semantic identity 不匹配或 backend 未就绪时服务保持 not-ready，mapping 不会静默
回退到其他模型。

## SLAM Handoff

语义地图消费 SLAM 输出，但不拥有 SLAM lifecycle 或 occupancy grid。启动在线 fusion/target resolution 前必须满足：

- `slam.global_frame` 是持久全局坐标系，且由唯一 SLAM authority 发布 `map -> odom`。
- `geometry_map_id` 和 `geometry_map_hash` 对应当前导航几何地图；active hash 必须与数据库 manifest 一致。
- localization session 已 ready，并能按 RGB timestamp 查询 global frame 到 D435 optical frame 的 TF。
- `cloud_map_topic` 已发布非空、同一 map identity 的点云。
- Nav2 footprint、obstacle map 和 reachability checker ready 后才允许生成 stand-off target。

任一条件不满足时 fusion/target resolution fail closed。离线 export 只生成 semantic manifest、object database 和
object-cloud artifacts，不生成或覆盖 `map.pgm`、Nav2 map YAML 或 SLAM database。
standalone launch 不启动或管理 SLAM，也没有伪造 readiness 的默认值。外部集成必须在配置指定的
transient-local topics 发布 active map hash，以及 localization、authoritative map-to-odom、footprint、obstacle map
和 reachability Bool 证据；cloud map 和 timestamped TF 仍从其原生数据流取得。任何证据尚未发布时默认保持
fail closed。

## Workflow Readiness Contract

`semantic_mapping.workflow_readiness` evaluates each workflow independently and returns a `WorkflowReadiness` value
containing all gate evidence, failed reasons, and the aggregate `ready` property. It does not expose one process-wide
readiness boolean and does not make captioning a gate.

| Workflow | Required gates |
| --- | --- |
| Offline map construction | Ready, compatible SAM2, RAM++, and SigLIP2 image; aligned RGB-D/CameraInfo; timestamped TF; localization input |
| Online map construction | Offline gates plus compatible active map identity, localization, authoritative `map -> odom`, `/cloud_map`, and queue/write admission |
| Structured query | Readable, compatible semantic database; no inference service |
| Text query | Structured-query gates plus ready SigLIP2 text with an embedding space compatible with stored image embeddings |
| Navigation staging | Action-admissible object, compatible active map, localization, timestamped TF, authoritative `map -> odom`, footprint, obstacle map, and reachability |
| Manipulation confirmation | Navigation and fresh-object admission plus ready Grounding DINO and confirmation SAM2 |
| Read-only diagnostics | Database successfully opened in diagnostic mode; no inference service |

Existing ROS responses carry applicable failures through `SemanticMapMetadata.readiness_reason` and `message`. Text
queries fail independently when text inference or embedding compatibility is unavailable. 当前接口没有 request
timestamp，也没有 fresh Grounding DINO + SAM2 confirmation result/diagnostic，因此 manipulation readiness 必须
fail closed，并明确报告 `fresh manipulation confirmation has not run`；service discovery 不作为模型 ready 证据。
导航只能复用最近一次 fusion 时按图像时间戳成功查询的 TF 证据，在尚无该证据时 fail closed；当前 target request
不携带时间戳，因此该证据只证明现有地图观测链路，不能声明请求时刻的新鲜 TF。No new ROS readiness
message is defined; callers needing every evidence item can use the pure-Python value object.

## Conformance And Timing

固定 D435 fixture 的 backend-neutral reference 位于
`perception_service/test/fixtures/realsense_rgbd_frame/conformance_reference.json`。conformance 分别比较 mask
count/IoU、embedding cosine、label top-1/score delta、3D centroid/extent 和 failure semantics，不使用一个混合
end-to-end IoU 掩盖不同 stage 的偏差。

性能报告按 backend 和 mask batch size 分组，并分别记录 inference、serialization、queue wait、service round
trip、fusion commit 和 end-to-end 的 P50/P95，同时报告 throughput 和 dropped frames。model-only timing 不能
作为 end-to-end latency；没有样本的报告以 0 和 processed_frames=0 明确表示，不作为 promotion evidence。

初始 production promotion gate 要求至少 100 个 timing samples、mask batch 不超过 8、model inference P50 不
超过 326 ms。在线模式还要求 end-to-end P95 ≤ 750 ms、queue-wait P95 ≤ 100 ms、drop ratio ≤ 5%；离线
模式要求 throughput ≥ 1 frame/s。所有 stage-specific conformance 必须先通过。阈值是初始软件 gate，必须在
目标硬件上记录实际报告后才能 promotion；当前未验证的 `ascend_om` 仍保持 not-ready。

无需 ROS 或真实设备，可使用仓库中的 RealSense RGB-D fixture 运行真实模型验证：

```bash
cd /data/Research/3D_semantic/Grounded-SAM-2
PYTHONPATH=/home/xqw/Research/IB_Robot/src/semantic_mapping \
  ../venv/bin/python \
  /home/xqw/Research/IB_Robot/src/semantic_mapping/scripts/verify_rgbd_fixture.py \
  --fixture /home/xqw/Research/IB_Robot/src/perception_service/test/fixtures/realsense_rgbd_frame \
  --grounding-model /data/Research/3D_semantic/models/grounding-dino-tiny \
  --sam-checkpoint /data/Research/3D_semantic/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --siglip-model models/siglip2_so400m_patch14_384/assets/model
```

安装感知依赖后，也可验证 ROS 2 节点的同步订阅、TF、持久化和查询服务。先启动节点：

```bash
source .shrc_local
ros2 run semantic_mapping semantic_mapping_node --ros-args \
  -p detector_backend:=huggingface \
  -p grounding_model_path:=/data/Research/3D_semantic/models/grounding-dino-tiny \
  -p sam_checkpoint:=/data/Research/3D_semantic/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  -p sam_config:=configs/sam2.1/sam2.1_hiera_l.yaml \
  -p siglip_model_path:=models/siglip2_so400m_patch14_384/assets/model
```

再从另一个终端发布 fixture 并等待查询结果：

```bash
source .shrc_local
python src/semantic_mapping/scripts/publish_rgbd_fixture.py \
  --fixture src/perception_service/test/fixtures/realsense_rgbd_frame
```

librealsense 设备 bag 应通过 `realsense2_camera` 的 `rosbag_filename` 参数回放。输入语义节点前必须确认
`aligned_depth_to_color/image_raw` 实际发布消息；仅出现在 `ros2 topic list` 中并不表示对齐滤镜已产生数据。

## Constraints

- 深度图必须已经对齐到彩色图，且二者分辨率一致。
- 相机内参必须对应彩色图像像素平面。
- TF 必须能按 RGB 图像时间戳查询，节点不会回退到“最新 TF”。
- 当前姿态表示目标中心点和世界坐标轴包围盒，不估计物体自身旋转。
- 关联假定目标在短时间内近似静止；被移动的物体可能在超过距离门限后生成新 ID。

## Local Verification Evidence

The following checks are reproducible without a 310P board:

```bash
source .shrc_local && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  src/perception_service/test src/semantic_mapping/test
source .shrc_local && ./scripts/build.sh --clean
openspec validate redesign-3d-semantic-mapping --strict
```

The current local result is 148 passing tests, a successful clean workspace build,
and successful strict OpenSpec validation. The local suite also verifies that
SAM2, SigLIP2, and Grounding DINO OM adapters fail closed with explicit ABI or
artifact diagnostics when their production contracts are not finalized.

Migration evidence is represented by `semantic_mapping.migration_evidence`.
It separates local gates from board-only gates and refuses a production-default
switch until hardware conformance and timing evidence are present. No local
test claims 310P ACL execution, OM numerical conformance, or production latency.
