# Data Flow Architecture

## When to Read

- 需要追踪 Observation Flow 或 Action Flow 的具体代码路径与执行顺序
- 排查推理（inference）相关的问题：模型输入输出、前后处理、Monolithic/Distributed 模式切换
- 排查动作抖动、chunk 对齐、temporal smoothing 相关问题
- 评估云边协同部署架构，决定如何拆分 Edge / Cloud 节点

## Observation Flow (Sensors → Inference)

```
Camera/JointState → ROS Topic → decode_value() → StreamBuffer → sample() → Preprocessor → Model
```

**Key Code Paths**:
1. `_obs_cb(msg, spec)` in `lerobot_policy_node.py:381-397`
2. `decode_value()` in `contract_utils.py`
3. `_sample_obs_frame()` in `lerobot_policy_node.py:399-420`

## Action Flow (Inference → Hardware)

```
Model → VariantsList → TemporalSmoother → Queue → TopicExecutor → Controller Topic → Hardware
```

**Key Code Paths**:
1. `_result_cb()` in `action_dispatcher_node.py:232-278`
2. `TemporalSmoother.update()` in `temporal_smoother.py`
3. `_control_loop()` in `action_dispatcher_node.py:172-201`

## Inference Execution Modes

### Monolithic Mode (Default)

All inference components in single process with zero-copy tensor passing:

```
lerobot_policy_node process:
  ├─ TensorPreprocessor (CPU)
  ├─ PureInferenceEngine (GPU)
  └─ TensorPostprocessor (CPU)
```

### Distributed Mode (Cloud-Edge)

Edge handles preprocessing/postprocessing, cloud handles GPU inference:

```
Edge Node                    Cloud Node
┌─────────────────┐         ┌─────────────────┐
│ Preprocessor    │ ──────► │ PureInference   │
│ (CPU)           │         │ Engine (GPU)    │
│                 │ ◄────── │                 │
│ Postprocessor   │         └─────────────────┘
│ (CPU)           │
└─────────────────┘
```

**Configuration**:
```yaml
inference:
  mode: distributed  # or monolithic
  edge_node: true    # for edge node
```

## Temporal Smoothing

Cross-frame action chunk smoothing for seamless motion:

```
weight[k] = exp(-temporal_ensemble_coeff * k)
```

- Default `temporal_ensemble_coeff = 0.01` (from ACT paper)
- Precomputed weights for fast blending
- Aligns new chunk with `actions_executed` from previous chunk

**File**: `src/action_dispatch/action_dispatch/temporal_smoother.py`
