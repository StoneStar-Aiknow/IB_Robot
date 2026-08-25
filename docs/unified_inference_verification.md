# Unified Inference Verification

Verification record for `refactor-heterogeneous-unified-inference`.

## Host Verification

- Manifest v3, repository manifest inventory, profile projection, artifact integrity, and exporter tests passed.
- Unified runtime lifecycle, recovery, streaming, registry, factory, provider injection, routing, migration, and ACL tests passed.
- Speech Direction streaming and model-session tests passed.
- `git diff --check`, targeted Ruff checks, Python compilation, shell syntax checks, and the inference manifest package build passed.
- Worktree environment verification passed with the main repository venv:
  `WORKSPACE=/home/xqw/Research/IB_Robot-heterogeneous-inference`, Python and Torch from
  `/home/xqw/Research/IB_Robot/venv`, and LeRobot from the worktree submodule.
- Fresh worktree overlay build passed in `build_ros/`, `install_ros/`, and `log_ros/`:
  generated `ibrobot_msgs`, `robot_config` dependency closure, `inference_service`,
  `perception_service`, `voice_tts_service`, `manipulation_service`, `model_utils`,
  `semantic_mapping`, and `action_dispatch` all built successfully.
- ROS mock/contract tests passed with `ROS_DOMAIN_ID=42`, `IBROBOT_TEST_ROS_DOMAIN_ID=42`,
  and `ROS_LOCALHOST_ONLY=1`, including the launched generic model-service Echo contract
  and perception request/response contract.
- Local LeRobot Torch inference passed using models copied from the main repository's
  `models/` directory into temporary v3 test bundles. The original model bundles were not
  modified. Single-run cold-load/inference results with zero-valued tensor observations were:

  | Policy | Device | Load | Inference | Action |
  | --- | --- | ---: | ---: | --- |
  | ACT | CPU | not recorded | `538.15 ms` | `(100, 6)` on CPU |
  | ACT | CUDA | not recorded | `314.69 ms` | `(100, 6)` on `cuda:0` |
  | PI0.5 | CPU | `94.28 s` | `45.27 s` | `(50, 6)` on CPU |
  | PI0.5 | CUDA | `92.49 s` | `707.11 ms` | `(50, 6)` on `cuda:0` |
  | SmolVLA | CPU | `8.96 s` | `9.21 s` | `(50, 6)` on CPU |
  | SmolVLA | CUDA | `4.66 s` | `474.99 ms` | `(50, 6)` on `cuda:0` |

- Local perception Torch inference passed using real weights copied into temporary schema-v3
  bundles. A synthetic `480x640` RGB image containing a red rectangle exercised preprocessing,
  the shared `TorchModelSession`, and model-specific postprocessing:

  | Model | Device | Load | Inference | Result |
  | --- | --- | ---: | ---: | --- |
  | RAM++ | CPU | `8.08 s` | `1.32 s` | logits `(1, 4585)`, 6 tags |
  | RAM++ | CUDA | `4.53 s` | `174.76 ms` | logits `(1, 4585)`, 6 tags |
  | SigLIP2 image | CPU | `775.03 ms` | `3.58 s` | embedding `(1, 1152)`, L2 norm `1.0` |
  | SigLIP2 text | CPU | shared | `399.58 ms` | embeddings `(3, 1152)`, L2 norms `1.0` |
  | SigLIP2 image | CUDA | `1.03 s` | `262.72 ms` | embedding `(1, 1152)`, L2 norm `1.0` |
  | SigLIP2 text | CUDA | shared | `15.96 ms` | embeddings `(3, 1152)`, L2 norms `1.0` |
  | SAM2 automatic | CPU | `400.90 ms` | `144.51 s` | 2 masks `(2, 480, 640)` |
  | SAM2 automatic | CUDA | `1.59 s` | `7.95 s` | 2 masks `(2, 480, 640)` |
  | Grounding DINO + SAM2 | CPU | `1.35 s` | `4.83 s` | 1 box and mask |
  | Grounding DINO + SAM2 | CUDA | `4.74 s` | `1.22 s` | 1 box and mask |

  The initial SAM2 publication bundle declared `configs/sam2.1/sam2.1_hiera_tiny.yaml`, while the
  installed SAM2 package provides `configs/sam2.1/sam2.1_hiera_t.yaml`. The main repository model
  bundle was subsequently corrected to the available config name, and its adapter artifact digest
  was regenerated.

The broader legacy ROS test collection still contains tests that instantiate production
nodes/plugins without the now-required `RegistrySet` and `RuntimeProviders`, or reuse one
`rclpy` context across test cases. Those failures are tracked as stale test-harness callers,
not treated as successful verification.

## Unavailable Verification

- Torch NPU/MPS matrix: unavailable because those devices are not present in this host environment.
- Ascend 310P/310B: requires the target board and ACL runtime; no hardware execution was performed.
- RKNN RK3588: requires RKNNLite and an RK3588 target; no hardware execution was performed.
- HMM/TCIM XH2: requires the vendor runtime and target board; no hardware execution was performed.
- Hisilicon SD3403: requires the worker executable and target board; no hardware execution was performed.
- GraspGen Torch CUDA: no GraspGen bundle or weights are present under the repository `models/`
  directory, so real-weight inference was not performed.

The unavailable entries require platform-specific CI or target-board validation; they are not represented as successful conformance evidence.
