# Voice TTS Service

[中文](README.md)

`voice_tts_service` is IB-Robot's general-purpose ROS 2 text-to-speech service. It accepts text and an optional
voice prompt, invokes an explicitly selected ZipVoice deployment, and returns one or more independently playable
mono WAV PCM16 audio segments.

This package is responsible only for converting text to audio. Speaker playback, ASR, business orchestration,
and remote inference over SSH are outside its boundary.

## 1. Responsibilities

The package:

1. Exposes a unified ROS 2 TTS service contract.
2. Validates the ZipVoice model bundle, manifest, and named deployment.
3. Manages first-use loading, resident model reuse, explicit preloading, and unloading.
4. Applies bounded long-text segmentation and wraps model output as WAV PCM16.
5. Orchestrates the Text Encoder OM, Flow Decoder OM, and CPU Vocos on Ascend 310P.
6. Enforces request, segment-count, and response-size limits.

It does not manage microphone capture, ASR, speaker playback, dialogue state, business workflows, backend
fallback, or runtime inference through SSH.

## 2. Entry Points

| Item | Path or entry point |
| --- | --- |
| ROS node | `voice_tts_service/voice_tts_node.py` |
| 310P adapter | `voice_tts_service/zipvoice_310p_adapter.py` |
| Bundle packager | `voice_tts_service/package_zipvoice_310p.py` |
| Debug launch | `launch/voice_tts.launch.py` |
| Console entry | `voice_tts_node = voice_tts_service.voice_tts_node:main` |
| Packager entry | `package_zipvoice_310p = voice_tts_service.package_zipvoice_310p:main` |

Production startup uses `robot_config`, whose robot YAML is the single source of truth:

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm
```

The package launch is intended only for standalone debugging:

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 launch voice_tts_service voice_tts.launch.py \
  bundle_path:=/path/to/zipvoice-bundle \
  deployment:=ascend_310p
```

## 3. Data Flow

```text
SynthesizeSpeech request
  -> request validation and text segmentation
  -> load the selected deployment on first use
  -> ZipVoice tokenizer
  -> Text Encoder OM
  -> Flow Decoder OM (four host-scheduled Euler steps)
  -> CPU Vocos
  -> mono float PCM
  -> WAV PCM16 encoding
  -> SynthesizedAudio[] response
```

The shared `ModelSession` serializes inference and owns admission, health, failure state, and close waiting.

## 4. ROS Interfaces

### 4.1 Speech synthesis

| Item | Value |
| --- | --- |
| Service | `/voice_tts/synthesize` |
| Type | `ibrobot_msgs/srv/SynthesizeSpeech` |
| Segment type | `ibrobot_msgs/msg/SynthesizedAudio` |

The request contains the complete text and an optional in-memory WAV prompt plus its transcript. The response
contains status, a stable error code, model runtime identity, timing, and one or more complete WAV segments.
Server-local input and output paths are not part of the contract, so the service can be called across ROS hosts.

A text-only request uses the configured bundle prompt profile:

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 service call /voice_tts/synthesize ibrobot_msgs/srv/SynthesizeSpeech \
  "{text: '机器人语音服务已准备就绪。', prompt_audio: [], prompt_audio_format: '', prompt_text: ''}"
```

The currently verified `ascend_310p` deployment uses a fixed prompt profile and does not support request-scoped
voice cloning. It returns `UNSUPPORTED_PROMPT` when prompt fields are supplied.

### 4.2 Model lifecycle

| Service | Type | Behavior |
| --- | --- | --- |
| `/voice_tts/load` | `std_srvs/srv/Trigger` | Explicitly load and pre-warm the model; idempotent |
| `/voice_tts/unload` | `std_srvs/srv/Trigger` | Wait for active synthesis, then release the model; idempotent |

With the default `load_on_startup=false`:

```text
node startup -> validate the bundle without allocating model runtime memory
first synthesize -> load the model and synthesize
later synthesize calls -> reuse the resident model
unload -> release OM sessions, ACL leases, Vocos, tokenizer, and prompt
next synthesize -> load automatically again
```

Pre-warm when the first synthesis must not pay the load latency:

```bash
ros2 service call /voice_tts/load std_srvs/srv/Trigger "{}"
```

Release model memory while leaving the ROS node and endpoints alive:

```bash
ros2 service call /voice_tts/unload std_srvs/srv/Trigger "{}"
```

Load, synthesize, and unload coordinate session creation and replacement. The shared `ModelSession` owns model
admission and cleanup, and unload waits for active inference before closing the session.

## 5. Model Bundle and Deployment

The model path is configured through `robot_config.bundle_path`; it is never hard-coded. Relative paths resolve
against the absolute `WORKSPACE` set by `.shrc_local`:

```yaml
bundle_path: models/voice_tts/zipvoice
deployment: ascend_310p
```

This selects `$WORKSPACE/models/voice_tts/zipvoice`. The bundle contains `inference_manifest.json`, deployment
artifacts, tokenizer assets, prompt profiles, and the vocoder checkpoint.

The bundle must use manifest schema v2, declare `model.kind=generic` and `model.family=zipvoice`, and contain the
selected named deployment.

The bundle digest and deployment fingerprint identify structure and deployment consistency; they do not read
model contents or provide runtime tamper protection. Before copying, the 310P packager verifies the SHA-256 of
known source OMs, the Vocos checkpoint, token table, and golden fixtures. Runtime validates the manifest, path
safety, file presence, and model ABI. Use signed read-only images or verity when production deployments require
content authenticity. The runtime never infers a backend from the operating system and never falls back after
load failure.

### 5.1 Packaging an Ascend 310P delivery

Large model files are not committed to Git. Package a prepared ZipVoice delivery into the standard bundle:

```bash
source .shrc_local
ZIPVOICE_SOURCE_DIR=/path/to/zipvoice-delivery
ros2 run voice_tts_service package_zipvoice_310p \
  --source "$ZIPVOICE_SOURCE_DIR" \
  --destination "$WORKSPACE/models/voice_tts/zipvoice"
```

The packager validates the Text Encoder OM, Flow Decoder OM, Vocos checkpoint, token table, and default prompt
before creating the `ascend_310p` deployment. Chinese frontend packages are installed from the pinned
`requirements/voice-tts.txt`; the checkpoint-compatible Vocos inference subset is versioned in
`voice_tts_service.vocos_backend`. The model bundle contains weights and configuration only, never executable
Python source. Device addresses and source checkout paths are not runtime configuration values.

The controlled Vocos implementation is recorded as `0.1.0-zipvoice-310p-delivery`; source SHA-256 values are
kept as module constants. It uses the workspace Torch, NumPy, and SciPy stack and intentionally does not install
the PyPI `vocos` package, whose transitive dependencies could replace the ROS/Ascend ABI.

## 6. `robot_config` Configuration

Production configuration lives under `robot.voice_tts` in the robot YAML:

```yaml
voice_tts:
  enabled: true
  bundle_path: models/voice_tts/zipvoice
  deployment: ascend_310p

  service_name: /voice_tts/synthesize
  load_service_name: /voice_tts/load
  unload_service_name: /voice_tts/unload
  load_on_startup: false

  prompt_profile: default
  segment_max_chars: 200
  segment_pause_ms: 150
  max_request_chars: 4000
  max_prompt_audio_bytes: 10485760
  max_prompt_duration_sec: 30.0
  max_segments: 32
  max_response_audio_bytes: 67108864

  device_id: 0
  exit_on_init_failure: true
```

The representative configuration defaults to `enabled=false`, so the feature does not alter existing robot
startup behavior until a bundle and deployment are explicitly enabled.

## 7. Verified Ascend 310P Scope

The current `ascend_310p` deployment:

- Targets Ascend 310P1.
- Keeps the Text Encoder and Flow Decoder OMs resident across requests.
- Runs four host-scheduled Euler flow steps.
- Runs Vocos with CPU PyTorch.
- Produces 24 kHz mono WAV PCM16.
- Supports Chinese, Arabic numbers, and common Chinese/English punctuation.
- Rejects ASCII English words explicitly.
- Uses a fixed bundle prompt and does not support request-scoped voice cloning.

`ZipVoiceAscendSession` subclasses the shared `AscendOmModelSession` and uses standard `session.infer()` to reuse
the ACL lease, OM resources, admission, health, and close waiting without initializing a second global ACL runtime.

## 8. Stable Error Codes

| Code | Meaning |
| --- | --- |
| `INVALID_TEXT` | Text is empty after normalization |
| `INVALID_PROMPT_PAIR` | Prompt audio, format, and transcript are not supplied together |
| `INVALID_PROMPT_AUDIO` | The prompt WAV cannot be decoded or is unsupported |
| `PROMPT_TOO_LARGE` | The prompt exceeds its byte or duration limit |
| `REQUEST_TOO_LARGE` | Text or segment count exceeds request limits |
| `RESPONSE_TOO_LARGE` | Synthesized audio exceeds the response-byte limit |
| `MODEL_NOT_READY` | Bundle, deployment, or model-session loading failed |
| `INFERENCE_FAILED` | Model inference failed |
| `INVALID_AUDIO_OUTPUT` | Model output is empty or contains NaN/Inf |
| `UNSUPPORTED_PROMPT` | The selected deployment does not support request-scoped voice cloning |
| `INTERNAL_ERROR` | An unclassified internal service failure |

## 9. Tests and Build

Load the project environment before running commands:

```bash
source .shrc_local
```

Run package tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q src/voice_tts_service/test
```

Build the package:

```bash
colcon build --symlink-install --merge-install --packages-select voice_tts_service
```
