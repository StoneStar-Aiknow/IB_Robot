# Voice TTS Service

[中文](README.md)

`voice_tts_service` is IB-Robot's ZipVoice model-service plugin. It is hosted by the shared
`inference_service/model_service_node`, accepts text and an optional
voice prompt, invokes an explicitly selected ZipVoice deployment, returns independently playable mono WAV PCM16
audio segments, and exposes local speaker playback on the machine hosting the service.

This package converts text to audio and plays a server-local WAV file. ASR, business orchestration, remote file
transfer, and remote inference over SSH are outside its boundary.

## 1. Responsibilities

The package:

1. Exposes a unified ROS 2 TTS service contract.
2. Validates the ZipVoice model bundle, manifest, and named deployment.
3. Uses the shared model-service host for loading, resident model reuse, and cleanup.
4. Applies bounded long-text segmentation and wraps model output as WAV PCM16.
5. Orchestrates the Text Encoder OM, Flow Decoder OM, and CPU Vocos on Ascend 310P.
6. Enforces request, segment-count, and response-size limits.
7. Publishes validated local WAV files through the shared `audio_common` playback path and reports a stable result.

It does not manage microphone capture, device discovery, hotplug, mixing, ASR, dialogue state, business workflows,
backend fallback, or runtime inference through SSH.

## 2. Entry Points

| Item | Path or entry point |
| --- | --- |
| Shared ROS host | `inference_service/inference_service/model_service_node.py` |
| TTS plugin | `voice_tts_service/voice_tts_service/model_service_plugin.py` |
| Audio playback node | `voice_tts_service/voice_tts_service/audio_playback_node.py` |
| 310P adapter | `voice_tts_service/zipvoice_310p_adapter.py` |
| ONNX adapter (Ubuntu) | `voice_tts_service/zipvoice_onnx_adapter.py` |
| Bundle packager | `voice_tts_service/package_zipvoice_310p.py` |
| Debug launch | `launch/voice_tts.launch.py` |
| Console entry | `model_service_node = inference_service.model_service_node:main` |
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

Ubuntu hosts can use the ONNX backend (onnxruntime + CPU Vocos) without Ascend hardware:

```bash
source .shrc_local
ros2 launch voice_tts_service voice_tts.launch.py \
  bundle_path:=/path/to/zipvoice-bundle \
  deployment:=ubuntu_onnx
```

## 3. Data Flow

```text
SynthesizeSpeech request
  -> request validation and text segmentation
  -> ModelRequest + ExecutionContext
  -> ModelRuntimeHandle.execute
  -> resident ZipVoice ModelSession resource loaded at startup
  -> ZipVoice tokenizer
  -> Text Encoder OM
  -> Flow Decoder OM (four host-scheduled Euler steps)
  -> CPU Vocos
  -> mono float PCM
  -> WAV PCM16 encoding
  -> SynthesizedAudio[] response
```

Playback is independent of model inference:

```text
PlayAudioFile request (absolute path on the playback host)
  -> validate WAV file
  -> audio_common audio_play (shared by Ubuntu and openEuler)
  -> success / error_code / message
```

`ZipVoiceSynthesizePlugin` places its `ModelSession` resource in a `RuntimeAssembly` and transfers ownership to a
`ModelRuntimeHandle`. The handle serializes admission and owns public lifecycle, health, cancellation, and close
draining; the session only owns and releases ZipVoice vendor model and device resources.

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

### 4.2 Audio playback

| Item | Value |
| --- | --- |
| Service | `/voice_tts/play` |
| Type | `ibrobot_msgs/srv/PlayAudioFile` |
| Input | Absolute WAV path on the machine hosting the playback service |

The call blocks until playback completes and then returns `success=true`. A missing path, invalid WAV, unavailable
playback node, or timeout returns `success=false` with a stable `error_code` and message.
The service does not fetch files over SSH or interpret a path from the caller's machine.

```bash
ros2 service call /voice_tts/play ibrobot_msgs/srv/PlayAudioFile \
  "{file_path: '/tmp/voice_tts/output.wav'}"
```

### 4.3 Model lifecycle

| Stage | Owner | Behavior |
| --- | --- | --- |
| Node startup | Shared host + TTS plugin | Validate the bundle, construct the session, `RuntimeAssembly`, and handle, then call `handle.load()` |
| Node shutdown | Plugin's `ModelRuntimeHandle` | Stop admission, drain active inference, and close the assembly-owned session resource |

The shared model-service host loads the named deployment at node startup:

```text
node startup -> validate bundle -> construct RuntimeAssembly/ModelRuntimeHandle -> handle.load()
synthesize -> handle.execute(ModelRequest, ExecutionContext) -> reuse the resident ModelSession resource
node shutdown -> handle.close() -> session releases OM resources, ACL leases, Vocos, tokenizer, and prompt
```

`ModelRuntimeHandle` owns admission, public lifecycle, health, cancellation, and close draining. `ModelSession` is a
handle-owned resource responsible for loading, executing, and releasing ZipVoice vendor resources. The handle waits
for active inference when host shutdown closes the plugin.

`exit_on_init_failure=false` (the shared host's `required=false`) only keeps the typed endpoint online and reports
`MODEL_NOT_READY` after initialization failure. The host does not retry initialization on later requests; restart the
node after repairing the bundle, dependency, or device. Request-scoped errors such as `INVALID_TEXT` and
`UNSUPPORTED_PROMPT` do not change handle health or block later valid requests.

## 5. Model Bundle and Deployment

The model path is configured through `robot_config.bundle_path`; it is never hard-coded. Relative paths resolve
against the absolute `WORKSPACE` set by `.shrc_local`:

```yaml
bundle_path: models/zipvoice
deployment: ascend_310p
```

This selects `$WORKSPACE/models/zipvoice`. The bundle contains `inference_manifest.json`, deployment
artifacts, tokenizer assets, prompt profiles, and the vocoder checkpoint.

The bundle must use manifest schema v3, declare `interface=tensor_model`, `model_type=zipvoice`, and
`operation=synthesize`, and contain the
selected named deployment.

### 5.1 Verified deployments

| deployment | backend | runtime | description |
| --- | --- | --- | --- |
| `ascend_310p` | ascend | ACL + OM | Orchestrates Text Encoder OM, Flow Decoder OM, and CPU Vocos on 310P |
| `ubuntu_onnx` | torch | onnxruntime + CPU | Loads upstream ONNX models via onnxruntime on Ubuntu, reusing 310P bundle tokens/Vocos/prompt assets |

The `ubuntu_onnx` deployment's ONNX models are fetched from [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice)
(ModelScope mirror) via `scripts/download_voice_tts_models.sh`, which also generates `zipvoice_onnx.json` and
`inference_manifest.json` so the bundle is immediately ready for `deployment:=ubuntu_onnx`.

The bundle digest and deployment fingerprint identify structure and deployment consistency; they do not read
model contents or provide runtime tamper protection. Before copying, the 310P packager verifies the SHA-256 of
known source OMs, the Vocos checkpoint, token table, and golden fixtures. Runtime validates the manifest, path
safety, file presence, and model ABI. Use signed read-only images or verity when production deployments require
content authenticity. The runtime never infers a backend from the operating system and never falls back after
load failure.

### 5.2 Packaging an Ascend 310P delivery

Large model files are not committed to Git. Package a prepared ZipVoice delivery into the standard bundle:

```bash
source .shrc_local
ZIPVOICE_SOURCE_DIR=/path/to/zipvoice-delivery
ros2 run voice_tts_service package_zipvoice_310p \
  --source "$ZIPVOICE_SOURCE_DIR" \
  --destination "$WORKSPACE/models/zipvoice"
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
  bundle_path: models/zipvoice
  deployment: ascend_310p

  service_name: /voice_tts/synthesize
  playback_service_name: /voice_tts/play
  playback_timeout_sec: 300.0

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
- Input normalization removes Markdown formatting and control characters. Emoji and other unspoken symbols are
  removed when adjacent to punctuation, otherwise replaced with a period to retain a pause. Line breaks also become
  periods.
- Rejects ASCII English words explicitly.
- Uses a fixed bundle prompt and does not support request-scoped voice cloning.

`ZipVoiceAscendSession` subclasses the shared `AscendOmModelSession` and is the `ModelSession` resource in the
plugin's `RuntimeAssembly`. It owns the ACL lease, OM resources, Vocos, tokenizer, and prompt. The plugin calls
`ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)`; the handle owns admission, lifecycle, health,
cancellation, and close waiting without initializing a second global ACL runtime.

## 8. Stable Error Codes

| Code | Meaning |
| --- | --- |
| `INVALID_TEXT` | Text is empty after normalization |
| `INVALID_PROMPT_PAIR` | Prompt audio, format, and transcript are not supplied together |
| `INVALID_PROMPT_AUDIO` | The prompt WAV cannot be decoded or is unsupported |
| `PROMPT_TOO_LARGE` | The prompt exceeds its byte or duration limit |
| `REQUEST_TOO_LARGE` | Text or segment count exceeds request limits |
| `RESPONSE_TOO_LARGE` | Synthesized audio exceeds the response-byte limit |
| `MODEL_NOT_READY` | Bundle, deployment, or model-runtime loading failed |
| `INFERENCE_FAILED` | Model inference failed |
| `INVALID_AUDIO_OUTPUT` | Model output is empty or contains NaN/Inf |
| `UNSUPPORTED_PROMPT` | The selected deployment does not support request-scoped voice cloning |
| `INTERNAL_ERROR` | An unclassified internal service failure |

The playback endpoint additionally returns `INVALID_PATH`, `FILE_NOT_FOUND`, `NOT_A_FILE`, `UNSUPPORTED_FORMAT`,
`INVALID_AUDIO_FILE`, `PLAYER_NOT_FOUND`, `PLAYBACK_TIMEOUT`, or `PLAYBACK_FAILED`.

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
