#!/usr/bin/env python3
"""ROS 2 node exposing the typed Voice TTS service."""

from __future__ import annotations

import logging
import threading
from array import array
from contextlib import suppress

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from ibrobot_msgs.msg import ModelRuntimeInfo, SynthesizedAudio
from ibrobot_msgs.srv import SynthesizeSpeech
from inference_service.backends import BackendError, RuntimeContext
from voice_tts_service.defaults import VOICE_TTS_DEFAULTS
from voice_tts_service.errors import TTSError
from voice_tts_service.model_manager import TTSBundle, load_tts_bundle
from voice_tts_service.service_core import TTSLimits, TTSServiceCore
from voice_tts_service.zipvoice_310p_adapter import ZipVoiceAscendSession


class VoiceTTSNode(Node):
    """Keep one shared ZipVoice model session resident until explicit unload."""

    def __init__(self) -> None:
        super().__init__("voice_tts_node")
        self._declare_parameters()
        self._bundle: TTSBundle | None = None
        self._session = None
        self._init_error = ""
        self._session_lock = threading.RLock()
        self._core = TTSServiceCore(None, self._limits)
        self._prepare_bundle()
        if self._load_on_startup:
            with self._session_lock:
                self._load_model_locked()
        group = ReentrantCallbackGroup()
        self._service = self.create_service(
            SynthesizeSpeech, self._service_name, self._on_synthesize, callback_group=group
        )
        self._load_service = self.create_service(Trigger, self._load_service_name, self._on_load, callback_group=group)
        self._unload_service = self.create_service(
            Trigger, self._unload_service_name, self._on_unload, callback_group=group
        )
        self.get_logger().info(f"Voice TTS service ready at {self._service_name}")

    def _declare_parameters(self) -> None:
        for name, default in VOICE_TTS_DEFAULTS.items():
            if name != "enabled":
                self.declare_parameter(name, default)
        self._bundle_path = str(self.get_parameter("bundle_path").value)
        self._deployment = str(self.get_parameter("deployment").value)
        self._service_name = str(self.get_parameter("service_name").value)
        self._load_service_name = str(self.get_parameter("load_service_name").value)
        self._unload_service_name = str(self.get_parameter("unload_service_name").value)
        self._load_on_startup = bool(self.get_parameter("load_on_startup").value)
        for name, value in (
            ("service_name", self._service_name),
            ("load_service_name", self._load_service_name),
            ("unload_service_name", self._unload_service_name),
        ):
            if not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute ROS service name")
        self._prompt_profile = str(self.get_parameter("prompt_profile").value)
        if not self._prompt_profile:
            raise ValueError("prompt_profile must be non-empty")
        self._device_id = int(self.get_parameter("device_id").value)
        self._exit_on_init_failure = bool(self.get_parameter("exit_on_init_failure").value)
        self._limits = TTSLimits(
            segment_max_chars=int(self.get_parameter("segment_max_chars").value),
            segment_pause_ms=int(self.get_parameter("segment_pause_ms").value),
            max_request_chars=int(self.get_parameter("max_request_chars").value),
            max_prompt_audio_bytes=int(self.get_parameter("max_prompt_audio_bytes").value),
            max_prompt_duration_sec=float(self.get_parameter("max_prompt_duration_sec").value),
            max_segments=int(self.get_parameter("max_segments").value),
            max_response_audio_bytes=int(self.get_parameter("max_response_audio_bytes").value),
        )
        if any(
            value <= 0
            for value in (
                self._limits.segment_max_chars,
                self._limits.max_request_chars,
                self._limits.max_prompt_audio_bytes,
                self._limits.max_prompt_duration_sec,
                self._limits.max_segments,
                self._limits.max_response_audio_bytes,
            )
        ):
            raise ValueError("Voice TTS request and response limits must be positive")
        if self._limits.segment_pause_ms < 0:
            raise ValueError("segment_pause_ms must be non-negative")

    def _prepare_bundle(self) -> None:
        try:
            self._bundle = load_tts_bundle(self._bundle_path, self._deployment)
            self._init_error = ""
        except Exception as exc:
            self._bundle = None
            self._init_error = str(exc)
            self.get_logger().error(f"Voice TTS bundle validation failed: {exc}")
            if self._exit_on_init_failure:
                raise

    def _new_session(self):
        if self._bundle is None:
            self._bundle = load_tts_bundle(self._bundle_path, self._deployment)
        if self._bundle.validated.deployment.backend != "ascend":
            raise RuntimeError("Voice TTS currently requires an Ascend deployment")
        session = ZipVoiceAscendSession(device_id=self._device_id, prompt_profile=self._prompt_profile)
        session.load(
            RuntimeContext(
                validated_manifest=self._bundle.validated,
                runtime_options={"device_id": self._device_id},
            )
        )
        return session

    def _load_model_locked(self) -> bool:
        if self._session is not None:
            return False
        session = None
        try:
            session = self._new_session()
            self._session = session
            self._core.infer = session.infer
            self._init_error = ""
            self.get_logger().info(f"Loaded Voice TTS deployment {self._deployment!r}")
            return True
        except Exception as exc:
            self._init_error = str(exc)
            if session is not None:
                with suppress(Exception):
                    session.close()
            raise

    def _unload_model_locked(self) -> bool:
        session = self._session
        if session is None:
            self._init_error = ""
            self._core.infer = None
            return False
        try:
            session.close()
        except Exception as exc:
            self._init_error = str(exc)
            raise
        finally:
            self._session = None
            self._core.infer = None
        self._init_error = ""
        self.get_logger().info(f"Unloaded Voice TTS deployment {self._deployment!r}")
        return True

    def _runtime_info(self) -> ModelRuntimeInfo:
        result = ModelRuntimeInfo()
        result.instance_id = self.get_name()
        if self._bundle is not None:
            validated = self._bundle.validated
            result.model_name = validated.manifest.bundle.name
            result.model_version = str(validated.manifest.bundle.revision)
            result.manifest_fingerprint = validated.manifest.bundle.digest.value
            result.deployment_name = validated.deployment_name
            result.deployment_fingerprint = validated.fingerprint
            result.backend = validated.deployment.backend
        session = self._session
        if session is None:
            result.runtime_state = "failed" if self._init_error else "unloaded"
            result.failure_reason = self._init_error
        else:
            health = session.health()
            result.runtime_state = health.state.value
            result.ready = health.ready
            result.failure_reason = health.message or health.reason_code or ""
            result.runtime_version = session.runtime_version
        result.message = result.failure_reason
        return result

    def _failure(self, response, code: str, message: str):
        response.success = False
        response.error_code = code
        response.message = message
        response.audio_segments = []
        response.total_duration_sec = 0.0
        response.total_inference_time_ms = 0.0
        response.model = self._runtime_info()
        return response

    def _on_synthesize(self, request, response):
        try:
            prepared = self._core.prepare_request(
                request.text, request.prompt_audio, request.prompt_audio_format, request.prompt_text
            )
            with self._session_lock:
                self._load_model_locked()
                output = self._core.synthesize_prepared(prepared)
            response.audio_segments = [
                SynthesizedAudio(
                    index=segment.index,
                    text=segment.text,
                    audio_data=array("B", segment.wav_data),
                    audio_format="wav_pcm_s16le",
                    sample_rate=segment.sample_rate,
                    channels=1,
                    duration_sec=segment.duration_sec,
                    inference_time_ms=segment.inference_time_ms,
                    pause_after_ms=segment.pause_after_ms,
                )
                for segment in output.segments
            ]
            response.success = True
            response.error_code = ""
            response.message = f"synthesized {len(output.segments)} audio segment(s)"
            response.total_duration_sec = output.total_duration_sec
            response.total_inference_time_ms = output.total_inference_time_ms
            response.model = self._runtime_info()
            return response
        except TTSError as exc:
            return self._failure(response, exc.code, str(exc))
        except BackendError as exc:
            code = "UNSUPPORTED_PROMPT" if exc.code == "unsupported_prompt" else "INFERENCE_FAILED"
            return self._failure(response, code, str(exc))
        except Exception as exc:
            self.get_logger().error(f"Voice TTS internal error: {exc}")
            return self._failure(response, "INTERNAL_ERROR", str(exc))

    def _on_load(self, _request, response):
        try:
            with self._session_lock:
                loaded = self._load_model_locked()
            response.success = True
            response.message = "Voice TTS model loaded" if loaded else "Voice TTS model is already loaded"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _on_unload(self, _request, response):
        try:
            with self._session_lock:
                unloaded = self._unload_model_locked()
            response.success = True
            response.message = "Voice TTS model unloaded" if unloaded else "Voice TTS model is already unloaded"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def destroy_node(self):
        if hasattr(self, "_session_lock"):
            with suppress(Exception), self._session_lock:
                self._unload_model_locked()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = VoiceTTSNode()
    except Exception as exc:
        logging.getLogger("voice_tts_node").error(f"Node initialization failed: {exc}")
        if rclpy.ok():
            rclpy.shutdown()
        return 1
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown(timeout_sec=0.0)
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
