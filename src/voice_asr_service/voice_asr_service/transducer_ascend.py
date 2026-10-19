"""Streaming Zipformer2 transducer with the encoder on an Ascend OM.

sherpa-onnx has no Ascend path for streaming transducers, so the ``ascend`` ASR deployment
reimplements the sherpa ``OnlineRecognizer`` pipeline around the NPU encoder:

* features: kaldi fbank identical to sherpa (80 bins, 25/10 ms Povey frames, snip_edges=False,
  dither 0, preemphasis 0.97, DC removal, mel range [20, 8000-400] Hz, samples in [-1, 1]);
* encoder: static OM taking ``T`` frames and advancing ``decode_chunk_len``; its cached states
  stay on the device in two ping-pong dataset banks (same pattern as SileroVadAclRunner), only
  ``encoder_out`` is copied back;
* search: greedy transducer search with the small decoder/joiner on ONNX Runtime CPU.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FBANK_SAMPLE_RATE = 16000
_FRAME_LEN = 400
_FRAME_SHIFT = 160
_NUM_BINS = 80
_FFT = 512
_EPS = float(np.finfo(np.float32).eps)


def _mel(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _mel_banks() -> np.ndarray:
    low, high = 20.0, FBANK_SAMPLE_RATE / 2 - 400.0
    mel_low, mel_high = _mel(low), _mel(high)
    delta = (mel_high - mel_low) / (_NUM_BINS + 1)
    freqs = np.arange(_FFT // 2 + 1) * FBANK_SAMPLE_RATE / _FFT
    mel_freqs = _mel(freqs)
    banks = np.zeros((_NUM_BINS, _FFT // 2 + 1), dtype=np.float32)
    for index in range(_NUM_BINS):
        left, center, right = (mel_low + (index + offset) * delta for offset in range(3))
        weights = np.maximum(
            0.0, np.minimum((mel_freqs - left) / (center - left), (right - mel_freqs) / (right - center))
        )
        weights[(freqs < low) | (freqs > high)] = 0.0
        banks[index] = weights
    return banks


# Manifest semantics of the encoder OM bindings (written by package_sherpa_asr_bundle).
ENCODER_FEATURES_SEMANTIC = "host.asr.encoder.features"
ENCODER_OUT_SEMANTIC = "host.asr.encoder.out"
ENCODER_STATE_IN_PREFIX = "host.asr.encoder.state_in."
ENCODER_STATE_OUT_PREFIX = "host.asr.encoder.state_out."

_BANKS = _mel_banks()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_FRAME_LEN) / (_FRAME_LEN - 1))) ** 0.85


class OnlineFbank:
    """Incremental kaldi fbank (snip_edges=False) matching kaldi-native-fbank's online output.

    Frame ``i`` is centred at ``i * shift + shift / 2``. Before ``input_finished`` only frames whose
    window lies inside the received audio are emitted; the final call reflects the last samples
    exactly like the offline computation.
    """

    def __init__(self) -> None:
        self._samples = np.zeros(0, dtype=np.float64)
        self._frames: list[np.ndarray] = []
        self._finished = False

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def accept(self, samples: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("fbank input already finished")
        self._samples = np.concatenate([self._samples, np.asarray(samples, dtype=np.float64).reshape(-1)])
        n = len(self._samples)
        ready = (n - (_FRAME_LEN - _FRAME_LEN // 2 + _FRAME_SHIFT // 2)) // _FRAME_SHIFT + 1 if n >= 280 else 0
        self._compute(ready)

    def input_finished(self) -> None:
        self._finished = True
        self._compute((len(self._samples) + _FRAME_SHIFT // 2) // _FRAME_SHIFT)

    def frames(self, start: int, count: int) -> np.ndarray:
        return np.stack(self._frames[start : start + count])

    def _compute(self, ready: int) -> None:
        first = len(self._frames)
        if ready <= first:
            return
        n = len(self._samples)
        starts = np.arange(first, ready) * _FRAME_SHIFT + _FRAME_SHIFT // 2 - _FRAME_LEN // 2
        index = starts[:, None] + np.arange(_FRAME_LEN)[None, :]
        index = np.where(index < 0, -index - 1, index)
        index = np.where(index >= n, 2 * n - index - 1, index)
        frames = self._samples[index]
        frames = frames - frames.mean(axis=1, keepdims=True)
        frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
        frames[:, 0] -= 0.97 * frames[:, 0]
        frames *= _WINDOW
        energies = (np.abs(np.fft.rfft(frames, n=_FFT)) ** 2).astype(np.float32) @ _BANKS.T
        self._frames.extend(np.log(np.maximum(energies, _EPS)).astype(np.float32))


class ZipformerEncoderAclRunner:
    """Encoder OM with device-resident states: input 0 is the feature chunk, output 0 encoder_out.

    Inputs ``1..N`` are the cached states and output ``k`` is ``new_<state>``; the runner pairs
    them by name and alternates two dataset banks so states never leave the device.
    """

    def __init__(self, model_path: Path, bindings, *, device_id: int, runtime_manager) -> None:
        from inference_service.backends.ascend.model import AclModel

        if not model_path.is_file():
            raise FileNotFoundError(f"ASR encoder OM not found: {model_path}")
        self._lock = threading.Lock()
        self._lease = runtime_manager.acquire(device_id)
        self._model = None
        self._bank = 0
        self._states: list[tuple[object, object]] = []
        try:
            self._model = AclModel(self._lease, "encoder", model_path, bindings)
            self._model.load_descriptor()
            outputs = {b.semantic: b.index for b in bindings.outputs}
            self.input_shape = tuple(next(b.shape for b in bindings.inputs if b.semantic == ENCODER_FEATURES_SEMANTIC))
            bank_a_in, bank_a_out, bank_b_in, bank_b_out = {}, {}, {}, {}
            for binding in bindings.inputs:
                if not binding.semantic.startswith(ENCODER_STATE_IN_PREFIX):
                    continue
                input_index = binding.index
                output_index = outputs[
                    ENCODER_STATE_OUT_PREFIX + binding.semantic.removeprefix(ENCODER_STATE_IN_PREFIX)
                ]
                size = self._model.input_descriptors[input_index].size
                first, second = self._model.allocate_device_buffer(size), self._model.allocate_device_buffer(size)
                self._states.append((first, second))
                bank_a_in[input_index], bank_a_out[output_index] = first, second
                bank_b_in[input_index], bank_b_out[output_index] = second, first
            self._model.prepare_dataset_banks((bank_a_in, bank_b_in), (bank_a_out, bank_b_out), host_output_indices={0})
            self.reset()
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        with self._lock:
            for first, _ in self._states:
                self._model.zero_device_buffer(first)
            self._bank = 0

    def infer(self, features: np.ndarray) -> np.ndarray:
        with self._lock:
            out = self._model.execute_bank(self._bank, {0: np.ascontiguousarray(features, dtype=np.float32)})[0]
            self._bank = 1 - self._bank
        return out.reshape(-1, out.shape[-1])

    def close(self) -> None:
        errors = []
        for resource in (self._model, self._lease):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:  # best-effort teardown, report every failure
                errors.append(str(exc))
        self._model = None
        self._lease = None
        if errors:
            raise RuntimeError("; ".join(errors))


@dataclass
class _TransducerStream:
    fbank: OnlineFbank = field(default_factory=OnlineFbank)
    consumed: int = 0
    hyp: list[int] = field(default_factory=lambda: [0, 0])
    decoder_out: np.ndarray | None = None


class AscendTransducerRuntime:
    """Unified-runtime stream adapter (same surface as the sherpa runtime) over the NPU encoder."""

    def __init__(
        self,
        encoder: ZipformerEncoderAclRunner,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        *,
        decode_chunk_len: int,
        final_tail_padding_seconds: float,
        sample_rate: int = FBANK_SAMPLE_RATE,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._decoder = ort.InferenceSession(decoder_path, options, providers=["CPUExecutionProvider"])
        self._joiner = ort.InferenceSession(joiner_path, options, providers=["CPUExecutionProvider"])
        self._encoder = encoder
        self._window = encoder.input_shape[1]
        self._chunk = decode_chunk_len
        self._sample_rate = sample_rate
        self._tail = np.zeros(int(max(0.0, final_tail_padding_seconds) * sample_rate), dtype=np.float32)
        self._tokens: dict[int, str] = {}
        for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                self._tokens[int(parts[1])] = parts[0]

    def load(self, _context=None) -> None:
        return None

    def open_stream(self, _context) -> _TransducerStream:
        self._encoder.reset()
        return self._new_stream()

    def reset_stream(self, stream: _TransducerStream, _context) -> None:
        self._encoder.reset()
        fresh = self._new_stream()
        stream.fbank, stream.consumed, stream.hyp, stream.decoder_out = (
            fresh.fbank,
            fresh.consumed,
            fresh.hyp,
            fresh.decoder_out,
        )

    def close_stream(self, _stream: _TransducerStream, _context) -> None:
        return None

    def close(self) -> None:
        self._encoder.close()

    def step(self, stream: _TransducerStream, request, context) -> dict[str, object]:
        from voice_asr_service.asr_inference_module import normalize_transcript

        context.check("ascend_transducer.accept_waveform")
        audio = request.inputs.get("audio")
        if not isinstance(audio, np.ndarray):
            raise TypeError("ASR stream request requires a numpy audio input")
        final = bool(request.metadata.get("final", False))
        stream.fbank.accept(audio)
        if final:
            if self._tail.size:
                stream.fbank.accept(self._tail)
            stream.fbank.input_finished()
        while stream.consumed + self._window <= stream.fbank.num_frames:
            features = stream.fbank.frames(stream.consumed, self._window)[None]
            for frame in self._encoder.infer(features):
                self._greedy_step(stream, frame)
            stream.consumed += self._chunk
        text = "".join(self._tokens.get(token, "") for token in stream.hyp[2:]).replace("▁", " ")
        return {"text": normalize_transcript(text), "is_final": final}

    def _new_stream(self) -> _TransducerStream:
        stream = _TransducerStream()
        stream.decoder_out = self._run_decoder(stream.hyp)
        return stream

    def _run_decoder(self, hyp: list[int]) -> np.ndarray:
        return self._decoder.run(None, {"y": np.array([hyp[-2:]], dtype=np.int64)})[0]

    def _greedy_step(self, stream: _TransducerStream, frame: np.ndarray) -> None:
        logits = self._joiner.run(
            None, {"encoder_out": frame[None].astype(np.float32), "decoder_out": stream.decoder_out}
        )[0]
        token = int(np.argmax(logits[0]))
        if token != 0:
            stream.hyp.append(token)
            stream.decoder_out = self._run_decoder(stream.hyp)
