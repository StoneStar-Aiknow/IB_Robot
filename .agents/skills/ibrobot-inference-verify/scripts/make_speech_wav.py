#!/usr/bin/env python3
"""Generate a 6ch 16kHz speech WAV for speech_direction verification.

Silero VAD rejects synthetic audio (harmonic/formant synthesis scores <0.6),
so this pulls REAL speech from the LibriSpeech dummy dataset (HF hub) and
pre-scores it with the repo's SileroVadEngine to guarantee VAD activation.

Usage: python3 make_speech_wav.py [-o /tmp/opencode/speech6ch.wav]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SILERO_ONNX = REPO_ROOT / "models/voice_asr/silero-vad/silero_vad_v5.onnx"


def score(sig: np.ndarray) -> float:
    sys.path.insert(0, str(REPO_ROOT / "src/voice_asr_service"))
    from voice_asr_service.speech_direction.speech_gate import SileroVadEngine

    eng = SileroVadEngine(model_path=str(SILERO_ONNX), sample_rate=16000, backend="onnx")
    probs = np.array([eng.inference(sig[i : i + 512]) for i in range(0, len(sig) - 511, 512)])
    return float(np.mean(probs > 0.65))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", default="/tmp/opencode/speech6ch.wav")
    args = p.parse_args()

    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
    best_sig, best_frac = None, 0.0
    for i in range(min(6, len(ds))):
        audio = ds[i]["audio"]
        sig = np.asarray(audio["array"], dtype=np.float32)
        if audio["sampling_rate"] != 16000:
            from scipy.signal import resample_poly

            sig = resample_poly(sig, 16000, audio["sampling_rate"]).astype(np.float32)
        sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
        frac = score(sig)
        print(f"sample {i}: {len(sig) / 16000:.1f}s frac>0.65={frac:.2f}", flush=True)
        if frac > best_frac:
            best_sig, best_frac = sig, frac
    if best_frac < 0.05:
        raise SystemExit("no sample reliably triggers Silero VAD")

    # repeat to ~3x length with pauses so the segment state machine fires
    gap = np.zeros(int(0.4 * 16000), np.float32)
    long_sig = np.concatenate([best_sig, gap, best_sig, gap, best_sig])
    six = np.repeat(long_sig[:, None], 6, axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, six, 16000, subtype="PCM_16")
    print(f"wrote {out} shape={six.shape} dur={len(long_sig) / 16000:.1f}s frac>0.65={best_frac:.2f}")


if __name__ == "__main__":
    main()
