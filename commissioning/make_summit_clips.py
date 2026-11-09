"""Extract the exhibition clips of aimdk_x2_summit from the recorded audio package.

The WAV files stay out of the repository. Run this once per checkout, then copy
commissioning/audio/summit/ to the robot's checkout:

    python3 commissioning/make_summit_clips.py /path/to/x2语音.zip
"""

import argparse
import io
import re
import wave
import zipfile
from pathlib import Path

# Package line id -> logical clip name in embodied.runtime_play_audio.clips.
CLIPS = {
    "W01": "greeting",
    "W04": "invitation",
    "I01": "self_intro",
    "I02": "intro_platform",
    "I05": "intro_agent",
    "I10": "pick_scope",
    "F01": "fruit_menu",
    "W07": "mimic_unavailable",
}
# Joined with a pause. I06C is left out: it describes placing the fruit into the pickup box.
DETAIL_PARTS = ("I06A", "I06B")
DETAIL_PAUSE_SEC = 0.4
RATE = 24000
LINE_ID = re.compile(r"([A-Z]\d+[A-C]?)_")


def _read_lines(package: zipfile.ZipFile) -> dict[str, bytes]:
    lines = {}
    for info in package.infolist():
        match = LINE_ID.match(Path(info.filename).name)
        if match is None:
            continue
        with wave.open(io.BytesIO(package.read(info))) as stream:
            params = stream.getparams()
            if (params.nchannels, params.sampwidth, params.framerate) != (1, 2, RATE):
                raise ValueError(f"{info.filename}: expected {RATE} Hz mono 16-bit, got {params}")
            lines[match.group(1)] = stream.readframes(params.nframes)
    return lines


def _write(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(pcm)
    print(f"{path.name}: {len(pcm) / (2 * RATE):.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", type=Path, help="the recorded audio zip")
    default_output = Path(__file__).resolve().parent / "audio" / "summit"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    with zipfile.ZipFile(args.package) as package:
        lines = _read_lines(package)
    args.output.mkdir(parents=True, exist_ok=True)
    for line_id, name in CLIPS.items():
        _write(args.output / f"{name}.wav", lines[line_id])
    pause = bytes(2 * int(DETAIL_PAUSE_SEC * RATE))
    _write(args.output / "intro_detail.wav", pause.join(lines[part] for part in DETAIL_PARTS))


if __name__ == "__main__":
    main()
