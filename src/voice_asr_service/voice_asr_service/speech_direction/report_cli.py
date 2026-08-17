"""生成 speech_direction 离线维测报告的命令行入口。"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from .diagnostics.audio_extract import AudioCoverageError, extract_gray_audio
from .diagnostics.report import (
    DiagnosticsReport,
    ReportTransactionError,
    ReportValidationError,
    load_session_report_data,
    require_report_dependencies,
)

EXPECTED_ERRORS = (
    ImportError,
    ReportValidationError,
    ReportTransactionError,
    AudioCoverageError,
    FileExistsError,
    OSError,
)


def build_parser() -> argparse.ArgumentParser:
    """构造离线报告命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="生成 speech_direction 离线 HTML/PNG 报告")
    parser.add_argument("session_dir", type=Path, help="待分析的维测会话目录")
    parser.add_argument(
        "--extract-gray-audio",
        action="store_true",
        help="报告生成成功后截取灰区音频",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖既有报告与灰区音频",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行依赖预检、会话校验、报告生成及可选灰区截取。"""
    args = build_parser().parse_args(argv)
    try:
        # 捕获整个成功流水线的运行时 warning，但异常仍按明确用户错误边界处理。
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            require_report_dependencies()
            data = load_session_report_data(args.session_dir)
            report_paths = DiagnosticsReport(args.session_dir).generate(overwrite=args.overwrite, data=data)

            audio_paths: tuple[Path, ...] = ()
            if args.extract_gray_audio:
                # 灰区截取复用报告阶段的同一份会话快照。
                audio_paths = extract_gray_audio(
                    data,
                    output_dir=data.session_dir / "reports" / "gray_audio",
                    overwrite=args.overwrite,
                )
        captured_warnings = list(caught)
    except EXPECTED_ERRORS as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    for output_kind in ("html", "png"):
        print(report_paths[output_kind])
    for audio_path in audio_paths:
        print(audio_path)
    for message in data.warnings:
        print(f"warning: {message}")
    for captured in captured_warnings:
        print(f"warning: {captured.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
