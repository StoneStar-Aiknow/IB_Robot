"""Offline same-access-unit RTP delivery metrics using captured clock mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .clock_calibration import load_calibration, reference_difference

_KEY = (
    "pipeline_id",
    "session_id",
    "session_generation",
    "observation_key",
    "stream_id",
    "ssrc",
    "rtp_timestamp",
    "last_sequence",
)
_STRING_KEYS = _KEY[:2] + _KEY[3:5]
_TIMES = {
    "send": ("send_start_ns", "send_end_ns"),
    "receive": ("receive_ns", "reassembly_end_ns"),
    "decode_call": ("decode_start_ns", "decode_end_ns"),
}


def read_metrics(path: Path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("RTP journal exceeds 64 MiB")
    records = []
    with path.open() as stream:
        line = stream.readline(65537)
        if len(line) > 65536 or not line.endswith("\n"):
            raise ValueError("incomplete or oversized RTP journal header")
        header = json.loads(line)
        if (
            not isinstance(header, dict)
            or header.get("kind") != "rtp_metrics"
            or header.get("schema_version") != 1
            or header.get("clock") != "monotonic_ns"
        ):
            raise ValueError("unsupported RTP journal schema")
        if any(
            not isinstance(header.get(key), str) or not header[key]
            for key in ("machine_id", "boot_id", "capture_session", "process_instance")
        ):
            raise ValueError("missing RTP journal clock provenance")
        footer = None
        for line in stream:
            if len(line) > 65536 or not line.endswith("\n"):
                raise ValueError("oversized RTP journal record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("invalid RTP record")
            if footer is not None:
                raise ValueError("records after journal footer")
            if row.get("kind") == "end":
                if type(row.get("dropped_records")) is not int or row["dropped_records"] < 0:
                    raise ValueError("invalid RTP journal footer")
                footer = row
                continue
            if row.get("kind") not in _TIMES:
                raise ValueError("unknown RTP record kind")
            if any(not isinstance(row.get(key), str) or not row[key] for key in _STRING_KEYS):
                raise ValueError("missing RTP frame identity")
            for key in (*[key for key in _KEY if key not in _STRING_KEYS], *_TIMES[row["kind"]]):
                if type(row.get(key)) is not int or not 0 <= row[key] < 2**63:
                    raise ValueError(f"invalid RTP field {key}")
            if (
                row["session_generation"] < 1
                or row["ssrc"] >= 2**32
                or row["rtp_timestamp"] >= 2**32
                or row["last_sequence"] >= 2**16
            ):
                raise ValueError("invalid RTP frame epoch/sequence")
            if row["kind"] in {"send", "receive"}:
                if type(row.get("first_sequence")) is not int or not 0 <= row["first_sequence"] < 65536:
                    raise ValueError("invalid first RTP sequence")
                if type(row.get("packet_count")) is not int or not 1 <= row["packet_count"] <= 65536:
                    raise ValueError("invalid RTP packet count")
            if len(records) >= 100_000:
                raise ValueError("RTP journal record limit exceeded")
            records.append(row)
    return header, records, footer


def analyze_rtp_metrics(inputs: list[tuple[Path, Path]]) -> dict:
    if not 2 <= len(inputs) <= 32:
        raise ValueError("need 2..32 journal/calibration pairs")
    groups = defaultdict(lambda: defaultdict(list))
    warnings = []
    references = set()
    source_paths = set()
    total = 0
    sources = []
    for journal, calibration_path in inputs:
        if journal.resolve() in source_paths:
            raise ValueError("duplicate RTP input")
        source_paths.add(journal.resolve())
        cal = load_calibration(calibration_path)
        references.add(cal.identity.reference_id)
        header, records, footer = read_metrics(journal)
        sources.append(
            {
                "journal": str(journal),
                "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                "calibration": str(calibration_path),
                "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                "calibration_policy": vars(cal.policy),
                "record_count": len(records),
                "footer": footer,
            }
        )
        total += len(records)
        if total > 100_000:
            raise ValueError("combined RTP record limit exceeded")
        if header["capture_session"] != cal.identity.session_id:
            raise ValueError("journal/calibration capture session mismatch")
        if footer is None or footer.get("dropped_records", 0):
            warnings.append(f"{journal}: incomplete capture or dropped journal records; only unambiguous pairs qualify")
        for row in records:
            groups[tuple(row[key] for key in _KEY)][row["kind"]].append((row, header, cal, str(journal)))
    if len(references) != 1:
        raise ValueError("RTP calibrations have different reference sessions")

    def mapped(entry, field):
        row, header, cal, _ = entry
        return cal.map_time(
            row[field], clock="monotonic_ns", machine_id=header["machine_id"], boot_id=header["boot_id"]
        )

    result = []
    for key, kinds in sorted(groups.items()):
        row = {**dict(zip(_KEY, key, strict=True)), "status": "unavailable"}
        row["records"] = {
            kind: [{"source": item[3], "record": item[0], "clock_identity": item[1]} for item in items]
            for kind, items in kinds.items()
        }
        if len(kinds["send"]) != 1 or len(kinds["receive"]) != 1:
            row["reason"] = (
                "ambiguous_frame" if len(kinds["send"]) > 1 or len(kinds["receive"]) > 1 else "unmatched_frame"
            )
            result.append(row)
            continue
        send, receive = kinds["send"][0], kinds["receive"][0]
        if (send[0]["first_sequence"], send[0]["packet_count"]) != (
            receive[0]["first_sequence"],
            receive[0]["packet_count"],
        ):
            row["reason"] = "incomplete_access_unit"
            result.append(row)
            continue
        start, end = mapped(send, "send_start_ns"), mapped(receive, "receive_ns")
        send_end = mapped(send, "send_end_ns")
        reassembled = mapped(receive, "reassembly_end_ns")
        if (
            not start.valid
            or not end.valid
            or not send_end.valid
            or not reassembled.valid
            or send_end.segment_id != start.segment_id
            or reassembled.segment_id != end.segment_id
        ):
            row["reason"] = (
                start.reason or end.reason or send_end.reason or reassembled.reason or "clock_segment_mismatch"
            )
        elif (
            send[0]["send_end_ns"] < send[0]["send_start_ns"]
            or receive[0]["reassembly_end_ns"] < receive[0]["receive_ns"]
        ):
            row["reason"] = "invalid_local_order"
        else:
            delta = reference_difference(start, end, send[2], receive[2])
            if not delta.valid:
                row["reason"] = delta.reason
            elif delta.timestamp_ns <= delta.uncertainty_ns:
                row["reason"] = "negative_or_unresolved_order"
                row["estimated_difference_ns"] = delta.timestamp_ns
                row["uncertainty_ns"] = delta.uncertainty_ns
            else:
                row.update(
                    status="valid",
                    frame_delivery_ns=delta.timestamp_ns,
                    uncertainty_ns=delta.uncertainty_ns,
                    send_call_ns=send[0]["send_end_ns"] - send[0]["send_start_ns"],
                    receive_queue_and_reassembly_ns=receive[0]["reassembly_end_ns"] - receive[0]["receive_ns"],
                )
                sender = send[0]
                if sender.get("encode_input_capture_ns") == sender.get("capture_timestamp_ns"):
                    a, b = sender.get("encode_start_ns"), sender.get("encode_end_ns")
                    if type(a) is int and type(b) is int and b >= a > 0:
                        row["encode_call_ns"] = b - a
                if len(kinds["decode_call"]) == 1:
                    decode = kinds["decode_call"][0]
                    if decode[1] == receive[1] and decode[0]["decode_end_ns"] >= decode[0]["decode_start_ns"]:
                        row["decode_call_ns"] = decode[0]["decode_end_ns"] - decode[0]["decode_start_ns"]
                row["interval"] = "sender packetization start -> receiver last RTP packet enqueue"
        result.append(row)
    return {
        "schema_version": 1,
        "reference_id": list(next(iter(references))),
        "sources": sources,
        "frames": result,
        "counts": dict(Counter(row.get("reason", row["status"]) for row in result)),
        "warnings": warnings,
        "quality": "conditional calibration uncertainty; application interval, not wire-only network latency",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("JOURNAL", "CALIBRATION"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze_rtp_metrics([(Path(journal), Path(cal)) for journal, cal in args.input])
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    print(
        json.dumps({"output": str(args.output), "counts": result["counts"], "warnings": result["warnings"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
