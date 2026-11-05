"""Resolve and freeze the validation artifact before catalog compilation."""

import hashlib
from pathlib import Path

from robot_runtime.path_validation import normalize_path_validation, path_validation_identity


def resolve_path_validation(robot, config_path):
    composer = robot.get("pose_composer") or {}
    path_validation_identity(composer)
    if composer.get("mode") != "fixed":
        return
    policy = normalize_path_validation(composer.get("path_validation"))
    if policy.get("certificate_file"):
        path = Path(policy["certificate_file"]).expanduser()
        if not path.is_absolute():
            path = Path(config_path).parent / path
        path = path.resolve()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if policy["certificate_sha256"] and sha != policy["certificate_sha256"]:
            raise ValueError("path validation artifact changed since configuration binding")
        policy.update(certificate_file=str(path), certificate_sha256=sha)
    composer["path_validation"] = policy
