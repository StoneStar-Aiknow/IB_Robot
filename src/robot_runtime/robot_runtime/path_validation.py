"""Startup-only fixed-pose validation policy, shared by catalog and executors."""

import hashlib
import json
from collections.abc import Mapping


def policy_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def normalize_path_validation(value=None):
    if value is None:
        value = {}
    if not isinstance(value, Mapping) or set(value) - {"hand", "body", "certificate_file", "certificate_sha256"}:
        raise ValueError("path_validation must contain only hand/body/certificate_file/certificate_sha256")
    result = {}
    for part in ("hand", "body"):
        mode = value.get(part, "online")
        if not isinstance(mode, str) or mode not in ("online", "off", "prevalidated"):
            raise ValueError(f"path_validation.{part} must be online, off or prevalidated")
        result[part] = mode
    certificate = value.get("certificate_file", "")
    sha = value.get("certificate_sha256", "")
    if not isinstance(certificate, str) or not isinstance(sha, str):
        raise ValueError("path_validation certificate path/digest must be strings")
    if sha and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError("invalid path_validation certificate_sha256")
    if "prevalidated" in result.values() and not certificate:
        raise ValueError("prevalidated path_validation requires certificate_file")
    if certificate:
        result.update(certificate_file=certificate, certificate_sha256=sha)
    elif sha:
        raise ValueError("certificate_sha256 requires certificate_file")
    return result


def path_validation_identity(composer):
    composer = composer or {}
    if composer.get("mode", "generated") != "fixed":
        if "path_validation" in composer:
            raise ValueError("path_validation is supported only in fixed mode")
        return {}
    policy = normalize_path_validation(composer.get("path_validation"))
    # Deployment paths are not execution semantics; exact file content is.
    policy.pop("certificate_file", None)
    return {"path_validation_digest": policy_digest(policy)}
