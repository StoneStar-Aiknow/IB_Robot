"""Transport binding and private local state helpers.

Target selection is gone: a caller binds one explicit transport for the whole
process lifetime. What remains here is the ROS environment scope (part of the
request identity), private JSON persistence used for execution references and
the legacy adapter that maps a transport to ROS bridge parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from .errors import ClientError
from .transport import GatewayTransport

#: Environment variables that decide which ROS graph a client can see. Two
#: deployments with identical endpoints on different graphs are different
#: targets, so the scope participates in every request identity.
_SCOPE_KEYS = (
    "ROS_DOMAIN_ID",
    "RMW_IMPLEMENTATION",
    "ROS_LOCALHOST_ONLY",
    "ROS_DISCOVERY_SERVER",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_STATIC_PEERS",
    "CYCLONEDDS_URI",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "ROS_SECURITY_ENABLE",
    "ROS_SECURITY_STRATEGY",
    "ROS_SECURITY_KEYSTORE",
    "ROS_SECURITY_ENCLAVE_OVERRIDE",
)


def ros_environment_scope() -> str:
    """Return a stable digest of the ROS environment a client is bound to."""
    values = {key: os.environ.get(key, "") for key in _SCOPE_KEYS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def read_private_json(path: Path) -> dict:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return {}
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ClientError("UNSAFE_LOCAL_STATE", f"{path} must be a private file owned by this user")
        text = stream.read(262145)
    if len(text) > 262144:
        raise ClientError("INVALID_LOCAL_STATE", "local connection state exceeds size limit")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ClientError("INVALID_LOCAL_STATE", "local connection state must be an object")
    return value


def write_private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ClientError("UNSAFE_LOCAL_STATE", f"{path.parent} must be private")
    descriptor, temporary = tempfile.mkstemp(prefix=".connection-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bridge_kwargs(transport: GatewayTransport) -> dict:
    """Return the ROS bridge parameters for one bound transport.

    ``GatewayTransport.endpoints`` owns the endpoint names, aliases and the
    contract key set, so the bridge cannot drift from the identity digest.
    """
    result = dict(transport.endpoints)
    result.update(robot_id=transport.robot_id, caller_scope=transport.caller_scope)
    return result
