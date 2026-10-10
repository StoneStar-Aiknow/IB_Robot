"""DDS YAML validation and atomic publication; no shell changes."""

import fcntl
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml

BACKENDS = {"fastdds", "cyclonedds"}
MAX_BYTES = 1024 * 1024


class ConfigError(ValueError):
    """A configuration or artifact contract was violated."""


class StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key == "<<" or key in result:
                raise ConfigError("Mapping keys must be unique strings; YAML merges are forbidden")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def mapping(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ConfigError("Expected a mapping")
    if set(value) - set(allowed) or set(required) - set(value):
        raise ConfigError(f"Unexpected or missing fields: allowed={sorted(allowed)}, required={sorted(required)}")
    return value


def literal(value):
    if not isinstance(value, str) or not value or any(c in value for c in ("$", "\x00", "\n", "\r")):
        raise ConfigError("Expected a nonempty literal string without environment substitution")
    return value


def state_root(home):
    home = Path(literal(home))
    if not home.is_absolute():
        raise ConfigError("HOME must be an absolute path")
    return home / ".local/state/ibrobot/dds"


def ipv4(value):
    try:
        address = ipaddress.IPv4Address(literal(value))
    except ValueError as exc:
        raise ConfigError("Expected an IPv4 address") from exc
    if address.is_multicast or address.is_unspecified or int(address) == 0xFFFFFFFF:
        raise ConfigError("Expected a unicast IPv4 address")
    return str(address)


def parse_config(text):
    """Normalize supported generated profiles; interface/routes are checked next."""
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ConfigError("YAML exceeds 1 MiB")
    try:
        depth = 0
        for event in yaml.parse(text):
            if (
                isinstance(event, yaml.events.AliasEvent)
                or getattr(event, "anchor", None)
                or getattr(event, "tag", None)
            ):
                raise ConfigError("YAML aliases, anchors and explicit tags are forbidden")
            if isinstance(event, yaml.events.MappingStartEvent | yaml.events.SequenceStartEvent):
                depth += 1
                if depth > 32:
                    raise ConfigError("YAML exceeds 32 nesting levels")
            elif isinstance(event, yaml.events.MappingEndEvent | yaml.events.SequenceEndEvent):
                depth -= 1
        data = yaml.load(text, Loader=StrictLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc
    mapping(data, {"schema_version", "backend", "network", "backends"}, {"schema_version", "backend", "backends"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ConfigError("schema_version must be integer 1")
    backend = literal(data["backend"])
    backends = mapping(data["backends"], BACKENDS)
    if backend not in BACKENDS or backend not in backends:
        raise ConfigError("Selected backend must have a known backend entry")
    for name, entry in backends.items():
        mapping(entry, {"profile"}, {"profile"})
        profile = mapping(entry["profile"], {"source", "preset", "parameters"}, {"source", "preset"})
        if profile["source"] != "generated":
            raise ConfigError("仅支持 generated 配置，不支持自备 XML。")
        if profile["preset"] != f"{name}/conservative-v1":
            raise ConfigError(f"Unsupported preset: {profile['preset']}")
        if type(profile.setdefault("parameters", {})) is not dict or profile["parameters"]:
            raise ConfigError("parameters 目前仅支持空映射。")
    network = mapping(data.get("network"), {"discovery", "interface", "address", "peers"}, {"discovery"})
    discovery = network["discovery"]
    if discovery == "local_only":
        mapping(network, {"discovery"})
    elif discovery in ("lan_multicast", "static_peers"):
        interface = literal(network.get("interface"))
        if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,15}", interface):
            raise ConfigError("Invalid interface name")
        if "address" in network:
            network["address"] = ipv4(network["address"])
        if discovery == "static_peers":
            peers = network.get("peers")
            if not isinstance(peers, list) or not peers:
                raise ConfigError("static_peers requires a nonempty IPv4 list")
            network["peers"] = sorted({ipv4(peer) for peer in peers})
        elif "peers" in network:
            raise ConfigError("peers is only allowed for static_peers")
    else:
        raise ConfigError("Unknown discovery policy")
    return data


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def publish_generation(root, identity, xml):
    """Store generated XML and identity; never update the user default here."""
    generation = digest(json_bytes(identity))
    files = {"profile.xml": xml}
    files["manifest.json"] = json_bytes(
        {"identity": identity, "files": {name: digest(data) for name, data in files.items()}}
    )
    parent = Path(root) / "generated"
    private_directory(parent)
    target = parent / generation

    def verify():
        verify_generation(target)
        if target.is_symlink() or not target.is_dir():
            raise ConfigError("Invalid generation directory")
        for name, content in files.items():
            path = target / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise ConfigError(f"Generation content mismatch: {name}")

    if target.exists() or target.is_symlink():
        verify()
        return target
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        for name, content in files.items():
            with (temporary / name).open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            verify()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def private_directory(path):
    """Create missing managed ancestors privately; never chmod existing parents.

    Reject symlinks in the write path. This is not a sandbox against another
    process with the same UID concurrently renaming parent directories.
    """
    path = Path(path).absolute()
    for component in (*reversed(path.parents), path):
        with suppress(FileExistsError):
            component.mkdir(mode=0o700)
        if component.is_symlink() or not component.is_dir():
            raise ConfigError(f"Managed path is not a directory or contains a symlink: {component}")
    return path


def snapshot(path):
    """Freeze bounded regular-file content, existence and symlink destination.

    An explicit readable symlink is valid input. A dangling link is an error,
    not an absent configuration. O_NONBLOCK prevents FIFO inputs hanging CLI.
    """
    path = Path(path).absolute()
    try:
        before = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "content": None, "resolved": None}
    try:
        resolved = str(path.resolve(strict=True))
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ConfigError(f"Input must be a regular file: {path}")
            content = stream.read(MAX_BYTES + 1)
            finished = os.fstat(stream.fileno())
        after = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(before, key) != getattr(after, key) for key in fields)
            or any(getattr(opened, key) != getattr(finished, key) for key in fields)
            or resolved != str(path.resolve(strict=True))
        ):
            raise ConfigError(f"Input changed while reading: {path}")
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"Cannot read input: {path}: {exc}") from exc
    if len(content) > MAX_BYTES:
        raise ConfigError(f"Input exceeds 1 MiB: {path}")
    return {"path": str(path), "content": content, "resolved": resolved}


def verify_generation(path):
    """Verify stored identity and exact file hashes, not DDS compatibility."""
    path = Path(path)
    expected = {"profile.xml", "manifest.json"}
    try:
        if path.is_symlink() or not path.is_dir() or {item.name for item in path.iterdir()} != expected:
            raise ConfigError("Generation directory is invalid")
        contents = {}
        for name in expected:
            item = path / name
            if item.is_symlink():
                raise ConfigError("Generation contains a symlink")
            contents[name] = snapshot(item)["content"]
        manifest = json.loads(contents["manifest.json"])
        identity = manifest["identity"]
        hashes = {name: digest(content) for name, content in contents.items() if name != "manifest.json"}
        if (
            manifest["files"] != hashes
            or identity["xml"] != hashes["profile.xml"]
            or digest(json_bytes(identity)) != path.name
        ):
            raise ConfigError("Generation manifest or content mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Preserve potentially live files. An explicit, non-overwriting move
        # lets the user keep the damaged evidence and prepare the same target.
        backup = path.parent.parent / f"{path.name}.damaged"
        move = shlex.join(["mv", "-T", "-n", "--", str(path), str(backup)])
        raise ConfigError(
            f"Generation verification failed: {path}: {exc}\n"
            "请先停止引用此配置的节点，将损坏目录移出后再重新 set：\n"
            f"  {move}\n"
            "若备份路径已存在，请另选备份名；仅重复 set 不会修复损坏文件。"
        ) from exc
    return manifest


def publish_configuration(source, target, content, generations, *, remove=False):
    """Commit already validated YAML after generation preparation.

    source/target are snapshots taken before user interaction or installation.
    All writers to the fixed target cooperate on its parent .publish.lock.
    No lock is held during generation/installation. External editors do not
    honor flock: we detect their edits up to the final pre-replace check, but
    cannot provide a filesystem compare-and-swap against arbitrary writers.
    """
    path = Path(target["path"])
    if remove and target["content"] is None:
        # Nothing to remove: do not create directories/locks in an unused HOME.
        if snapshot(path) != target:
            raise ConfigError(f"Configuration changed: {path}")
        return False
    parent = private_directory(path.parent)
    lock_path = parent / ".publish.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError as exc:
        raise ConfigError(f"Cannot open publication lock: {exc}") from exc
    with os.fdopen(descriptor, "rb") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ConfigError("Publication lock must be a regular file")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigError("Configuration is locked by another publisher; retry") from exc

        def recheck():
            for original in (source, target):
                if snapshot(original["path"]) != original:
                    raise ConfigError(f"Configuration changed: {original['path']}")

        recheck()
        if remove:
            if path.is_symlink():
                raise ConfigError("默认配置是符号链接；请先处理该链接，或用 unset --tmp 仅退出当前终端。")
            if target["content"] is None:
                return False
            # Same publication lock and optimistic snapshot check as set. Only
            # the fixed default file is removed; generations remain untouched.
            path.unlink()
            return True
        if path.is_symlink() and content != target["content"]:
            raise ConfigError("Refusing to overwrite a configuration symlink; manage its target explicitly")
        if not generations:
            raise ConfigError("At least one prepared generation is required")
        for generation in generations:
            verify_generation(generation)
        if not isinstance(content, bytes) or len(content) > MAX_BYTES:
            raise ConfigError("Configuration must be bounded UTF-8 bytes")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError("Configuration must be UTF-8") from exc
        if content == target["content"]:
            recheck()
            return False
        descriptor, name = tempfile.mkstemp(prefix=".config-", dir=parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            recheck()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return True
