"""Read-only Linux runtime, interface and route facts; no ROS participants."""

import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET


class FactsError(ValueError):
    """Dependency, version or interface facts cannot be established."""


def resolve_binding(network, interfaces):
    local = network["discovery"] == "local_only"
    candidates = []
    for interface in interfaces:
        flags = set(interface.get("flags", []))
        if "UP" not in flags:
            continue
        if local:
            if "LOOPBACK" not in flags:
                continue
        elif (
            interface.get("ifname") != network["interface"]
            or "LOWER_UP" not in flags
            or (network["discovery"] == "lan_multicast" and "MULTICAST" not in flags)
            or "LOOPBACK" in flags
        ):
            continue
        for entry in interface.get("addr_info", []):
            if entry.get("family") != "inet":
                continue
            try:
                address = ipaddress.IPv4Interface(f"{entry['local']}/{entry['prefixlen']}")
            except (ValueError, KeyError) as exc:
                raise FactsError("Invalid interface address facts") from exc
            if local:
                if str(address.ip) != "127.0.0.1":
                    continue
            elif (
                entry.get("scope") != "global"
                or address.ip.is_loopback
                or address.ip.is_multicast
                or address.ip.is_unspecified
                or address.ip.is_reserved
                or (
                    address.network.prefixlen < 31
                    and address.ip in (address.network.broadcast_address, address.network.network_address)
                )
            ):
                continue
            candidates.append({"interface": interface["ifname"], "address": str(address.ip)})
    if network.get("address"):
        candidates = [entry for entry in candidates if entry["address"] == network["address"]]
    if len(candidates) != 1:
        raise FactsError("Select one IPv4 on an UP interface; missing, ambiguous or mismatched address")
    return candidates[0]


def file_fingerprint(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        checksum = (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _hash_stream(stream)
        )
    return {"path": str(path), "sha256": checksum}


def _hash_stream(stream):
    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(block)
    return checksum.hexdigest()


def runtime_facts(backend):
    """Inspect selected package prefixes and loader files, not actual dlopen.

    Only the current system Humble combinations are admitted. Unknown overlays
    are diagnostic failures. Native acceptance remains an explicit probe step.
    """
    dds, version, rmw, rmw_version, library = (
        ("fastrtps", "2.6.11", "rmw_fastrtps_cpp", "6.2.10", "libfastrtps.so")
        if backend == "fastdds"
        else ("cyclonedds", "0.10.5", "rmw_cyclonedds_cpp", "1.3.4", "libddsc.so")
    )
    prefixes = [Path(p) for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    if not prefixes or any(not p.is_absolute() for p in prefixes):
        raise FactsError("Load the project .shrc_local before inspecting DDS packages")
    packages = {}
    system = Path("/opt/ros/humble")
    for name, expected in ((dds, version), (rmw, rmw_version)):
        package_name = "ros-humble-" + name.replace("_", "-")
        if shutil.which("dpkg-query"):
            command = ["dpkg-query", "-W", "-f=${db:Status-Status}|${Version}", package_name]
            output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5).stdout
            status, _, installed_version = output.partition("|")
            if status != "installed":
                raise FactsError(f"Package is not installed: {package_name}")
        elif shutil.which("rpm"):
            command = ["rpm", "-q", "--qf", "%{VERSION}-%{RELEASE}", package_name]
            installed_version = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5).stdout
        else:
            raise FactsError("Unsupported system package database")
        if not installed_version.startswith(expected + "-"):
            raise FactsError(f"Unsupported {package_name} version {installed_version}; expected {expected}")
        packages[name] = {"version": expected, "package_version": installed_version}
        # DDS vendor libraries are plain CMake packages on these systems and
        # need not ship package.xml or ament markers. RMW packages do.
        if name == rmw:
            selected = next(
                (p for p in prefixes if (p / "share/ament_index/resource_index/packages" / name).exists()), None
            )
            if selected is None or selected.resolve() != system.resolve():
                raise FactsError(f"Missing or unsupported RMW overlay: {name}")
            metadata = selected / "share" / name / "package.xml"
            if ET.parse(metadata).getroot().findtext("version") != expected:
                raise FactsError(f"RMW metadata and installed package version disagree: {name}")
            packages[name]["metadata"] = file_fingerprint(metadata)
    loader_dirs = [Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    # Re-sourcing the project can prepend identical directories. Repeated
    # occurrences do not alter loader resolution; retain first-occurrence order.
    loader_dirs = list(dict.fromkeys(loader_dirs))
    prefixes = list(dict.fromkeys(prefixes))
    if any(not p.is_absolute() for p in loader_dirs):
        raise FactsError("Relative loader search paths are unsupported")
    search = loader_dirs + [system / "lib", system / "lib64"] + sorted((system / "lib").glob("*-linux-gnu"))
    libraries = {}
    for name in (library, f"lib{rmw}.so"):
        selected = next((directory / name for directory in search if (directory / name).is_file()), None)
        if selected is None or not selected.resolve().is_relative_to(system.resolve()):
            raise FactsError(f"Missing or shadowed runtime library: {name}")
        libraries[name] = file_fingerprint(selected)
    return {
        "backend_version": version,
        "packages": packages,
        "libraries": libraries,
        "loader_search": [str(p) for p in loader_dirs],
        "ament_prefixes": [str(p) for p in prefixes],
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "inspection": "metadata-only",
    }


def collect_facts(backend, network):
    try:
        runtime = runtime_facts(backend)
        if runtime["ld_preload"]:
            raise FactsError("LD_PRELOAD requires separate runtime validation")
        result = subprocess.run(
            ["ip", "-j", "-4", "addr", "show"], capture_output=True, text=True, check=True, timeout=5
        )
        interfaces = json.loads(result.stdout)
        binding = resolve_binding(network, interfaces)
        if network["discovery"] == "static_peers":
            routes = {}
            for peer in network["peers"]:
                query = subprocess.run(
                    ["ip", "-j", "-4", "route", "get", peer, "from", binding["address"]],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                routes[peer] = json.loads(query.stdout)
            validate_peer_routes(network, binding, interfaces, routes)
            # Stable route facts matter for reuse, unlike ephemeral cache fields.
            runtime["peer_routes"] = {
                peer: {key: route[0].get(key) for key in ("dev", "gateway", "prefsrc", "src")}
                for peer, route in routes.items()
            }
        return runtime, binding
    except (OSError, subprocess.SubprocessError, ET.ParseError, ValueError) as exc:
        raise FactsError(str(exc)) from exc


def validate_peer_routes(network, binding, interfaces, routes):
    local = set()
    invalid = set()
    for interface in interfaces:
        for entry in interface.get("addr_info", []):
            if entry.get("family") == "inet":
                address = ipaddress.IPv4Interface(f"{entry['local']}/{entry['prefixlen']}")
                local.add(address.ip)
                if address.network.prefixlen < 31:
                    invalid.update((address.network.broadcast_address, address.network.network_address))
    for peer in network["peers"]:
        address = ipaddress.IPv4Address(peer)
        if (
            address in local
            or address in invalid
            or address.is_loopback
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise FactsError(f"Peer must be a remote unicast IPv4, not local/broadcast: {peer}")
        route = routes.get(peer, [])
        if (
            len(route) != 1
            or route[0].get("dev") != binding["interface"]
            or route[0].get("type", "unicast") != "unicast"
        ):
            raise FactsError(f"Peer route does not use the selected interface: {peer}")
        source = route[0].get("prefsrc", route[0].get("src", binding["address"]))
        if source != binding["address"]:
            raise FactsError(f"Peer route source differs from selected address: {peer}")
