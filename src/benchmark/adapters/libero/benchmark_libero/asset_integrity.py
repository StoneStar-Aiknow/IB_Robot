"""Pinned LIBERO asset validation shared by runtime and pre-build setup.

This module deliberately uses only the standard library and no package-relative
imports: setup loads this source file directly, without importing the adapter.
Every validation reads the payload again; a setup stamp is not an integrity check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

EXPECTED_ASSET_DIRS = {
    "hf-libero": (
        "articulated_objects",
        "scenes",
        "stable_hope_objects",
        "stable_scanned_objects",
        "textures",
        "turbosquid_objects",
    ),
    "libero-plus": (
        "articulated_objects",
        "new_objects",
        "scenes",
        "stable_hope_objects",
        "stable_scanned_objects",
        "textures",
        "turbosquid_objects",
    ),
}
# LIBERO-plus: Sylvest/LIBERO-plus assets.zip at b548dd25ee0401c46217ba7e3614a598e8979e48,
# archive SHA-256 96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf (6,395,849,578 bytes).
# The extracted tree was rehashed in sorted relative Path order and includes 215
# .msh meshes. It reproduced the historical 439,246-file SHA-256
# accd5d96e860345d75036e2771a9894d62b398c8a5139ac2884e33b4a599c047 when excluding
# .msh; including the runtime meshes gives the pin below. The archive digest and
# wrapper are kept here as retrieval provenance, while runtime validation remains
# based on the extracted tree and does not trust archive metadata alone.
# Keep the revision, wrapper, suffixes, file count and tree hash in sync.
EXPECTED_ASSET_FILE_COUNT = {"hf-libero": 585, "libero-plus": 439461}
EXPECTED_ASSET_TREE_SHA256 = {
    "hf-libero": "ac6d9c70ae4de9b8e4781d0b85958b0bf6f5e5378bdcf74846e79c3dba9b5b37",
    "libero-plus": "3b4f399974057ba7585509c7e034d703b36283df22422ccc69b54ac7d7d544e6",
}
LIBERO_PLUS_BUSINESS_SUFFIXES = (
    ".xml",
    ".stl",
    ".msh",
    ".obj",
    ".mtl",
    ".png",
    ".jpg",
    ".jpeg",
    ".usd",
    ".usda",
    ".usdc",
    ".dds",
    ".hdr",
    ".ttf",
)


class AssetIntegrityError(RuntimeError):
    """An asset tree is absent, incomplete, unreadable, or has changed."""


def is_pending_assets_tree(root: Path) -> bool:
    """Only an absent path or a genuinely empty directory is a placeholder.

    Symlinks are user-selected trees, not placeholders that setup may replace.
    In particular, a dangling link must not be mistaken for an absent path.
    """
    return not root.is_symlink() and (not root.exists() or (root.is_dir() and next(root.iterdir(), None) is None))


def validate_assets_tree(root: Path, provider: str) -> tuple[int, str]:
    """Validate required directories, business-file count, and pinned tree hash.

    The two providers retain their original hash encodings. hf-libero hashes
    ``path NUL hex-file-sha LF``; LIBERO-plus hashes ``path NUL raw-file-sha``.
    Hugging Face cache metadata is not part of the hf-libero payload, while
    LIBERO-plus selects the upstream mesh/texture/XML suffixes.
    """
    if provider not in EXPECTED_ASSET_DIRS:
        raise ValueError(f"Unknown LIBERO asset provider: {provider!r}")
    try:
        if not root.is_dir():
            raise AssetIntegrityError(f"{provider} assets directory does not exist: {root}")
        missing = [name for name in EXPECTED_ASSET_DIRS[provider] if not (root / name).is_dir()]
        if missing:
            raise AssetIntegrityError(f"{provider} assets are missing required directories: {missing}")
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in LIBERO_PLUS_BUSINESS_SUFFIXES
                if provider == "libero-plus"
                else ".cache" not in path.relative_to(root).parts and path.name != ".gitattributes"
            )
        )
        count = len(files)
        if count != EXPECTED_ASSET_FILE_COUNT[provider]:
            raise AssetIntegrityError(
                f"{provider} assets contain {count} business files; expected {EXPECTED_ASSET_FILE_COUNT[provider]}"
            )
        digest = hashlib.sha256()
        for path in files:
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            if provider == "hf-libero":
                digest.update(file_digest.hexdigest().encode() + b"\n")
            else:
                digest.update(file_digest.digest())
        tree_sha256 = digest.hexdigest()
        if tree_sha256 != EXPECTED_ASSET_TREE_SHA256[provider]:
            raise AssetIntegrityError(
                f"{provider} asset tree SHA-256 is {tree_sha256}; expected {EXPECTED_ASSET_TREE_SHA256[provider]}"
            )
        return count, tree_sha256
    except OSError as exc:
        raise AssetIntegrityError(f"Cannot read {provider} assets at {root}: {exc}") from exc
