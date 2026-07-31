"""Explicit prototype semantic-database migration tool."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from .association import FROZEN, SemanticTrack
from .database import SemanticMapDatabase, SemanticMapManifest, inspect_database


def manifest_from_json(path: str | Path) -> SemanticMapManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = SemanticMapManifest(**payload)
    manifest.validate()
    return manifest


def migrate_prototype_database(
    source: str | Path,
    destination: str | Path,
    manifest: SemanticMapManifest,
    *,
    confirmed: bool,
) -> int:
    """Copy prototype objects into a new versioned map as frozen records."""
    if not confirmed:
        raise PermissionError("prototype migration requires explicit operator confirmation")
    manifest.validate()
    source = Path(source).expanduser()
    destination = Path(destination).expanduser()
    if inspect_database(source) != "prototype":
        raise ValueError("source is not a supported prototype semantic database")
    if destination.exists():
        raise FileExistsError(f"migration destination already exists: {destination}")

    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM semantic_objects ORDER BY object_id").fetchall()
    finally:
        connection.close()

    database = SemanticMapDatabase(str(destination), manifest)
    try:
        for row in rows:
            embedding = None
            if row["embedding"] is not None and row["embedding_size"] > 0:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_size"]).copy()
            attributes = json.loads(row["attributes_json"])
            attributes["migration_source"] = str(source)
            attributes["prototype_active"] = bool(row["active"])
            database.upsert(
                SemanticTrack(
                    object_id=row["object_id"],
                    canonical_label=row["label"].casefold(),
                    label=row["label"],
                    confidence=row["confidence"],
                    position=np.asarray(json.loads(row["position_json"]), dtype=np.float64),
                    size=np.asarray(json.loads(row["size_json"]), dtype=np.float64),
                    point_count=row["point_count"],
                    first_seen_ns=row["first_seen_ns"],
                    last_seen_ns=row["last_seen_ns"],
                    observation_count=row["observation_count"],
                    embedding=embedding,
                    state=FROZEN,
                    map_version=manifest.geometry_map_hash,
                    session_id=manifest.localization_session_id,
                    semantic_identities={
                        role: identity.to_dict() for role, identity in manifest.canonical_semantic_identities.items()
                    },
                    attributes=attributes,
                )
            )
    except Exception:
        database.close()
        destination.unlink(missing_ok=True)
        raise
    database.close()
    return len(rows)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Copy an IB-Robot prototype semantic database into schema v3. "
        "The source is never modified; rebuild instead if identity metadata is unknown."
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--manifest", required=True, help="JSON file containing all SemanticMapManifest fields")
    parser.add_argument("--confirm", action="store_true", help="Confirm supplied identity metadata is authoritative")
    parsed = parser.parse_args(args)
    count = migrate_prototype_database(
        parsed.source,
        parsed.destination,
        manifest_from_json(parsed.manifest),
        confirmed=parsed.confirm,
    )
    print(f"Migrated {count} objects to {parsed.destination}")
