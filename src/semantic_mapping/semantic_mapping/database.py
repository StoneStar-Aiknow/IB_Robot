"""Versioned SQLite persistence for semantic maps and observation provenance."""

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .association import SemanticObservation, SemanticTrack
from .runtime_identity import (
    REQUIRED_MAPPING_ROLES,
    MappingRunPin,
    SemanticIdentity,
    parse_semantic_identities,
    semantic_identities_dict,
)

SCHEMA_VERSION = 4
PROTOTYPE_TABLE = "semantic_objects"


class DatabaseCompatibilityError(RuntimeError):
    """Raised when a database cannot be fused with the active runtime."""


@dataclass(frozen=True)
class SemanticMapManifest:
    global_frame: str
    geometry_map_id: str
    geometry_map_hash: str
    localization_session_id: str
    calibration_id: str
    urdf_hash: str
    coordinate_convention: str
    semantic_identities: dict[str, SemanticIdentity | dict]
    schema_version: int = SCHEMA_VERSION
    settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_roles = () if self.settings.get("mapping_backend") == "embedded" else REQUIRED_MAPPING_ROLES
        object.__setattr__(
            self,
            "semantic_identities",
            parse_semantic_identities(self.semantic_identities, required_roles=required_roles),
        )

    def validate(self) -> None:
        required = {
            "global_frame": self.global_frame,
            "geometry_map_id": self.geometry_map_id,
            "geometry_map_hash": self.geometry_map_hash,
            "localization_session_id": self.localization_session_id,
            "calibration_id": self.calibration_id,
            "urdf_hash": self.urdf_hash,
            "coordinate_convention": self.coordinate_convention,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            details = []
            if missing:
                details.append(f"missing identity fields: {', '.join(missing)}")
            raise ValueError("; ".join(details))
        required_roles = () if self.settings.get("mapping_backend") == "embedded" else REQUIRED_MAPPING_ROLES
        parse_semantic_identities(self.semantic_identities, required_roles=required_roles)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"manifest schema version must be {SCHEMA_VERSION}")

    def compatibility_errors(self, other: "SemanticMapManifest") -> list[str]:
        fields = (
            "schema_version",
            "global_frame",
            "geometry_map_id",
            "geometry_map_hash",
            "localization_session_id",
            "calibration_id",
            "urdf_hash",
            "coordinate_convention",
            "semantic_identities",
        )
        return [name for name in fields if getattr(self, name) != getattr(other, name)]

    @property
    def canonical_semantic_identities(self) -> dict[str, SemanticIdentity]:
        required_roles = () if self.settings.get("mapping_backend") == "embedded" else REQUIRED_MAPPING_ROLES
        return parse_semantic_identities(self.semantic_identities, required_roles=required_roles)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["semantic_identities"] = semantic_identities_dict(self.canonical_semantic_identities)
        return value


@dataclass(frozen=True)
class MappingRunRecord:
    run_id: str
    configuration_generation: int
    expected_service_instance_ids: dict[str, str]
    required_semantic_identities: dict[str, SemanticIdentity | dict]
    status: str
    started_ns: int
    updated_ns: int
    ended_ns: int | None = None
    status_reason: str = ""

    def pin(self) -> MappingRunPin:
        return MappingRunPin(
            self.run_id,
            self.configuration_generation,
            self.expected_service_instance_ids,
            self.required_semantic_identities,
        )


@dataclass(frozen=True)
class CaptionRecord:
    object_id: str
    caption: str
    model_identity: str
    created_ns: int
    success: bool = True
    message: str = ""


@dataclass(frozen=True)
class ObjectGeometryRecord:
    object_id: str
    object_version: int
    artifact_type: str
    artifact_path: str
    artifact_hash: str
    point_count: int
    created_ns: int


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def inspect_database(path: str | Path) -> str:
    """Return ``missing``, ``prototype``, ``versioned``, or ``unknown``."""
    database_path = Path(path).expanduser()
    if not database_path.exists():
        return "missing"
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        tables = _table_names(connection)
    finally:
        connection.close()
    if "semantic_manifest" in tables:
        return "versioned"
    if PROTOTYPE_TABLE in tables:
        return "prototype"
    return "unknown"


class SemanticMapDatabase:
    def __init__(
        self,
        path: str,
        manifest: SemanticMapManifest | None = None,
        *,
        read_only: bool = False,
        diagnostic: bool = False,
    ):
        self.path = Path(path).expanduser()
        self.read_only = read_only
        self.diagnostic = diagnostic
        self._lock = threading.RLock()
        state = inspect_database(self.path)
        if state == "prototype":
            raise DatabaseCompatibilityError(
                "prototype semantic database has no identity manifest; use explicit migration with all identity fields "
                "or rebuild the semantic map"
            )
        if state == "unknown":
            raise DatabaseCompatibilityError("database schema is unknown; open a supported map or rebuild it")
        if state == "missing" and read_only:
            raise FileNotFoundError(f"semantic database does not exist: {self.path}")
        if state == "missing" and manifest is None:
            raise ValueError("a manifest is required when creating a semantic database")

        if read_only:
            self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        if state == "missing":
            manifest.validate()
            self._create_schema()
            self._write_manifest(manifest)

        self.manifest = self._read_manifest()
        self.manifest.validate()
        if manifest is not None:
            manifest.validate()
            errors = self.manifest.compatibility_errors(manifest)
            if errors and not (read_only and diagnostic):
                self.close()
                raise DatabaseCompatibilityError(
                    "semantic database identity mismatch: "
                    + ", ".join(errors)
                    + "; reopen read-only diagnostic mode or rebuild"
                )
            self.compatibility_errors = errors
        else:
            self.compatibility_errors = []

    def _create_schema(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE semantic_manifest (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    global_frame TEXT NOT NULL,
                    geometry_map_id TEXT NOT NULL,
                    geometry_map_hash TEXT NOT NULL,
                    localization_session_id TEXT NOT NULL,
                    calibration_id TEXT NOT NULL,
                    urdf_hash TEXT NOT NULL,
                    coordinate_convention TEXT NOT NULL,
                    semantic_identities_json TEXT NOT NULL,
                    settings_json TEXT NOT NULL
                );
                CREATE TABLE mapping_runs (
                    run_id TEXT PRIMARY KEY,
                    configuration_generation INTEGER NOT NULL,
                    expected_service_instance_ids_json TEXT NOT NULL,
                    required_semantic_identities_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_ns INTEGER NOT NULL,
                    updated_ns INTEGER NOT NULL,
                    ended_ns INTEGER,
                    status_reason TEXT NOT NULL
                );
                CREATE TABLE semantic_objects (
                    object_id TEXT PRIMARY KEY,
                    canonical_label TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    position_json TEXT NOT NULL,
                    size_json TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    first_seen_ns INTEGER NOT NULL,
                    last_seen_ns INTEGER NOT NULL,
                    observation_count INTEGER NOT NULL,
                    embedding BLOB,
                    embedding_size INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    map_version TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    object_version INTEGER NOT NULL,
                    model_versions_json TEXT NOT NULL,
                    semantic_identities_json TEXT NOT NULL,
                    deployment_provenance_json TEXT NOT NULL,
                    lifecycle_evidence_json TEXT NOT NULL,
                    attributes_json TEXT NOT NULL
                );
                CREATE TABLE semantic_observations (
                    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id TEXT NOT NULL REFERENCES semantic_objects(object_id) ON DELETE CASCADE,
                    object_version INTEGER NOT NULL,
                    source_stamp_ns INTEGER NOT NULL,
                    source_frame TEXT NOT NULL,
                    label TEXT NOT NULL,
                    canonical_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    position_json TEXT NOT NULL,
                    size_json TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    map_version TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    model_versions_json TEXT NOT NULL,
                    semantic_identities_json TEXT NOT NULL,
                    deployment_provenance_json TEXT NOT NULL,
                    mapping_run_id TEXT REFERENCES mapping_runs(run_id),
                    association_evidence_json TEXT NOT NULL,
                    attributes_json TEXT NOT NULL
                );
                CREATE INDEX observations_object_stamp
                    ON semantic_observations(object_id, source_stamp_ns);
                CREATE TABLE semantic_captions (
                    object_id TEXT PRIMARY KEY REFERENCES semantic_objects(object_id) ON DELETE CASCADE,
                    caption TEXT NOT NULL,
                    model_identity TEXT NOT NULL,
                    created_ns INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    message TEXT NOT NULL
                );
                CREATE TABLE object_geometry (
                    object_id TEXT NOT NULL REFERENCES semantic_objects(object_id) ON DELETE CASCADE,
                    object_version INTEGER NOT NULL,
                    artifact_type TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_hash TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    created_ns INTEGER NOT NULL,
                    PRIMARY KEY (object_id, object_version, artifact_type)
                );
                """
            )
            self.connection.commit()

    def _write_manifest(self, manifest: SemanticMapManifest) -> None:
        self.connection.execute(
            "INSERT INTO semantic_manifest VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                manifest.schema_version,
                manifest.global_frame,
                manifest.geometry_map_id,
                manifest.geometry_map_hash,
                manifest.localization_session_id,
                manifest.calibration_id,
                manifest.urdf_hash,
                manifest.coordinate_convention,
                _json(semantic_identities_dict(manifest.canonical_semantic_identities)),
                _json(manifest.settings),
            ),
        )
        self.connection.commit()

    def _read_manifest(self) -> SemanticMapManifest:
        try:
            row = self.connection.execute("SELECT * FROM semantic_manifest WHERE singleton = 1").fetchone()
        except sqlite3.OperationalError as exc:
            raise DatabaseCompatibilityError("semantic database manifest table is invalid") from exc
        if row is None:
            raise DatabaseCompatibilityError("semantic database has no identity manifest; migrate or rebuild it")
        try:
            return SemanticMapManifest(
                schema_version=row["schema_version"],
                global_frame=row["global_frame"],
                geometry_map_id=row["geometry_map_id"],
                geometry_map_hash=row["geometry_map_hash"],
                localization_session_id=row["localization_session_id"],
                calibration_id=row["calibration_id"],
                urdf_hash=row["urdf_hash"],
                coordinate_convention=row["coordinate_convention"],
                semantic_identities=json.loads(row["semantic_identities_json"]),
                settings=json.loads(row["settings_json"]),
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise DatabaseCompatibilityError(
                f"semantic database schema is incompatible with version {SCHEMA_VERSION}; migrate or rebuild it"
            ) from exc

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("semantic database is open read-only")

    def load(self) -> list[SemanticTrack]:
        with self._lock:
            rows = self.connection.execute("SELECT * FROM semantic_objects ORDER BY object_id").fetchall()
        tracks = []
        for row in rows:
            embedding = None
            if row["embedding"] is not None and row["embedding_size"] > 0:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_size"]).copy()
            tracks.append(
                SemanticTrack(
                    object_id=row["object_id"],
                    canonical_label=row["canonical_label"],
                    label=row["label"],
                    confidence=row["confidence"],
                    position=np.asarray(json.loads(row["position_json"]), dtype=np.float64),
                    size=np.asarray(json.loads(row["size_json"]), dtype=np.float64),
                    point_count=row["point_count"],
                    first_seen_ns=row["first_seen_ns"],
                    last_seen_ns=row["last_seen_ns"],
                    observation_count=row["observation_count"],
                    embedding=embedding,
                    state=row["state"],
                    map_version=row["map_version"],
                    session_id=row["session_id"],
                    object_version=row["object_version"],
                    model_versions=json.loads(row["model_versions_json"]),
                    semantic_identities=json.loads(row["semantic_identities_json"]),
                    deployment_provenance=json.loads(row["deployment_provenance_json"]),
                    lifecycle_evidence=json.loads(row["lifecycle_evidence_json"]),
                    attributes=json.loads(row["attributes_json"]),
                )
            )
        return tracks

    def upsert(self, track: SemanticTrack, observation: SemanticObservation | None = None) -> None:
        self._require_writable()
        embedding = None if track.embedding is None else np.asarray(track.embedding, dtype=np.float32)
        values = (
            track.object_id,
            track.canonical_label or track.label.casefold(),
            track.label,
            track.confidence,
            _json(track.position.tolist()),
            _json(track.size.tolist()),
            track.point_count,
            track.first_seen_ns,
            track.last_seen_ns,
            track.observation_count,
            None if embedding is None else embedding.tobytes(),
            0 if embedding is None else embedding.size,
            track.state,
            track.map_version,
            track.session_id,
            track.object_version,
            _json(track.model_versions),
            _json(track.semantic_identities),
            _json(track.deployment_provenance),
            _json(track.lifecycle_evidence),
            _json(track.attributes),
        )
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO semantic_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    canonical_label=excluded.canonical_label,
                    label=excluded.label,
                    confidence=excluded.confidence,
                    position_json=excluded.position_json,
                    size_json=excluded.size_json,
                    point_count=excluded.point_count,
                    first_seen_ns=excluded.first_seen_ns,
                    last_seen_ns=excluded.last_seen_ns,
                    observation_count=excluded.observation_count,
                    embedding=excluded.embedding,
                    embedding_size=excluded.embedding_size,
                    state=excluded.state,
                    map_version=excluded.map_version,
                    session_id=excluded.session_id,
                    object_version=excluded.object_version,
                    model_versions_json=excluded.model_versions_json,
                    semantic_identities_json=excluded.semantic_identities_json,
                    deployment_provenance_json=excluded.deployment_provenance_json,
                    lifecycle_evidence_json=excluded.lifecycle_evidence_json,
                    attributes_json=excluded.attributes_json
                """,
                values,
            )
            if observation is not None:
                self._insert_observation(track, observation)
            self.connection.commit()

    def _insert_observation(self, track: SemanticTrack, observation: SemanticObservation) -> None:
        self.connection.execute(
            """
            INSERT INTO semantic_observations (
                object_id, object_version, source_stamp_ns, source_frame, label, canonical_label,
                confidence, position_json, size_json, point_count, map_version, session_id,
                model_versions_json, semantic_identities_json, deployment_provenance_json,
                mapping_run_id, association_evidence_json, attributes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track.object_id,
                track.object_version,
                observation.stamp_ns,
                observation.source_frame or str(observation.attributes.get("source_frame", "")),
                observation.label,
                observation.canonical_label or observation.label.casefold(),
                observation.confidence,
                _json(np.asarray(observation.position).tolist()),
                _json(np.asarray(observation.size).tolist()),
                observation.point_count,
                observation.map_version,
                observation.session_id,
                _json(observation.model_versions),
                _json(observation.semantic_identities),
                _json(observation.deployment_provenance),
                observation.mapping_run_id or None,
                _json(track.lifecycle_evidence),
                _json(observation.attributes),
            ),
        )

    def observation_count(self, object_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM semantic_observations WHERE object_id = ?", (object_id,)
        ).fetchone()
        return int(row["count"])

    def create_mapping_run(self, record: MappingRunRecord) -> MappingRunPin:
        self._require_writable()
        pin = record.pin()
        if record.status not in {"starting", "active", "paused", "completed", "failed"}:
            raise ValueError(f"invalid mapping run status: {record.status}")
        if record.started_ns < 0 or record.updated_ns < record.started_ns:
            raise ValueError("mapping run timestamps are invalid")
        if record.ended_ns is not None and record.ended_ns < record.started_ns:
            raise ValueError("mapping run end timestamp is invalid")
        with self._lock:
            self.connection.execute(
                "INSERT INTO mapping_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.run_id,
                    record.configuration_generation,
                    _json(record.expected_service_instance_ids),
                    _json(semantic_identities_dict(pin.required_semantic_identities)),
                    record.status,
                    record.started_ns,
                    record.updated_ns,
                    record.ended_ns,
                    record.status_reason,
                ),
            )
            self.connection.commit()
        return pin

    def get_mapping_run(self, run_id: str) -> MappingRunRecord | None:
        row = self.connection.execute("SELECT * FROM mapping_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return MappingRunRecord(
            run_id=row["run_id"],
            configuration_generation=row["configuration_generation"],
            expected_service_instance_ids=json.loads(row["expected_service_instance_ids_json"]),
            required_semantic_identities=json.loads(row["required_semantic_identities_json"]),
            status=row["status"],
            started_ns=row["started_ns"],
            updated_ns=row["updated_ns"],
            ended_ns=row["ended_ns"],
            status_reason=row["status_reason"],
        )

    def update_mapping_run_status(
        self,
        run_id: str,
        status: str,
        updated_ns: int,
        *,
        ended_ns: int | None = None,
        reason: str = "",
    ) -> None:
        self._require_writable()
        if status not in {"active", "paused", "completed", "failed"}:
            raise ValueError(f"invalid mapping run status: {status}")
        with self._lock:
            cursor = self.connection.execute(
                "UPDATE mapping_runs SET status = ?, updated_ns = ?, ended_ns = ?, status_reason = ? WHERE run_id = ?",
                (status, updated_ns, ended_ns, reason, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"mapping run not found: {run_id}")
            self.connection.commit()

    def upsert_caption(self, record: CaptionRecord) -> None:
        self._require_writable()
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO semantic_captions VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    caption=excluded.caption, model_identity=excluded.model_identity,
                    created_ns=excluded.created_ns, success=excluded.success, message=excluded.message
                """,
                (
                    record.object_id,
                    record.caption,
                    record.model_identity,
                    record.created_ns,
                    int(record.success),
                    record.message,
                ),
            )
            self.connection.commit()

    def get_caption(self, object_id: str) -> CaptionRecord | None:
        row = self.connection.execute("SELECT * FROM semantic_captions WHERE object_id = ?", (object_id,)).fetchone()
        if row is None:
            return None
        return CaptionRecord(
            object_id=row["object_id"],
            caption=row["caption"],
            model_identity=row["model_identity"],
            created_ns=row["created_ns"],
            success=bool(row["success"]),
            message=row["message"],
        )

    def upsert_geometry(self, record: ObjectGeometryRecord) -> None:
        self._require_writable()
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO object_geometry VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id, object_version, artifact_type) DO UPDATE SET
                    artifact_path=excluded.artifact_path, artifact_hash=excluded.artifact_hash,
                    point_count=excluded.point_count, created_ns=excluded.created_ns
                """,
                tuple(asdict(record).values()),
            )
            self.connection.commit()

    def close(self) -> None:
        with self._lock:
            self.connection.close()
