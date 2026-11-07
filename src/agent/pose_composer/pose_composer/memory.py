"""Long-term memory of arm poses the robot has done, shared by every conversation.

Two tables, so one pose can be recalled by many phrasings:

* ``poses``   one pose: the model's description, the joint targets of the arms it
  moves (an arm it leaves alone keeps wherever it is when recalled), the
  fingerprint it was solved under, and its standing.
* ``aliases`` one normalized phrasing -> one pose, or banned for that pose.

A pose is a *candidate* after successful execution. Repeated successes only make
it eligible for a question; the Composer requires explicit user acceptance of
that question before confirming it. Candidates never bypass planning or the
composer's model call. A pose solved under another
fingerprint (prompt, model, joint travel, rest pose) is *stale* until it is
re-solved from its description, which needs no model call. Retired poses are
kept for ``TOMBSTONE_SEC`` so the audit can explain them, then dropped.

Two poses that move the same arms to the same place are the same pose: a new
phrasing of it becomes an alias, not a copy. "The same place" is judged on the
pose's signature, the body-frame positions of the moved arms' elbow, wrist and
hand, within ``SAME_POSE_M`` -- not on joint angles, which differ for one visible
pose (a straight arm's shoulder yaw is free) -- and on joints within
``SAME_POSE_RAD`` when a pose has no signature. Every change is
written to ``audit`` in the same transaction.

Pure Python and SQLite: no ROS. One process writes (the composer node holds
``<path>.lock``); the maintenance CLI writes only when it can take that lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import re
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

SAME_POSE_RAD = 0.1
SAME_POSE_M = 0.05
MAX_POSES = 500
CANDIDATE_TTL_SEC = 7 * 86400.0
TOMBSTONE_SEC = 30 * 86400.0
BAD_TO_RETIRE = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS poses (
  pose_id TEXT PRIMARY KEY,
  state TEXT NOT NULL,                -- live | stale | retired
  confirmed INTEGER NOT NULL,
  pinned INTEGER NOT NULL,
  description_json TEXT NOT NULL,
  arms_json TEXT NOT NULL,            -- the arms the pose moves
  targets_json TEXT NOT NULL,         -- joint -> rad, moved arms only
  signature_json TEXT NOT NULL DEFAULT '{}',  -- point -> [x, y, z] m, moved arms only
  fingerprint TEXT NOT NULL,
  user_message TEXT NOT NULL,
  source TEXT NOT NULL,               -- generated | seed | import
  successes INTEGER NOT NULL DEFAULT 0,
  hits INTEGER NOT NULL DEFAULT 0,
  good INTEGER NOT NULL DEFAULT 0,
  bad INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  created REAL NOT NULL,
  last_used REAL NOT NULL,
  retired_at REAL
);
CREATE TABLE IF NOT EXISTS aliases (
  phrase TEXT NOT NULL,
  pose_id TEXT NOT NULL,
  state TEXT NOT NULL,                -- active | banned | superseded
  origin TEXT NOT NULL,               -- utterance | step | seed | import
  created REAL NOT NULL,
  last_used REAL NOT NULL,
  PRIMARY KEY (phrase, pose_id)
);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  at REAL NOT NULL,
  pose_id TEXT,
  phrase TEXT,
  action TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# Courtesy and filler that do not change which pose is meant.
_FILLER = ("请你", "请", "麻烦", "帮我", "给我", "一下", "吧", "呀", "啊", "哦", "嘛", "呢", "好吗", "可以吗")
_SYNONYMS = (("两只手", "双手"), ("两手", "双手"), ("胳膊", "手"), ("手臂", "手"), ("双臂", "双手"))
_PUNCT = re.compile(r"[\s　，。！？、,.!?;；:：'\"“”‘’（）()\[\]【】~～…·-]+")
# Wording that makes the pose depend on the current one or on the conversation.
RELATIVE = re.compile(r"再|一点|一些|也这样|也一样|也照|同样|刚才|刚刚|换成|回到|继续|还原|恢复|另一只|反过来")


def normalize(text: str) -> str:
    """The form phrases are matched in: no punctuation, filler or spacing; common synonyms unified."""
    folded = _PUNCT.sub("", str(text).strip().lower())
    for word in _FILLER:
        folded = folded.replace(word, "")
    for old, new in _SYNONYMS:
        folded = folded.replace(old, new)
    return folded


def is_relative(text: str) -> bool:
    return bool(RELATIVE.search(str(text)))


@dataclass(frozen=True)
class PoseRecord:
    pose_id: str
    state: str  # candidate | active | pinned | stale | retired
    description: dict
    arms: tuple[str, ...]
    targets: dict[str, float]
    user_message: str
    fingerprint: str
    successes: int
    good: int
    bad: int
    version: int
    source: str


def _state(row) -> str:
    if row["state"] != "live":
        return row["state"]
    if row["pinned"]:
        return "pinned"
    return "active" if row["confirmed"] else "candidate"


def _record(row) -> PoseRecord:
    return PoseRecord(
        pose_id=row["pose_id"],
        state=_state(row),
        description=json.loads(row["description_json"]),
        arms=tuple(json.loads(row["arms_json"])),
        targets={k: float(v) for k, v in json.loads(row["targets_json"]).items()},
        user_message=row["user_message"],
        fingerprint=row["fingerprint"],
        successes=row["successes"],
        good=row["good"],
        bad=row["bad"],
        version=row["version"],
        source=row["source"],
    )


class MemoryLocked(RuntimeError):
    """Another process (the composer node) owns the memory for writing."""


class PoseMemory:
    def __init__(
        self,
        path: str | Path,
        fingerprint: str,
        *,
        max_poses: int = MAX_POSES,
        candidate_ttl_sec: float = CANDIDATE_TTL_SEC,
        clock: Callable[[], float] = time.time,
        writer: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self._max_poses = int(max_poses)
        self._candidate_ttl_sec = float(candidate_ttl_sec)
        self._clock = clock
        self._lock_file = None
        if writer:
            self._lock_file = open(f"{self.path}.lock", "w")  # noqa: SIM115 - held for the process lifetime
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._lock_file.close()
                raise MemoryLocked(f"{self.path} is in use by another writer") from exc
        self._db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        if self._lock_file is not None:
            # Fold the write-ahead log into the database file, so a copy of the file alone is complete.
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._db.close()
        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()

    # --- transactions and audit ---------------------------------------------

    @contextlib.contextmanager
    def _tx(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def _audit(self, db, action: str, pose_id: str | None = None, phrase: str | None = None, **detail) -> None:
        db.execute(
            "INSERT INTO audit (at, pose_id, phrase, action, detail_json) VALUES (?, ?, ?, ?, ?)",
            (self._clock(), pose_id, phrase, action, json.dumps(detail, ensure_ascii=False, sort_keys=True)),
        )

    # --- reads -------------------------------------------------------------

    def get(self, pose_id: str) -> PoseRecord | None:
        row = self._db.execute("SELECT * FROM poses WHERE pose_id = ?", (pose_id,)).fetchone()
        return _record(row) if row else None

    def lookup(self, text: str, *, confirmed_only: bool) -> PoseRecord | None:
        """The live pose these words recall under the current fingerprint, or None.

        Relative wording never recalls a pose: what it means depends on the
        pose the robot is in now.
        """
        phrase = normalize(text)
        if not phrase or is_relative(text):
            return None
        rows = self._db.execute(
            "SELECT p.* FROM aliases a JOIN poses p ON p.pose_id = a.pose_id "
            "WHERE a.phrase = ? AND a.state = 'active' AND p.state = 'live' AND p.fingerprint = ? "
            "ORDER BY p.pinned DESC, p.confirmed DESC, p.last_used DESC",
            (phrase, self.fingerprint),
        ).fetchall()
        for row in rows:
            if confirmed_only and not (row["confirmed"] or row["pinned"]):
                continue
            return _record(row)
        return None

    def planning_catalog(self) -> list[dict]:
        """Only currently recallable poses, excluding candidate, banned and stale entries."""
        entries = []
        for record in self.list():
            if record.state not in ("active", "pinned") or record.fingerprint != self.fingerprint:
                continue
            phrases = [
                a["phrase"]
                for a in self.aliases(record.pose_id)
                if a["state"] == "active"
                and not is_relative(a["phrase"])
                and (hit := self.lookup(a["phrase"], confirmed_only=True)) is not None
                and hit.pose_id == record.pose_id
            ]
            if phrases:
                entries.append(
                    {
                        "pose_id": record.pose_id,
                        "target_name": phrases[0],
                        "phrases": phrases,
                        "arms": list(record.arms),
                        "description": record.description,
                    }
                )
        return entries

    def aliases(self, pose_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT phrase, state, origin FROM aliases WHERE pose_id = ? ORDER BY created", (pose_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def list(self, *, include_retired: bool = False) -> list[PoseRecord]:
        sql = "SELECT * FROM poses" + ("" if include_retired else " WHERE state != 'retired'") + " ORDER BY created"
        return [_record(row) for row in self._db.execute(sql).fetchall()]

    def audit_log(self, limit: int = 200) -> list[dict]:
        rows = self._db.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def meta(self, key: str, default: str = "") -> str:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # --- writes ------------------------------------------------------------

    def _same_pose(
        self, db, arms: Iterable[str], targets: Mapping[str, float], signature: Mapping | None = None, exclude: str = ""
    ) -> str | None:
        arms = sorted(arms)
        for row in db.execute(
            "SELECT pose_id, arms_json, targets_json, signature_json FROM poses WHERE state = 'live' "
            "AND fingerprint = ? AND pose_id != ? ORDER BY pinned DESC, confirmed DESC, created",
            (self.fingerprint, exclude),
        ):
            if sorted(json.loads(row["arms_json"])) != arms:
                continue
            other_sig = json.loads(row["signature_json"] or "{}")
            if signature and other_sig and set(other_sig) == set(signature):
                gap = max(
                    sum((a - b) ** 2 for a, b in zip(other_sig[k], signature[k], strict=True)) ** 0.5 for k in signature
                )
                if gap < SAME_POSE_M:
                    return row["pose_id"]
                continue
            other = json.loads(row["targets_json"])
            if set(other) == set(targets) and max(abs(other[k] - targets[k]) for k in targets) < SAME_POSE_RAD:
                return row["pose_id"]
        return None

    def _attach(self, db, pose_id: str, phrases: Iterable[tuple[str, str]]) -> list[str]:
        """Point each phrase at pose_id unless that pairing is banned; a phrase recalls one pose."""
        now = self._clock()
        attached = []
        for text, origin in phrases:
            phrase = normalize(text)
            if not phrase or is_relative(text):
                continue
            current = db.execute(
                "SELECT state FROM aliases WHERE phrase = ? AND pose_id = ?", (phrase, pose_id)
            ).fetchone()
            if current and current["state"] == "banned":
                continue
            for other in db.execute(
                "SELECT pose_id FROM aliases WHERE phrase = ? AND state = 'active' AND pose_id != ?", (phrase, pose_id)
            ).fetchall():
                db.execute(
                    "UPDATE aliases SET state = 'superseded' WHERE phrase = ? AND pose_id = ?",
                    (phrase, other["pose_id"]),
                )
                self._audit(db, "alias_superseded", other["pose_id"], phrase, by=pose_id)
            if current:
                db.execute(
                    "UPDATE aliases SET state = 'active', last_used = ? WHERE phrase = ? AND pose_id = ?",
                    (now, phrase, pose_id),
                )
            else:
                db.execute(
                    "INSERT INTO aliases (phrase, pose_id, state, origin, created, last_used) VALUES (?, ?, 'active', ?, ?, ?)",
                    (phrase, pose_id, origin, now, now),
                )
                self._audit(db, "alias_added", pose_id, phrase, origin=origin)
            attached.append(phrase)
        return attached

    def record_success(
        self,
        phrases: Iterable[tuple[str, str]],
        *,
        description: Mapping,
        arms: Iterable[str],
        targets: Mapping[str, float],
        user_message: str,
        pose_id: str = "",
        signature: Mapping | None = None,
    ) -> str:
        """A pose was executed successfully: remember it (or count the repeat) and attach the phrases.

        ``pose_id`` names a remembered pose that was recalled; a repeat success
        increments its count without confirming a candidate.
        """
        now = self._clock()
        arms = sorted(arms)
        targets = {k: float(v) for k, v in targets.items()}
        signature = {k: [float(x) for x in v] for k, v in (signature or {}).items()}
        with self._tx() as db:
            existing = pose_id or self._same_pose(db, arms, targets, signature)
            if (
                existing
                and db.execute("SELECT 1 FROM poses WHERE pose_id = ? AND state = 'live'", (existing,)).fetchone()
            ):
                db.execute(
                    "UPDATE poses SET successes = successes + 1, last_used = ? WHERE pose_id = ?",
                    (now, existing),
                )
                self._audit(db, "success", existing)
                # A candidate's confirmation must save the last achieved pose,
                # rather than the first near-equivalent pose in the cluster.
                db.execute(
                    "UPDATE poses SET targets_json = ?, signature_json = ?, description_json = ? "
                    "WHERE pose_id = ? AND confirmed = 0 AND pinned = 0",
                    (
                        json.dumps(targets, sort_keys=True),
                        json.dumps(signature, sort_keys=True),
                        json.dumps(description, ensure_ascii=False, sort_keys=True),
                        existing,
                    ),
                )
                self._attach(db, existing, phrases)
                return existing
            new_id = uuid.uuid4().hex[:12]
            db.execute(
                "INSERT INTO poses (pose_id, state, confirmed, pinned, description_json, arms_json, targets_json, "
                "signature_json, fingerprint, user_message, source, successes, created, last_used) "
                "VALUES (?, 'live', 0, 0, ?, ?, ?, ?, ?, ?, 'generated', 1, ?, ?)",
                (new_id, json.dumps(description, ensure_ascii=False, sort_keys=True), json.dumps(arms),
                 json.dumps(targets, sort_keys=True), json.dumps(signature, sort_keys=True), self.fingerprint,
                 user_message, now, now),
            )  # fmt: skip
            self._audit(db, "added", new_id, state="candidate", arms=arms)
            self._attach(db, new_id, phrases)
            self._evict_locked(db)
            return new_id

    def note_hit(self, pose_id: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE poses SET hits = hits + 1, last_used = ? WHERE pose_id = ?", (self._clock(), pose_id))

    def feedback(self, pose_id: str, verdict: str, phrases: Iterable[str] = ()) -> str:
        """Explicit good confirms; next_task does not save; bad bans phrases and may retire."""
        with self._tx() as db:
            row = db.execute("SELECT * FROM poses WHERE pose_id = ?", (pose_id,)).fetchone()
            if row is None or row["state"] == "retired":
                return "pose no longer remembered"
            if verdict == "next_task":
                return "not saved"
            if verdict == "good":
                db.execute(
                    "UPDATE poses SET confirmed = 1, good = good + ? WHERE pose_id = ?",
                    (1 if verdict == "good" else 0, pose_id),
                )
                if not row["confirmed"]:
                    self._audit(db, "confirmed", pose_id, reason=verdict)
                return "confirmed"
            if verdict != "bad":
                raise ValueError(f"unknown verdict {verdict!r}")
            for phrase in {normalize(p) for p in phrases if normalize(p)}:
                db.execute(
                    "INSERT INTO aliases (phrase, pose_id, state, origin, created, last_used) "
                    "VALUES (?, ?, 'banned', 'feedback', ?, ?) "
                    "ON CONFLICT (phrase, pose_id) DO UPDATE SET state = 'banned'",
                    (phrase, pose_id, self._clock(), self._clock()),
                )
                self._audit(db, "alias_banned", pose_id, phrase)
            bad = row["bad"] + 1
            db.execute("UPDATE poses SET bad = ? WHERE pose_id = ?", (bad, pose_id))
            if bad >= BAD_TO_RETIRE and not row["pinned"]:
                self._retire_locked(db, pose_id, f"{bad} bad verdicts")
                return "retired"
            return "banned for these words"

    def _retire_locked(self, db, pose_id: str, reason: str) -> None:
        db.execute("UPDATE poses SET state = 'retired', retired_at = ? WHERE pose_id = ?", (self._clock(), pose_id))
        self._audit(db, "retired", pose_id, reason=reason)

    def _evict_locked(self, db) -> None:
        now = self._clock()
        for row in db.execute(
            "SELECT pose_id FROM poses WHERE state = 'live' AND confirmed = 0 AND pinned = 0 AND last_used < ?",
            (now - self._candidate_ttl_sec,),
        ).fetchall():
            self._retire_locked(db, row["pose_id"], "candidate unused")
        live = db.execute(
            "SELECT pose_id, successes, hits, good, bad, last_used FROM poses WHERE state = 'live' AND pinned = 0"
        ).fetchall()
        pinned = db.execute("SELECT COUNT(*) FROM poses WHERE state = 'live' AND pinned = 1").fetchone()[0]
        excess = len(live) + pinned - self._max_poses
        if excess > 0:

            def score(row):
                age_days = (now - row["last_used"]) / 86400.0
                return row["successes"] + row["hits"] + 3 * row["good"] - 5 * row["bad"] - age_days

            for row in sorted(live, key=score)[:excess]:
                self._retire_locked(db, row["pose_id"], "capacity")
        db.execute("DELETE FROM aliases WHERE pose_id IN (SELECT pose_id FROM poses WHERE state = 'retired' AND retired_at < ?)",
                   (now - TOMBSTONE_SEC,))  # fmt: skip
        db.execute("DELETE FROM poses WHERE state = 'retired' AND retired_at < ?", (now - TOMBSTONE_SEC,))

    def evict(self) -> None:
        with self._tx() as db:
            self._evict_locked(db)

    # --- fingerprint changes, seeds, import --------------------------------

    def mark_stale(self) -> int:
        """Every live pose solved under another fingerprint becomes stale."""
        with self._tx() as db:
            rows = db.execute(
                "SELECT pose_id FROM poses WHERE state = 'live' AND fingerprint != ?", (self.fingerprint,)
            ).fetchall()
            for row in rows:
                db.execute("UPDATE poses SET state = 'stale' WHERE pose_id = ?", (row["pose_id"],))
                self._audit(db, "stale", row["pose_id"], fingerprint=self.fingerprint)
            return len(rows)

    def revalidate(self, resolve: Callable[[dict], tuple | None]) -> dict[str, int]:
        """Re-solve every stale pose from its description; ``resolve`` returns (arms, targets[, signature]) or None.

        No model call: the description is what the model said, solving it again
        is deterministic. A re-solved pose that is now the same as another live
        one is merged into it.
        """
        counts = {"restored": 0, "retired": 0, "merged": 0}
        for row in self._db.execute("SELECT * FROM poses WHERE state = 'stale'").fetchall():
            solved = resolve(json.loads(row["description_json"]))
            with self._tx() as db:
                if solved is None:
                    self._retire_locked(db, row["pose_id"], "no longer solvable within the current travel")
                    counts["retired"] += 1
                    continue
                arms, targets = sorted(solved[0]), {k: float(v) for k, v in solved[1].items()}
                signature = {k: [float(x) for x in v] for k, v in (solved[2] if len(solved) > 2 else {}).items()}
                twin = self._same_pose(db, arms, targets, signature, exclude=row["pose_id"])
                if twin:
                    phrases = [
                        (a["phrase"], a["origin"])
                        for a in db.execute(
                            "SELECT phrase, origin FROM aliases WHERE pose_id = ? AND state = 'active'",
                            (row["pose_id"],),
                        ).fetchall()
                    ]
                    self._attach(db, twin, phrases)
                    db.execute(
                        "UPDATE poses SET pinned = MAX(pinned, ?), confirmed = MAX(confirmed, ?) WHERE pose_id = ?",
                        (row["pinned"], row["confirmed"], twin),
                    )
                    self._retire_locked(db, row["pose_id"], f"merged into {twin}")
                    counts["merged"] += 1
                    continue
                db.execute(
                    "UPDATE poses SET state = 'live', arms_json = ?, targets_json = ?, signature_json = ?, "
                    "fingerprint = ?, version = version + 1 WHERE pose_id = ?",
                    (
                        json.dumps(arms),
                        json.dumps(targets, sort_keys=True),
                        json.dumps(signature, sort_keys=True),
                        self.fingerprint,
                        row["pose_id"],
                    ),
                )
                self._audit(db, "revalidated", row["pose_id"], fingerprint=self.fingerprint)
                counts["restored"] += 1
        return counts

    def add_described(
        self, entries: Iterable[Mapping], *, source: str, pinned: bool, meta: Mapping[str, str] = ()
    ) -> int:
        """Seeds and imports: descriptions with phrases, stored stale and re-solved by ``revalidate``."""
        added = 0
        with self._tx() as db:
            for entry in entries:
                pose_id = f"{source}-{entry['id']}" if source == "seed" else uuid.uuid4().hex[:12]
                if db.execute("SELECT 1 FROM poses WHERE pose_id = ?", (pose_id,)).fetchone():
                    db.execute("UPDATE poses SET state = 'retired', retired_at = ? WHERE pose_id = ?", (0.0, pose_id))
                    db.execute("DELETE FROM aliases WHERE pose_id = ?", (pose_id,))
                    db.execute("DELETE FROM poses WHERE pose_id = ?", (pose_id,))
                now = self._clock()
                db.execute(
                    "INSERT INTO poses (pose_id, state, confirmed, pinned, description_json, arms_json, targets_json, "
                    "fingerprint, user_message, source, created, last_used) "
                    "VALUES (?, 'stale', 1, ?, ?, '[]', '{}', '', ?, ?, ?, ?)",
                    (pose_id, int(pinned), json.dumps(entry["description"], ensure_ascii=False, sort_keys=True),
                     str(entry.get("user_message", "")), source, now, now),
                )  # fmt: skip
                self._audit(db, f"{source}_added", pose_id, pinned=pinned)
                self._attach(db, pose_id, [(phrase, source) for phrase in entry.get("phrases", [])])
                added += 1
            for key, value in dict(meta).items():
                db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = ?",
                           (key, value, value))  # fmt: skip
        return added

    def export(self) -> list[dict]:
        """Confirmed and pinned poses as descriptions with their active phrases (re-solved on import)."""
        out = []
        for record in self.list():
            if record.state not in ("active", "pinned"):
                continue
            phrases = [a["phrase"] for a in self.aliases(record.pose_id) if a["state"] == "active"]
            out.append({"id": record.pose_id, "user_message": record.user_message, "pinned": record.state == "pinned",
                        "phrases": phrases, "description": record.description})  # fmt: skip
        return out

    # --- maintenance -------------------------------------------------------

    def set_pinned(self, pose_id: str, pinned: bool) -> None:
        with self._tx() as db:
            db.execute("UPDATE poses SET pinned = ?, confirmed = MAX(confirmed, ?) WHERE pose_id = ?",
                       (int(pinned), int(pinned), pose_id))  # fmt: skip
            self._audit(db, "pinned" if pinned else "unpinned", pose_id)

    def delete(self, pose_id: str, reason: str = "deleted by operator") -> None:
        with self._tx() as db:
            self._retire_locked(db, pose_id, reason)

    def ban(self, phrase: str, pose_id: str) -> None:
        with self._tx() as db:
            key = normalize(phrase)
            db.execute(
                "INSERT INTO aliases (phrase, pose_id, state, origin, created, last_used) VALUES (?, ?, 'banned', 'operator', ?, ?) "
                "ON CONFLICT (phrase, pose_id) DO UPDATE SET state = 'banned'",
                (key, pose_id, self._clock(), self._clock()),
            )
            self._audit(db, "alias_banned", pose_id, key, by="operator")
