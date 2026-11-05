"""Owner-only offline reconciliation against a definite, identity-bound Gateway terminal."""

import argparse
import getpass
import os
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

from ibrobot_agent.contracts import RequestKey
from ibrobot_agent.deployment_lock import DeploymentLock
from ibrobot_agent.request_store import SQLiteRequestStore


def reconcile(requests, executions, agent_lock, key, *, caller_scope, evidence):
    requests, executions, agent_lock = (Path(p).expanduser().resolve() for p in (requests, executions, agent_lock))
    for path in (requests, executions, agent_lock):
        if not path.is_file() or path.stat().st_uid != os.getuid():
            raise ValueError("existing ledgers and Agent lock must belong to the current OS user")
    with ExitStack() as stack:
        # Do not open SQLiteExecutionStore: its constructor performs restart recovery.
        for lock_path in (agent_lock, executions.with_suffix(executions.suffix + ".lock")):
            lock = DeploymentLock(lock_path)
            stack.callback(lock.close)
        gateway = stack.enter_context(closing(sqlite3.connect(executions.as_uri() + "?mode=ro", uri=True)))
        gateway.row_factory = sqlite3.Row
        gateway.execute("BEGIN")
        if gateway.execute(
            "SELECT 1 FROM executions WHERE robot_id=? AND state NOT IN ('SUCCEEDED','FAILED','CANCELED') LIMIT 1",
            (key.robot_scope,),
        ).fetchone():
            raise ValueError("Gateway has a nonterminal or UNKNOWN execution for this robot")
        rows = gateway.execute(
            "SELECT * FROM executions WHERE request_id=? AND robot_id=? AND caller_scope=?",
            (key.request_id, key.robot_scope, caller_scope),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("expected one matching Gateway execution")
        store = SQLiteRequestStore(requests)
        stack.callback(store.close)
        store.reconcile_unknown(
            key,
            caller_scope=caller_scope,
            terminal=dict(rows[0]),
            operator_scope=f"os:{os.getuid()}:{getpass.getuser()}",
            evidence=evidence,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "requests",
        "executions",
        "agent-lock",
        "robot-scope",
        "channel-id",
        "principal-id",
        "request-id",
        "caller-scope",
        "evidence",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    key = RequestKey(args.robot_scope, args.channel_id, args.principal_id, args.request_id)
    reconcile(
        args.requests, args.executions, args.agent_lock, key, caller_scope=args.caller_scope, evidence=args.evidence
    )
    print("Gateway terminal reconciled. Original UNKNOWN retained; no action replayed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
