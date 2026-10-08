"""Offline, owner-only reconciliation of quarantined executions."""

import argparse
import getpass
import os
from pathlib import Path

from skill_library.execution_store import SQLiteExecutionStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument(
        "--evidence", required=True, help="How stopped hardware and cleared controller queues were verified"
    )
    parser.add_argument("--confirm-hardware-stopped", action="store_true", required=True)
    parser.add_argument("--confirm-controller-queues-reset", action="store_true", required=True)
    args = parser.parse_args(argv)
    path = args.store.expanduser().resolve()
    if not path.is_file() or path.stat().st_uid != os.getuid():
        parser.error("store must exist and be owned by the current operating-system user")
    # Opening obtains the same exclusive lock as the Gateway.  A live runtime
    # cannot be bypassed by editing its ledger through this command.
    store = SQLiteExecutionStore(path)
    try:
        store.resolve_unknown(
            args.execution_id, operator_scope=f"os:{os.getuid()}:{getpass.getuser()}", evidence=args.evidence
        )
    finally:
        store.close()
    print("Reconciliation recorded. Original UNKNOWN result retained; no action was replayed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
