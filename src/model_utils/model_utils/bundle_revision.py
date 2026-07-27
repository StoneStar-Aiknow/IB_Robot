"""Publish a new structural bundle revision after replacing semantic assets."""

from __future__ import annotations

import argparse

from model_utils.inference_manifest_export import refresh_bundle_revision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root")
    args = parser.parse_args(argv)
    manifest = refresh_bundle_revision(args.bundle_root)
    print(
        f"Published bundle {manifest.bundle.uuid} revision {manifest.bundle.revision} ({manifest.bundle.digest.value})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
