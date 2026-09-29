# AtomGit SDK attribution

Parts of the AtomGit CLI migration retain local behavior adapted from
`atomgit-sdk` 0.1.0, originally Copyright 2026 XiaoqiangWu, licensed under
Apache-2.0. The accompanying `LICENSE` is retained from that distribution.

Upstream project: https://github.com/wuxiaoqiang12/AtomGit_SDK

Affected adaptations:

- `.agents/skills/_shared/review_helpers.py`: local line numbering and diff
  interpretation utilities, modified for the official CLI's new-side line
  coordinates and strict content/completeness checks.
- `.agents/skills/atomgit-review-resolution/scripts/review_resolution.py`:
  author/bot classification, review-thread grouping and comment serialization,
  adapted to CLI-only access and the version-2 recovery ledger.
- The code and architecture finding dataclasses retain the former model fields
  with project-local validation; no SDK model is imported at runtime.

These are modified project adaptations, not an unmodified SDK distribution.
The SDK is no longer installed by project requirements. The official AtomGit
CLI is a separate developer-installed tool; no CLI binary or upstream CLI
source is redistributed in this change.
