# skill_catalog

`skill_catalog` owns the immutable, profile-selected Robot Skill Package catalog. Runtime state, leases, task IDs,
authorization, and action execution do not belong in this package.

## Layout

- `config/skills/<name>/manifest.yaml`: semantic description, public capability schema, implementation index.
- `config/skills/<name>/implementations/<profile>.yaml`: robot-specific primitive or delegated execution body.
- `config/skills/<name>/SKILL.md`: human/Agent documentation only; never executable configuration.
- `config/profiles/<profile>.yaml`: enabled package, selected implementation, and planner visibility.

The current source-workspace profiles are `so101_single_arm`, `so101_handeye_realsense_grasp`, and
`so101_rtp_distributed`. Their migration tests require exact execution-body, public-capability, enabled-set, and
planner-visible parity with the compatibility inline definitions in `robot_config`.

## Source Modes

- `development`: validates a mutable staging tree before and after compilation. Source-workspace robot configs use this
  mode with an absolute path resolved by `embodied_bringup`.
- `installed`: resolves `skill_catalog` through the ament package index. It rejects symlinked implementation files, so
  it is intended for non-`--symlink-install` release builds.
- `production`: requires a release directory selected through `current` and verified by release manifest/digest.
- `legacy_inline`: one-release compatibility mode only.

Every successful compile produces immutable registry, capability, and provenance digests. Consumers must query an exact
`(registry_epoch, generation, registry_digest)` through Gateway services; they must not read these files directly at
validation or execution time.
