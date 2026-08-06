# skill_catalog

`skill_catalog` is the SSOT for skill manifests, implementation bodies, and
profile definitions. Robot YAML in `robot_config` selects the active profile;
this package owns the resulting immutable Robot Skill Package catalog. Runtime
state, leases, task IDs, authorization, and action execution do not belong here.

## Layout

- `config/skills/<name>/manifest.yaml`: semantic description, public capability schema, implementation index.
- `config/skills/<name>/implementations/<implementation>.yaml`: robot-specific primitive or delegated execution body.
- `config/skills/<name>/SKILL.md`: human/Agent documentation only; never executable configuration.
- `config/profiles/<profile>.yaml`: enabled package, selected implementation, and planner visibility.

The current source-workspace profiles are `so101_single_arm`,
`so101_handeye_realsense_grasp`, and `so101_rtp_distributed`. The shared stable
implementation `so101_arm_v1` is selected by every enabled entry in both
`so101_single_arm` and `so101_rtp_distributed`; the two profiles compile the
same execution body, so their implementation content cannot drift through
duplicated files. Registry/capability identity can still differ through the
canonical robot execution context.

## Source Modes

Robot YAML selects the source through `embodied.skill_catalog_source_mode`
(`installed`, `development`, or `production`), `embodied.skill_catalog_source_root`,
and `embodied.skill_catalog_profile`. Inline `embodied.skill_templates` is no
longer a valid execution SSOT; `robot_config.loader` rejects it with a single
error pointing at `embodied.skill_catalog_profile`.

- `development`: validates a mutable staging tree before and after compilation. Source-workspace robot configs use this
  mode with an absolute path resolved by `embodied_bringup`.
- `installed`: resolves `skill_catalog` through the ament package index. It rejects symlinked implementation files, so
  it is intended for non-`--symlink-install` release builds.
- `production`: requires a release directory selected through `current` and verified by release manifest/digest.

## Compilation integrity

Every non-hidden package directory under `config/skills/` must contain a
`manifest.yaml`. A malformed package directory (missing manifest, hidden
manifest, symlinked or escaped implementation path, forbidden release entry)
fails compilation with a `SKILL_SCHEMA_INVALID` / `SKILL_REFERENCE_MISSING` /
`SKILL_RELEASE_NOT_IMMUTABLE` diagnostic, even when the package is not enabled
in the active profile. Manifest schema validation also runs for packages
outside the enabled profile set, so catalog-wide drift is caught at compile
time rather than at execution time.

## Execution-context digest

`robot_config.loader.robot_config_digest` is the canonical execution-context
digest passed to `SkillRobotContext.robot_config_digest`. It deliberately
excludes catalog source selection (`skill_catalog_source_mode` /
`skill_catalog_source_root`), the active profile (`skill_catalog_profile`),
the resolved config path, and unrelated robot configuration. Flipping only
those fields keeps the same execution identity, so a compiled snapshot can be
reused across source/profile/path metadata changes; changing named poses,
named targets, joint limits, workspace, control mode, timeout policy, or
execution endpoints produces a different digest and forces a new registry
generation.

## Consumer identity and generation sync

Every successful compile produces immutable registry, capability, and
provenance digests. Consumers must query an exact
`(registry_epoch, generation, registry_digest)` through Gateway services;
they must not read these files directly at validation or execution time.
`CatalogViewSynchronizer` never leaves a stale generation visible: when a
newer identity is announced, the previously verified view is cleared and the
consumer's exact identity becomes unavailable until the new snapshot has been
fetched and verified. Callers that cannot tolerate that gap must retain the
previous generation through the registry's retain/release API instead of
relying on the synchronizer's `current` view.
