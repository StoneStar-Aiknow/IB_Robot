# place_in_container

The legacy primitive implementation moves the held object to the configured fixed
release pose, opens the gripper through the guarded primitive path, and requires
fresh visual confirmation that the target is inside the explicitly named container.
For that implementation, `container_name` is only a post-release GDINO query; it
does not change the fixed motion. Hermes and `robot-skill` invoke this
capability through `/embodied/execute_skill`; callers must not send `PlaceObject`
directly.

The X2 delegated placement pipeline detects and plans against the requested
container before release. Preserve the user's color, material and location
qualifiers in `container_name`. The deployment's fruit-delivery container is a
default only when the user names no destination. Ask about an ambiguous
destination; if detection fails, report failure without substituting the default.
