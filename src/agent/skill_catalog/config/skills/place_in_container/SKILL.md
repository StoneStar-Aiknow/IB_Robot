# place_in_container

Move the held object to the configured fixed release pose, open the gripper through
the guarded primitive path, and require fresh visual confirmation that the target is
inside the explicitly named container. `container_name` is only a post-release GDINO
query; it does not change the fixed motion. Hermes and `robot-skill` invoke this
capability through `/embodied/execute_skill`; callers must not send `PlaceObject`
directly.
