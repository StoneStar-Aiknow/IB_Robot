"""The visual interaction session: what is worth saying, and whether to say it now.

Nothing in this package imports ROS, so all of it can be exercised with a fake
model and a fake clock. ``visual_interaction_node`` is the only ROS adapter: it
fills these inputs from messages and publishes what the session decides.

The split follows the X2 mirror session (``manipulation_execution
.mirror_session``), for the same reason — the decisions worth testing are the
ones that have nothing to do with ROS.

| module | owns |
|---|---|
| ``semantics`` | reading one model answer, or rejecting it |

Design: docs/x2_visual_interaction_design_zh.md
"""
