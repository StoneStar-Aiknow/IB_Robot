"""The visual interaction session: what is worth saying, and whether to say it now.

Nothing in this package imports ROS, so all of it can be exercised with a fake
model and a fake clock. ``visual_interaction_node`` is the only ROS adapter: it
fills these inputs from messages and publishes what the session decides.

The split follows the X2 mirror session (``manipulation_execution
.mirror_session``), for the same reason — the decisions worth testing are the
ones that have nothing to do with ROS.

| module | owns |
|---|---|
| ``core`` | one observation round, and the order its checks run in |
| ``hooks`` | every extension point: the model, the speaker, the change gate |
| ``semantics`` | reading one model answer, or rejecting it |
| ``topics`` | what has been said, and what two windows have confirmed |
| ``validity`` | short-term association, session boundaries, evidence age |
| ``fake_vl`` | a scripted model, for tests and for the first version |

Design: docs/x2_visual_interaction_design_zh.md
"""
