"""robot_mcp - MCP tool layer for IB-Robot runtime skills and status.

Phase 0 (read-only): exposes ``list_skills`` / ``list_poses`` / ``get_status``.
The skill/pose catalog is driven by the robot_config SSOT (single YAML), and
runtime status is sampled from the existing ROS 2 topics (no new publishers).
"""

__all__ = ["server"]
