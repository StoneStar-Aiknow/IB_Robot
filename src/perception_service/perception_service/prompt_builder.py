"""Compatibility exports for shared VLM prompt builders."""

from embodied_common.vlm_prompt_builder import append_images, build_scene_analysis_messages, sanitize_user_text

_sanitize_user_text = sanitize_user_text

__all__ = ["_sanitize_user_text", "append_images", "build_scene_analysis_messages", "sanitize_user_text"]
