"""ROS-free stop vocabulary shared with the installed Hermes adapter."""

STOP_PHRASES = frozenset({"别动了", "别动", "停止", "停下", "停", "stop", "halt"})


def is_stop(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in STOP_PHRASES
