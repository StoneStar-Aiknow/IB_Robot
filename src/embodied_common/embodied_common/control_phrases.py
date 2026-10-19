"""Spoken control vocabulary shared by every operator channel.

The text closed loop (robot_skill_cli), the Hermes adapter and the Agent voice
bridge all interrupt on the same stop words and continue on the same continue
words. Each channel decides how to match (whole utterance or keyword); the words
themselves live only here. Keep this module dependency-free: hermes_configure
copies it next to stop_contract.py into the installed Hermes plugins.
"""

STOP_PHRASES = ("别动了", "别动", "停止", "停下", "停", "stop", "halt")
CONTINUE_PHRASES = ("继续吧", "继续", "接着来", "go on", "continue")
