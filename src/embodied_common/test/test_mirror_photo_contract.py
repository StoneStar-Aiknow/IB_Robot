"""The photo trigger contract: what a sender writes is what the photo node reads, and nothing else comes with it.

The message itself is tested from the reader's side in
``manipulation_execution/test/test_mirror_photo_config.py``; here only what
any other package may rely on when it imports the contract.
"""

import subprocess
import sys

from embodied_common.mirror_photo_contract import TRIGGER_TOPIC, TriggerRequest, parse_trigger, trigger_json


def test_a_sender_and_the_node_agree_on_the_message():
    text = trigger_json("heart", source="preset_motions", score=0.7, stamp=101.25)
    assert parse_trigger(text) == TriggerRequest("preset_motions", "heart", 0.7, 101.25)
    assert TRIGGER_TOPIC.startswith("/")


def test_the_module_stays_dependency_free():
    script = (
        "import sys; import embodied_common.mirror_photo_contract; "
        "leaked = sorted("
        "  name for name in sys.modules"
        "  if name.split('.')[0] in {'rclpy', 'launch', 'robot_config', 'ibrobot_msgs', 'manipulation_execution',"
        "  'numpy', 'cv2'}"
        ");"
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
