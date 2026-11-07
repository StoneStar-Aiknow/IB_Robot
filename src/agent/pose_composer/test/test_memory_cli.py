"""The maintenance CLI reads any time and writes only when it holds the memory's lock."""

import yaml

from pose_composer.memory import PoseMemory
from pose_composer.memory_cli import main


def _seeded(path):
    memory = PoseMemory(path, "fp")
    pose_id = memory.record_success([("左手向前伸直", "step")], description={"kind": "pose"}, arms=["left"],
                                    targets={"left_elbow_joint": -0.1}, user_message="好的")  # fmt: skip
    memory.feedback(pose_id, "good")
    return memory, pose_id


def test_list_and_export_work_while_the_composer_holds_the_memory(tmp_path, capsys):
    memory, pose_id = _seeded(tmp_path / "m.sqlite3")
    try:
        assert main(["--path", str(tmp_path / "m.sqlite3"), "list"]) == 0
        assert pose_id in capsys.readouterr().out
        assert main(["--path", str(tmp_path / "m.sqlite3"), "export", str(tmp_path / "out.yaml")]) == 0
        exported = yaml.safe_load((tmp_path / "out.yaml").read_text())["poses"]
        assert exported[0]["phrases"] == ["左手向前伸直"]
        assert main(["--path", str(tmp_path / "m.sqlite3"), "delete", pose_id]) == 2  # locked
    finally:
        memory.close()


def test_writes_once_the_composer_is_stopped(tmp_path):
    memory, pose_id = _seeded(tmp_path / "m.sqlite3")
    memory.close()
    assert main(["--path", str(tmp_path / "m.sqlite3"), "pin", pose_id]) == 0
    assert main(["--path", str(tmp_path / "m.sqlite3"), "ban", pose_id, "左手向前伸直"]) == 0
    reader = PoseMemory(tmp_path / "m.sqlite3", "fp", writer=False)
    assert reader.get(pose_id).state == "pinned"
    assert reader.lookup("左手向前伸直", confirmed_only=False) is None
    reader.close()
