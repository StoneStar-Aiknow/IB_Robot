"""The pose memory's life cycle: add, merge, confirm, ban, retire, revalidate, evict, seed."""

import pytest

from pose_composer.memory import MemoryLocked, PoseMemory, is_relative, normalize

FORWARD = {"left_shoulder_pitch_joint": -1.5, "left_elbow_joint": -0.1}
UP = {"left_shoulder_pitch_joint": -2.9, "left_elbow_joint": -0.1}
DESC = {"kind": "pose", "left": {"upper": [1, 0, 0], "fore": [1, 0, 0]}, "right": {"keep": True}}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def memory(tmp_path, clock):
    store = PoseMemory(tmp_path / "poses.sqlite3", "fp1", clock=clock, max_poses=3, candidate_ttl_sec=100.0)
    yield store
    store.close()


def _add(memory, text, targets=FORWARD, **kw):
    return memory.record_success([(text, "step")], description=DESC, arms=["left"], targets=targets,
                                 user_message="好的", **kw)  # fmt: skip


def test_normalization_ignores_courtesy_punctuation_and_synonyms():
    assert normalize("请把两只手往前伸一下吧！") == normalize("把双手往前伸")
    assert normalize("举左手") != normalize("举右手")


@pytest.mark.parametrize("text", ["再抬高一点", "左手也这样", "回到刚才的姿势", "换成侧平举"])
def test_relative_wording_is_recognized(text):
    assert is_relative(text)


def test_a_first_success_is_a_candidate_recalled_only_without_confirmation(memory):
    pose_id = _add(memory, "左手向前伸直")
    assert memory.get(pose_id).state == "candidate"
    assert memory.lookup("左手向前伸直", confirmed_only=False).pose_id == pose_id
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None


def test_a_second_success_stays_candidate_until_explicit_acceptance(memory):
    pose_id = _add(memory, "左手向前伸直")
    assert _add(memory, "左手向前伸直", pose_id=pose_id) == pose_id
    assert memory.get(pose_id).state == "candidate"
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None


def test_explicit_acceptance_confirms(memory):
    pose_id = _add(memory, "左手向前伸直")
    assert memory.feedback(pose_id, "good") == "confirmed"
    assert memory.get(pose_id).state == "active"


def test_planning_catalog_excludes_candidates_banned_and_other_fingerprints(memory):
    pose_id = _add(memory, "左手向前伸直")
    assert memory.planning_catalog() == []
    memory.feedback(pose_id, "good")
    (entry,) = memory.planning_catalog()
    assert entry["target_name"] == "左手向前伸直"
    assert entry["arms"] == ["left"] and entry["description"] == DESC
    assert memory.lookup(entry["target_name"], confirmed_only=True).pose_id == pose_id
    memory.fingerprint = "another-model"
    assert memory.planning_catalog() == []
    memory.fingerprint = "fp1"
    memory.feedback(pose_id, "bad", ["左手向前伸直"])
    assert memory.planning_catalog() == []


def test_the_same_pose_in_other_words_is_one_pose_with_two_phrases(memory):
    first = _add(memory, "左手向前伸直")
    nearly = {k: v + 0.05 for k, v in FORWARD.items()}
    assert _add(memory, "左手往前伸", targets=nearly) == first
    assert {a["phrase"] for a in memory.aliases(first)} == {normalize("左手向前伸直"), normalize("左手往前伸")}
    assert memory.get(first).state == "candidate"
    assert _add(memory, "左手举高", targets=UP) != first


def test_bad_bans_the_words_for_that_pose_and_twice_retires_it(memory):
    pose_id = _add(memory, "左手向前伸直")
    memory.feedback(pose_id, "good")
    assert memory.feedback(pose_id, "bad", ["左手向前伸直"]) == "banned for these words"
    assert memory.lookup("左手向前伸直", confirmed_only=False) is None
    _add(memory, "左手向前伸直", pose_id=pose_id)  # the banned pairing is not restored
    assert memory.lookup("左手向前伸直", confirmed_only=False) is None
    assert memory.feedback(pose_id, "bad", []) == "retired"
    assert memory.get(pose_id).state == "retired"


def test_a_phrase_moves_to_the_newer_pose_and_the_old_link_is_kept_as_superseded(memory):
    old = _add(memory, "左手抬起来")
    new = _add(memory, "左手抬起来", targets=UP)
    assert memory.lookup("左手抬起来", confirmed_only=False).pose_id == new
    assert [a["state"] for a in memory.aliases(old)] == ["superseded"]


def test_relative_words_are_never_attached_or_recalled(memory):
    pose_id = _add(memory, "再抬高一点")
    assert memory.aliases(pose_id) == []
    assert memory.lookup("再抬高一点", confirmed_only=False) is None


def test_a_new_fingerprint_makes_poses_stale_until_revalidated(tmp_path, clock):
    first = PoseMemory(tmp_path / "m.sqlite3", "fp1", clock=clock)
    pose_id = _add(first, "左手向前伸直")
    first.feedback(pose_id, "good")
    first.close()
    second = PoseMemory(tmp_path / "m.sqlite3", "fp2", clock=clock)
    assert second.mark_stale() == 1
    assert second.lookup("左手向前伸直", confirmed_only=False) is None
    counts = second.revalidate(
        lambda description: (["left"], {"left_shoulder_pitch_joint": -1.4, "left_elbow_joint": -0.1})
    )
    assert counts == {"restored": 1, "retired": 0, "merged": 0}
    record = second.get(pose_id)
    assert (record.state, record.version, record.targets["left_shoulder_pitch_joint"]) == ("active", 2, -1.4)
    second.close()


def test_a_pose_no_longer_solvable_is_retired(tmp_path, clock):
    first = PoseMemory(tmp_path / "m.sqlite3", "fp1", clock=clock)
    pose_id = _add(first, "左手向前伸直")
    first.close()
    second = PoseMemory(tmp_path / "m.sqlite3", "fp2", clock=clock)
    second.mark_stale()
    assert second.revalidate(lambda description: None)["retired"] == 1
    assert second.get(pose_id).state == "retired"
    second.close()


def test_unused_candidates_expire_and_capacity_keeps_pinned_and_useful_poses(memory, clock):
    stale_candidate = _add(memory, "a")
    clock.now += 200.0
    memory.evict()
    assert memory.get(stale_candidate).state == "retired"
    poses = [_add(memory, f"p{k}", targets={"left_elbow_joint": -0.5 * k - 0.2}) for k in range(3)]
    memory.set_pinned(poses[0], True)
    memory.feedback(poses[1], "good")
    _add(memory, "p-new", targets={"left_elbow_joint": -2.0})  # one over capacity
    assert memory.get(poses[0]).state == "pinned"
    assert memory.get(poses[2]).state == "retired"


def test_seeds_are_pinned_merged_and_recallable_after_revalidation(memory):
    entries = [
        {"id": "a", "phrases": ["秀肌肉"], "description": DESC},
        {"id": "b", "phrases": ["投降"], "description": DESC},
    ]
    assert memory.add_described(entries, source="seed", pinned=True, meta={"seed_version": "1"}) == 2
    assert memory.lookup("秀肌肉", confirmed_only=True) is None  # stale until solved
    counts = memory.revalidate(lambda description: (["left"], dict(FORWARD)))
    assert counts["merged"] == 1
    assert memory.lookup("秀肌肉", confirmed_only=True).pose_id == memory.lookup("投降", confirmed_only=True).pose_id
    assert memory.meta("seed_version") == "1"


def test_every_change_is_audited(memory):
    pose_id = _add(memory, "左手向前伸直")
    memory.feedback(pose_id, "good")
    actions = [row["action"] for row in memory.audit_log()]
    assert {"added", "alias_added", "confirmed"} <= set(actions)


def test_only_one_writer(tmp_path):
    first = PoseMemory(tmp_path / "m.sqlite3", "fp")
    with pytest.raises(MemoryLocked):
        PoseMemory(tmp_path / "m.sqlite3", "fp")
    PoseMemory(tmp_path / "m.sqlite3", "fp", writer=False).close()
    first.close()


def test_one_visible_pose_reached_with_other_joint_angles_is_still_one_pose(memory):
    """A straight arm's shoulder yaw is free: the signature (where the arm is) decides."""
    signature = {"left_elbow": [0.2, 0.0, 0.0], "left_wrist": [0.4, 0.0, 0.0], "left_hand": [0.48, 0.0, 0.0]}
    first = _add(memory, "左手向前伸直", signature=signature)
    twisted = {"left_shoulder_pitch_joint": -1.5, "left_elbow_joint": -0.1, "left_shoulder_yaw_joint": 0.8}
    assert _add(memory, "左手平伸向前", targets=twisted, signature=signature) == first
    elsewhere = {k: [v[0], v[1], v[2] + 0.2] for k, v in signature.items()}
    assert _add(memory, "左手斜向上", signature=elsewhere) != first
