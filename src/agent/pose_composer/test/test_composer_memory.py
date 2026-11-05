"""The composer with a memory: recall without the model, remember what succeeded, take the verdict."""

import json

import pytest

from pose_composer import kin
from pose_composer.composer import Composer
from pose_composer.memory import PoseMemory
from pose_composer.model import load

FORWARD = {"kind": "pose", "left": {"upper": [1, 0, 0], "fore": [1, 0, 0]}, "right": {"keep": True},
           "user_message": "好的"}  # fmt: skip


@pytest.fixture
def memory(tmp_path):
    store = PoseMemory(tmp_path / "poses.sqlite3", "fp")
    yield store
    store.close()


def _composer(memory, reply=FORWARD):
    calls = []

    def model(messages):
        calls.append(messages)
        return json.dumps(reply, ensure_ascii=False)

    return Composer(model, memory=memory), calls


def _run(composer, text, start, session="s1", utterance=""):
    prepared = composer.compose(text, start, session_id=session, utterance=utterance)
    composer.take(prepared.text, prepared.composition_id)
    return prepared, composer.succeeded(prepared)


def test_only_an_explicitly_saved_pose_skips_the_model_and_keeps_the_other_arm(memory):
    composer, calls = _composer(memory)
    rest = dict(load().rest)
    first, pose_id = _run(composer, "左手向前伸直", rest, utterance="请把左手向前伸直")
    assert pose_id and not first.from_memory and len(calls) == 1
    assert composer.feedback("s1", "good")[0] == ""  # no question, no consent
    _, same = _run(composer, "左手向前伸直", rest)
    assert same == pose_id and len(calls) == 2
    assert composer.feedback("s1", "offer")[0] == pose_id
    assert composer.feedback("s1", "good") == (pose_id, "confirmed")
    raised_right = dict(rest, right_shoulder_pitch_joint=-1.0)
    again, same = _run(composer, "左手往前伸直", raised_right, utterance="请把左手向前伸直")
    assert again.from_memory and len(calls) == 2 and same == pose_id
    frame = again.composition.frames[-1]
    assert frame["right_shoulder_pitch_joint"] == pytest.approx(-1.0, abs=0.05)  # the kept arm stays put
    assert kin.angle_deg(kin.fk(frame).fore("left"), [1, 0, 0]) < 10
    assert again.composition.user_message == "好的，请把左手向前伸直"
    assert memory.get(pose_id).state == "active"


def test_relative_or_context_dependent_poses_are_not_remembered(memory):
    composer, _ = _composer(memory, dict(FORWARD, depends_on_current=True))
    _, pose_id = _run(composer, "左手照右手那样", dict(load().rest))
    assert pose_id == ""
    composer, _ = _composer(memory)
    _, pose_id = _run(composer, "再抬高一点", dict(load().rest))
    assert pose_id == ""


def test_normalized_target_does_not_hide_relative_original_request(memory):
    composer, calls = _composer(memory)
    rest = dict(load().rest)
    _, pose_id = _run(composer, "左手向前伸直", rest)
    memory.feedback(pose_id, "good")
    relative = "左手再向前伸一点"
    prepared = composer.compose("左手向前伸直", rest, utterance=relative)
    assert not prepared.from_memory and len(calls) == 2
    assert relative in json.dumps(calls[-1], ensure_ascii=False)


def test_two_successes_only_offer_saving_and_next_task_does_not_confirm(memory):
    composer, _ = _composer(memory)
    rest = dict(load().rest)
    _, pose_id = _run(composer, "左手向前伸直", rest)
    assert composer.feedback("s1", "offer")[0] == ""
    _, same = _run(composer, "左手向前伸直", rest)
    assert same == pose_id
    assert composer.feedback("s1", "offer") == (pose_id, "刚才这个动作到位吗？可以保存吗？")
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None
    composer.feedback("s1", "next_task")
    assert composer.feedback("s1", "good")[0] == ""
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None


def test_save_confirmation_is_bound_to_session_and_expires(memory, monkeypatch):
    from pose_composer import composer as module

    composer, _ = _composer(memory)
    rest = dict(load().rest)
    _run(composer, "左手向前伸直", rest)
    _, pose_id = _run(composer, "左手向前伸直", rest)
    composer.feedback("s1", "offer")
    assert composer.feedback("another-person", "good")[0] == ""
    now = module.time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: now + 61.0)
    assert composer.feedback("s1", "good")[0] == ""
    assert memory.get(pose_id).state == "candidate"


def test_other_sessions_near_pose_cannot_replace_the_pose_being_confirmed(memory):
    composer, _ = _composer(memory)
    rest = dict(load().rest)
    _run(composer, "左手向前伸直", rest)
    _, pose_id = _run(composer, "左手向前伸直", rest)
    assert composer.feedback("s1", "offer")[0] == pose_id
    near = dict(FORWARD, left={"upper": [1, 0, 0.03], "fore": [1, 0, 0.03]})
    other, _ = _composer(memory, near)
    _, same = _run(other, "左手向前伸直", rest, session="other")
    assert same == pose_id
    assert composer.feedback("s1", "good")[0] == ""
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None


def test_rejected_save_never_enters_recall(memory):
    composer, _ = _composer(memory)
    rest = dict(load().rest)
    _run(composer, "左手向前伸直", rest)
    _, pose_id = _run(composer, "左手向前伸直", rest)
    composer.feedback("s1", "offer")
    composer.feedback("s1", "bad")
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None
    assert memory.get(pose_id).state == "candidate"


def test_approximate_pose_requires_explicit_acceptance_of_achieved_pose(memory):
    composer, _ = _composer(memory)
    rest = dict(load().rest)
    for _ in range(2):
        prepared = composer.compose("左手向前伸直", rest, session_id="s1")
        prepared.composition.approx = True
        composer.take(prepared.text, prepared.composition_id)
        pose_id = composer.succeeded(prepared)
    assert composer.feedback("s1", "offer") == (pose_id, "刚才实际完成的是近似姿态，到位吗？可以保存吗？")
    assert memory.lookup("左手向前伸直", confirmed_only=True) is None
    assert composer.feedback("s1", "good") == (pose_id, "confirmed")
    assert memory.lookup("左手向前伸直", confirmed_only=True).pose_id == pose_id


def test_the_verdict_applies_to_the_conversations_last_pose_only(memory):
    composer, _ = _composer(memory)
    _, pose_id = _run(composer, "左手向前伸直", dict(load().rest), session="a")
    assert composer.feedback("b", "good") == ("", "no pose awaiting a verdict")
    assert composer.feedback("a", "bad") == (pose_id, "banned for these words")
    assert composer.feedback("a", "good")[0] == ""  # consumed


def test_relative_history_is_kept_per_conversation(memory):
    composer, calls = _composer(memory, dict(FORWARD, depends_on_current=True))
    _run(composer, "左手往前", dict(load().rest), session="a")
    composer.compose("再高一点", dict(load().rest), session_id="b")
    assert "左手往前" not in calls[-1][1]["content"]
    composer.compose("再高一点", dict(load().rest), session_id="a")
    assert "左手往前" in calls[-1][1]["content"]
