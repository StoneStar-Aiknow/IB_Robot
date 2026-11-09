"""The node's look mode: the real __init__ and _look_round, a scripted model and clock.

Where ibrobot_msgs is not built (a workstation) a stand-in for its two messages
lets the module import, for these tests only; on the board the real ones are
used and every assignment is type-checked. Publishers and subscriptions are
recorded, not created; the callbacks are called directly and nothing spins.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types

import pytest

rclpy = pytest.importorskip("rclpy")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import CompressedImage  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from test_launch_builder import node_parameters  # noqa: E402
from test_look_builder import logged, look_config  # noqa: E402, F401 - logged is a fixture
from test_look_say import pools  # noqa: E402

from visual_interaction.launch_builders.visual_interaction import generate_visual_interaction_nodes  # noqa: E402
from visual_interaction.session import key_choice  # noqa: E402
from visual_interaction.session.hooks import SessionHooks  # noqa: E402
from visual_interaction.session.internvl_vision import QuestionError, VisionBackendError  # noqa: E402
from visual_interaction.session.look_say import QUESTIONS  # noqa: E402

_NODE_MODULE = "visual_interaction.visual_interaction_node"
_MIRROR_TOPIC = "/mirror_session/state"
_BBOX = (10.0, 20.0, 50.0, 40.0)
_JPEG = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()


def _stand_in_messages() -> dict[str, types.ModuleType]:
    class _Stamped:
        def __init__(self) -> None:
            self.header = types.SimpleNamespace(stamp=None)

    class ActorState(_Stamped):
        pass

    class VisualInteractionUtterance(_Stamped):
        pass

    msg = types.ModuleType("ibrobot_msgs.msg")
    msg.ActorState = ActorState
    msg.VisualInteractionUtterance = VisualInteractionUtterance
    package = types.ModuleType("ibrobot_msgs")
    package.msg = msg
    return {"ibrobot_msgs": package, "ibrobot_msgs.msg": msg}


@pytest.fixture(scope="module")
def node_module():
    """The node's module; test_visual_interaction_node still skips here, the stand-in leaves no trace."""
    if importlib.util.find_spec("ibrobot_msgs") is not None:
        yield importlib.import_module(_NODE_MODULE)
        return
    stand_in = _stand_in_messages()
    sys.modules.update(stand_in)
    sys.modules.pop(_NODE_MODULE, None)
    try:
        yield importlib.import_module(_NODE_MODULE)
    finally:
        for name in (*stand_in, _NODE_MODULE):
            sys.modules.pop(name, None)
        package = sys.modules.get("visual_interaction")
        if package is not None and hasattr(package, "visual_interaction_node"):
            del package.visual_interaction_node


class FakeVision:
    """InternVlVision's look surface: refuses what the real one refuses, costs what it is told."""

    def __init__(self, load_error=None, refused=None, **answers) -> None:
        self.answers = {"q1": "没什么"}
        for key, answer in answers.items():
            self.answers[key.replace("_", ".")] = answer
        self.load_error = load_error
        self.refused = dict(refused or {})
        self.question_failures: dict[str, str] = {}
        self.registered: dict = {}
        self.loads = 0
        self.prepares = 0
        self.bboxes: list = []
        self.asked: list[str] = []
        self.prepare_error = None
        self.cost = 0.0
        self.rig = None
        self.choices: dict | None = None
        self.batches: list[int] = []
        self.scored: list[str] = []
        self.score_error = None
        # What the key question answers, by answer: on live pictures and on the grey one.
        self.key_scores = {"none": 0.9}
        self.blank_scores: dict[str, float] = {}

    def register_questions(self, questions) -> None:
        self.registered = dict(questions)

    def register_choices(self, questions) -> None:
        self.choices = dict(questions)

    def load(self) -> None:
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error
        self.question_failures.update(self.refused)

    def prepare(self, frame, bbox, crop=True):
        self.prepares += 1
        self.bboxes.append(bbox)
        assert frame.shape == (48, 64, 3)
        if self.prepare_error is not None:
            raise self.prepare_error
        self.rig.now += self.cost
        return types.SimpleNamespace(image=np.zeros((448, 448, 3), np.uint8), crop_fallback=False)

    def prepare_many(self, images):
        self.prepares += 1
        self.batches.append(len(images))
        if self.prepare_error is not None:
            raise self.prepare_error
        self.rig.now += self.cost
        blank = all((image == 128).all() for image in images)
        return types.SimpleNamespace(
            embeds=np.zeros((len(images), 1)), image=images[-1], crop_fallback=False, seconds=0.0, blank=blank
        )

    def score(self, seen, key):
        self.scored.append(key)
        if self.score_error is not None:
            raise self.score_error
        wanted = self.blank_scores if seen.blank else self.key_scores
        answers = key_choice.options(key_choice.ORDERS["fwd"])
        return types.SimpleNamespace(
            probs={
                letter: wanted.get(answer, 0.02) for letter, answer in zip(key_choice.LETTERS, answers, strict=False)
            }
        )

    def ask(self, seen, key):
        if key in self.question_failures:
            raise QuestionError(f"{key} was refused at load")
        self.asked.append(key)
        answer = self.answers.get(key, "是")
        if isinstance(answer, Exception):
            raise answer
        return types.SimpleNamespace(text=answer)


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def info(self, text: str) -> None:
        self.infos.append(text)


class _Publisher:
    def __init__(self) -> None:
        self.messages: list = []

    def publish(self, message) -> None:
        self.messages.append(message)


class Rig:
    """One node, with the clock, the stream, the mirror and the voice in the test's hands."""

    def __init__(self, module, vision, *, implementation="internvl", demo_root="/nowhere", **look) -> None:
        settings = {"first_after_sec": [1.0, 1.0], "gap_sec": [4.0, 4.0], "pools": _as_json(pools())}
        settings.update(look)
        self.now = 100.0
        self.vision = vision
        if vision is not None:
            vision.rig = self
        self.log = _Log()
        self.publishers: dict[str, _Publisher] = {}
        self.busy = False
        self.heartbeat = True
        self.voice_reports = True
        self.actor_present = True
        self.jpeg = _JPEG
        self.state = ""
        self.module = module
        rig = self

        class Node(module.VisualInteractionNode):
            def _now(self) -> float:
                return rig.now

            def get_logger(self):
                return rig.log

            def create_publisher(self, _kind, topic, _qos):
                return rig.publishers.setdefault(topic, _Publisher())

            def create_subscription(self, *_args, **_kwargs):
                return None

        parameters = {
            "camera_topic": "/camera/compressed",
            "actor_state_topic": "/actor",
            "mirror_state_topic": _MIRROR_TOPIC,
            "voice_status_topic": "/mirror_voice/status",
            "vl_implementation": implementation,
            "vl_demo_root": demo_root,
            "look_json": json.dumps(settings, ensure_ascii=False),
        }
        self.node = Node(
            hooks=SessionHooks(vision=vision) if vision is not None else None,
            parameter_overrides=[Parameter(name, value=value) for name, value in parameters.items()],
        )

    @property
    def said(self) -> list:
        return self.publishers[self.module.UTTERANCE_TOPIC].messages

    @property
    def pools_said(self) -> list[str]:
        return [message.category for message in self.said]

    @property
    def look_warnings(self) -> list[str]:
        return [text for text in self.log.warnings if text.startswith("look")]

    @property
    def counts(self) -> dict[str, int]:
        return self.node._counts

    def preload(self) -> Rig:
        self.node._preload()
        return self

    def mirror(self, state: str = "MIRRORING") -> Rig:
        self.state = state
        self._heartbeat()
        return self

    def _heartbeat(self) -> None:
        self.node._on_mirror_state(String(data=json.dumps({"state": self.state})))

    def _feed(self) -> None:
        frame = CompressedImage(format="jpeg", data=self.jpeg)
        frame.header.stamp = self.module._as_time(self.now)
        self.node._on_frame(frame)
        actor = self.module.ActorState()
        actor.header.stamp = self.module._as_time(self.now)
        actor.actor_present = self.actor_present
        actor.bbox = list(_BBOX) if self.actor_present else [0.0, 0.0, 0.0, 0.0]
        actor.reason = "highest_confidence_person" if self.actor_present else "no_person"
        self.node._on_actor(actor)
        if self.voice_reports:
            self.node._on_voice_status(String(data=json.dumps({"speaking": self.busy, "in_flight": "", "queue": 0})))
        if self.heartbeat and self.state:
            self._heartbeat()

    def run(self, seconds: float, step: float = 0.05) -> Rig:
        """Ticks for this long; a look's cost moves the clock too."""
        end = self.now + seconds
        while self.now < end:
            self.now += step
            self._feed()
            self.node._look_round()
        return self

    def destroy(self) -> None:
        self.node.destroy_node()


def _as_json(clip_pools: dict) -> dict:
    return {pool: [list(clip) for clip in clips] for pool, clips in clip_pools.items()}


@pytest.fixture
def rig(node_module):
    """A factory of rigs, with rclpy up for as long as they live."""
    started = not rclpy.ok()
    if started:
        rclpy.init()
    rigs: list[Rig] = []

    def build(vision, **options) -> Rig:
        made = Rig(node_module, vision, **options)
        rigs.append(made)
        return made

    yield build
    for made in rigs:
        made.destroy()
    if started:
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# §4.6: a model that cannot load, and the questions it refuses.


def test_a_failed_load_turns_the_look_off_once_and_for_all(rig):
    vision = FakeVision(load_error=VisionBackendError("no om"), q1="比心")
    look = rig(vision).preload().mirror().run(20.0)
    assert vision.loads == 1
    assert vision.prepares == 0 and vision.asked == []
    assert look.said == []
    assert len(look.look_warnings) == 1 and "model load failed" in look.look_warnings[0]


def test_a_missing_deployment_turns_the_look_off_and_the_node_still_runs(rig):
    look = rig(None, demo_root="/nowhere").preload().mirror().run(10.0)
    assert look.said == []
    assert len(look.look_warnings) == 1 and "no model" in look.look_warnings[0]
    look.node._publish_status()
    status = json.loads(look.publishers[look.module.STATUS_TOPIC].messages[-1].data)
    assert status["look"]["ready"] is False and "no model" in status["look"]["off"]


def test_the_scripted_vision_cannot_look(rig):
    with pytest.raises(ValueError, match="internvl"):
        rig(None, implementation="fake")


def test_a_refused_q1_turns_the_look_off(rig):
    vision = FakeVision(refused={"q1": "over budget"})
    look = rig(vision).preload().mirror().run(10.0)
    assert vision.prepares == 0
    assert look.said == []
    assert len(look.look_warnings) == 1 and "q1 refused" in look.look_warnings[0]


def test_a_refused_q2_is_skipped_and_the_look_goes_on(rig):
    vision = FakeVision(refused={"q2": "over budget"}, q1="在比手势")
    look = rig(vision).preload().mirror().run(1.5)
    assert vision.prepares > 1
    assert "q2" not in vision.asked
    assert look.pools_said == ["hand.any"]
    assert any("q2 refused" in text for text in look.look_warnings)


def test_a_refused_q3_is_skipped_and_the_gesture_goes_on_q1_alone(rig):
    vision = FakeVision(refused={"q3.heart": "over budget"}, q1="比心")
    look = rig(vision).preload().mirror().run(1.5)
    assert "q3.heart" not in vision.asked
    assert look.pools_said == ["hand.heart"]
    assert look.counts["q3_skipped"] == 1


def test_every_parameter_the_builder_passes_is_declared(node_module, logged):  # noqa: F811
    """rclpy drops an undeclared override silently."""
    (node,) = generate_visual_interaction_nodes(look_config())
    assert set(node_parameters(node)) <= set(node_module.PARAMETERS)


def test_every_question_is_registered_before_the_load(rig):
    vision = FakeVision()
    rig(vision)
    assert vision.registered == dict(QUESTIONS) and vision.loads == 0


def test_nothing_is_looked_at_before_the_load(rig):
    vision = FakeVision(q1="比心")
    rig(vision).mirror().run(0.5)
    assert vision.prepares == 0


# ---------------------------------------------------------------------------
# §4.1: a look, and its line.


def test_a_named_gesture_is_verified_then_said_with_the_plan_s_fields(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.heart"]
    (message,) = look.said
    assert message.kind == look.module.KIND_CHATTER
    assert message.session_id == "look" and message.episode_seq == 1
    assert message.event_id.startswith("look-1-hand_heart_")
    assert message.text.startswith("hand.heart ")
    latest = message.latest_start_at.sec + message.latest_start_at.nanosec * 1e-9
    assert 101.0 + 2.0 <= latest <= 101.1 + 2.0
    assert vision.asked.count("q3.heart") == 1
    assert vision.bboxes[-1] == _BBOX


def test_a_q3_no_downgrades_to_the_coarse_pool(rig):
    vision = FakeVision(q1="比心", q3_heart="不是")
    look = rig(vision).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.any"]
    assert look.counts["q3_no"] >= 1


def test_a_q3_that_fails_downgrades_too(rig):
    vision = FakeVision(q1="比心", q3_heart=VisionBackendError("generator"))
    look = rig(vision).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.any"]
    assert look.counts["look_error"] >= 1


def test_q2_names_what_q1_could_not(rig):
    vision = FakeVision(q1="在比手势", q2="比心", q3_heart="是")
    look = rig(vision).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.heart"]


def test_nobody_in_the_picture_is_looked_at_whole(rig):
    vision = FakeVision()
    made = rig(vision).preload().mirror()
    made.actor_present = False
    made.run(0.2)
    assert vision.prepares > 0 and vision.bboxes[-1] is None


# ---------------------------------------------------------------------------
# Continuous looking; the voice gates only the publishing.


def test_with_no_window_every_tick_looks(rig):
    vision = FakeVision()
    rig(vision, first_after_sec=[30.0, 30.0]).preload().mirror().run(1.0, step=0.05)
    assert 19 <= vision.prepares <= 21


def test_a_look_is_not_re_entered_and_the_next_starts_after_it(rig):
    vision = FakeVision()
    vision.cost = 0.5
    rig(vision, first_after_sec=[30.0, 30.0]).preload().mirror().run(5.0, step=0.05)
    assert 9 <= vision.prepares <= 10


def test_a_period_spaces_the_looks(rig):
    vision = FakeVision()
    rig(vision, first_after_sec=[30.0, 30.0], round_period_sec=0.5).preload().mirror().run(5.0)
    assert 9 <= vision.prepares <= 11


def test_the_look_goes_on_under_a_clip_and_the_line_waits_for_quick_gap(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision, first_after_sec=[30.0, 30.0], quick_gap_sec=2.0).preload().mirror()
    look.busy = True
    look.run(5.0)
    assert vision.prepares > 50
    assert look.said == []
    look.busy = False
    quiet_at = look.now
    look.run(1.8)
    assert look.said == []
    look.run(0.4)
    assert look.pools_said == ["hand.heart"]
    stamp_of_quick_line = look.said[0].latest_start_at.sec + look.said[0].latest_start_at.nanosec * 1e-9
    assert quiet_at + 2.0 + 2.0 <= stamp_of_quick_line + 0.1


def test_a_status_from_before_the_line_never_lets_a_second_one_through(rig):
    """The double line: the voice's last status (idle, still fresh) came before our publish."""
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision, quick_gap_sec=0.5).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.heart"]
    look.voice_reports = False
    look.run(2.8)
    assert vision.asked.count("q1") > 50
    assert look.pools_said == ["hand.heart"]


def test_after_the_line_played_a_quick_reply_follows(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision, quick_gap_sec=0.5).preload().mirror().run(1.5)
    vision.answers["q1"] = "竖大拇指"
    look.busy = True
    look.run(1.0)
    look.busy = False
    look.run(0.4)
    assert look.pools_said == ["hand.heart"]
    look.run(0.2)
    assert look.pools_said == ["hand.heart", "hand.thumb"]


def test_set_enabled_false_stops_the_look_and_true_resumes_it(rig):
    from std_srvs.srv import SetBool

    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision).preload().mirror()
    look.node._on_set_enabled(SetBool.Request(data=False), SetBool.Response())
    look.run(5.0)
    assert vision.prepares == 0 and look.said == []
    look.node._on_set_enabled(SetBool.Request(data=True), SetBool.Response())
    look.run(0.2)
    assert vision.prepares > 0


def test_a_window_holds_the_look_until_shortly_before_the_line(rig):
    vision = FakeVision()
    look = rig(vision, window_sec=1.0, first_after_sec=[3.0, 3.0], fallback_after_sec=20.0).preload().mirror()
    look.run(1.9)
    assert vision.prepares == 0
    look.run(0.3)
    assert vision.prepares > 0


# ---------------------------------------------------------------------------
# The round: five lines, MIRRORING, and starting over.


def test_after_five_lines_the_round_stops_looking(rig):
    vision = FakeVision()
    look = (
        rig(vision, first_after_sec=[0.1, 0.1], gap_sec=[0.1, 0.1], fallback_after_sec=0.5, max_per_round=5)
        .preload()
        .mirror()
    )
    look.run(30.0)
    assert len(look.said) == 5
    prepares = vision.prepares
    look.run(5.0)
    assert vision.prepares == prepares and len(look.said) == 5


def test_outside_mirroring_nothing_is_looked_at_or_said(rig):
    vision = FakeVision(q1="比心")
    look = rig(vision).preload().mirror("STANDBY").run(5.0)
    assert vision.prepares == 0 and look.said == []


def test_leaving_stops_the_look_and_coming_back_is_a_new_round(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision).preload().mirror().run(1.5)
    assert look.node._look.spoken == 1
    look.mirror("STANDBY").run(2.0)
    prepares = vision.prepares
    look.run(1.0)
    assert vision.prepares == prepares
    look.mirror("MIRRORING").run(0.1)
    assert look.node._look.spoken == 0 and vision.prepares > prepares


def test_a_stale_heartbeat_holds_the_look_but_does_not_restart_the_round(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision).preload().mirror().run(1.5)
    assert look.node._look.spoken == 1
    look.heartbeat = False
    look.run(5.0)
    prepares = vision.prepares
    assert look.counts[look.module.SUPPRESS_MIRROR_INACTIVE] > 0
    look.run(1.0)
    assert vision.prepares == prepares
    look.heartbeat = True
    look.run(0.2)
    assert look.node._look.spoken >= 1
    assert vision.prepares > prepares
    assert len([text for text in look.log.infos if "a new round" in text]) == 1


# ---------------------------------------------------------------------------
# §4.6: slow and failing looks.


def test_three_slow_looks_stop_the_looking_and_the_fallback_still_speaks(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    vision.cost = 3.0
    look = rig(vision, fallback_after_sec=0.5).preload().mirror().run(20.0)
    assert vision.prepares == 3
    assert look.counts["look_slow"] == 3
    assert look.node._look.stopped
    assert any("no more looking" in text for text in look.look_warnings)
    assert look.said and set(look.pools_said) <= {"seq.invite", "other.any"}


def test_a_failing_look_is_silent_rate_limited_and_the_fallback_still_speaks(rig):
    vision = FakeVision()
    vision.prepare_error = VisionBackendError("vision om")
    look = rig(vision, fallback_after_sec=0.5).preload().mirror().run(10.0)
    assert vision.prepares > 50
    assert look.counts["look_error"] == vision.prepares
    assert len([text for text in look.look_warnings if "look failed" in text]) == 1
    assert look.said and set(look.pools_said) <= {"seq.invite", "other.any"}


def test_a_stale_stream_is_not_looked_at(rig):
    vision = FakeVision()
    look = rig(vision).preload().mirror()
    look.node._look_round()
    assert vision.prepares == 0 and look.counts["look_no_stream"] == 1


def test_a_frame_that_does_not_decode_is_not_looked_at(rig):
    vision = FakeVision()
    look = rig(vision).preload().mirror()
    look.jpeg = b"not a jpeg"
    look.run(0.2)
    assert vision.prepares == 0 and look.counts["look_bad_frame"] > 0


# ---------------------------------------------------------------------------
# dump_dir.


def test_dump_dir_keeps_each_look_s_crop_and_answers(rig, tmp_path):
    vision = FakeVision(q1="比心", q3_heart="是")
    rig(vision, dump_dir=str(tmp_path)).preload().mirror().run(1.5)
    lines = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text("utf-8").splitlines()]
    assert len(lines) == vision.prepares
    assert all(line["q1"] == "比心" and line["label"] == "hand/heart" for line in lines)
    assert all((tmp_path / line["image"]).is_file() for line in lines)
    said = [line for line in lines if "clip" in line]
    assert len(said) == 1 and said[0]["pool"] == "hand.heart" and said[0]["q3_answer"] == "是"


def test_without_dump_dir_nothing_is_written(rig, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rig(FakeVision(q1="比心")).preload().mirror().run(0.5)
    assert list(tmp_path.iterdir()) == []


def test_a_dump_that_cannot_be_written_does_not_stop_the_look(rig, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("in the way")
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision, dump_dir=str(blocked)).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.heart"]
    assert look.counts["look_dump_error"] > 0


# ---------------------------------------------------------------------------
# Key mode: the choice question every look, a confident action answered at once.

_KEY_ON = {"enabled": True, "frames": 4, "calibrate": "none", "smooth": 1, "confirm": 2}
_WAVE = {"wave": 0.9}


def test_without_key_actions_no_choice_question_exists(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision).preload().mirror().run(1.5)
    assert vision.choices is None and vision.scored == [] and vision.batches == []
    assert look.pools_said == ["hand.heart"]
    assert look.node._key_gate is None


def test_key_actions_written_off_are_the_same_look(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    look = rig(vision, key_actions={**_KEY_ON, "enabled": False}).preload().mirror().run(1.5)
    assert vision.choices is None and vision.scored == []
    assert look.pools_said == ["hand.heart"]


@pytest.mark.parametrize("frames", [1, 4])
def test_only_the_configured_key_question_is_registered(rig, frames):
    vision = FakeVision()
    look = rig(vision, key_actions={**_KEY_ON, "frames": frames}).preload().mirror().run(0.3)
    assert list(vision.choices) == [f"key.{frames}.fwd"]
    assert set(vision.batches) == {frames}
    assert set(vision.scored) == {f"key.{frames}.fwd"}
    assert vision.prepares == len(vision.batches)
    assert look.node._key_gate is not None


def test_a_confident_action_is_said_at_once_and_once_while_held(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(0.5)
    assert look.pools_said == ["big.wave"]
    assert look.counts["key_trigger_wave"] == 1
    look.run(6.0)
    assert look.pools_said.count("big.wave") == 1
    assert look.counts["key_trigger_wave"] == 1


def test_one_look_over_the_threshold_is_not_enough(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(0.05)
    assert "key_trigger_wave" not in look.counts
    vision.key_scores = {"none": 0.9}
    look.run(0.5)
    assert "key_trigger_wave" not in look.counts and look.said == []


def test_an_action_without_a_threshold_is_never_said(rig):
    vision = FakeVision()
    vision.key_scores = {"heart": 0.9}
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(0.8)
    assert look.said == []
    assert not [name for name in look.counts if name.startswith("key_trigger")]


def test_q1_is_asked_at_most_every_period_and_only_names_the_general_line(rig):
    vision = FakeVision(q1="比心")
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(1.5)
    assert vision.asked == ["q1"]
    assert look.pools_said == ["hand.any"]
    assert look.node._look.spoken == 1
    look.run(5.0)
    assert vision.asked.count("q1") == 3
    assert not [key for key in vision.asked if key != "q1"]


def test_the_key_line_waits_for_the_voice(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    made = rig(vision, key_actions=_KEY_ON).preload()
    made.busy = True
    made.mirror().run(1.0)
    assert made.said == []
    made.busy = False
    made.run(0.2)
    assert made.pools_said == ["big.wave"]


def test_a_key_line_the_voice_never_let_through_is_dropped_and_counted(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    made = rig(vision, key_actions=_KEY_ON).preload()
    made.busy = True
    made.mirror().run(4.0)
    assert made.counts["key_stale"] == 1
    made.busy = False
    made.run(0.5)
    assert "big.wave" not in made.pools_said


def test_a_full_round_still_answers_a_key_action(rig):
    vision = FakeVision(q1="比心")
    look = rig(vision, key_actions=_KEY_ON, max_per_round=1).preload().mirror().run(1.5)
    assert look.pools_said == ["hand.any"] and look.node._look.done
    look.busy = True  # the line plays
    look.run(0.3)
    look.busy = False
    asked = len(vision.asked)
    vision.key_scores = _WAVE
    look.run(5.0)
    assert look.pools_said == ["hand.any", "big.wave"]
    assert look.node._look.spoken == 1
    assert len(vision.asked) == asked


def test_a_new_round_starts_the_gate_over(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(0.5)
    assert look.counts["key_trigger_wave"] == 1
    look.mirror("STANDBY").run(1.0)
    look.mirror("MIRRORING").run(0.5)
    assert look.counts["key_trigger_wave"] == 2


def test_a_failing_key_look_feeds_nothing_to_the_gate(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    vision.prepare_error = VisionBackendError("npu")
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(0.5)
    assert look.counts["look_error"] > 0
    assert "key_trigger_wave" not in look.counts and look.said == []


def test_slow_key_looks_feed_nothing_to_the_gate_and_stop_the_looking(rig):
    vision = FakeVision()
    vision.key_scores = _WAVE
    vision.cost = 3.0
    look = rig(vision, key_actions=_KEY_ON, fallback_after_sec=0.5).preload().mirror().run(20.0)
    assert look.counts["look_slow"] == 3 and look.node._look.stopped
    assert "key_trigger_wave" not in look.counts
    assert "big.wave" not in look.pools_said


def test_the_status_says_key_mode(rig):
    look = rig(FakeVision(), key_actions=_KEY_ON).preload().mirror().run(0.1)
    look.node._publish_status()
    status = json.loads(look.publishers[look.module.STATUS_TOPIC].messages[-1].data)
    assert status["look"]["key"] is True


def test_blank_calibration_is_scored_on_grey_pictures_before_the_first_look(rig):
    vision = FakeVision()
    vision.blank_scores = {"heart": 0.6, "wave": 0.08, "yeah": 0.08, "thumb": 0.08, "open": 0.08, "none": 0.08}
    made = rig(vision, key_actions={**_KEY_ON, "calibrate": "blank"}).preload()
    assert vision.scored == ["key.4.fwd"] and vision.batches == [4]
    assert any("key blank shares" in text for text in made.log.infos)


def test_blank_calibration_is_what_lets_a_biased_answer_through(rig):
    """Raw, wave's share stays under its threshold; over the grey picture's bias it clears it."""
    live = {"wave": 0.3, "heart": 0.5}
    blank = {"heart": 0.6, "wave": 0.08, "yeah": 0.08, "thumb": 0.08, "open": 0.08, "none": 0.08}
    for calibrate, said in (("none", []), ("blank", ["big.wave"])):
        vision = FakeVision()
        vision.key_scores = live
        vision.blank_scores = blank
        look = rig(vision, key_actions={**_KEY_ON, "calibrate": calibrate}).preload().mirror().run(0.5)
        assert look.pools_said == said


def _key_went_off(look, why: str) -> None:
    assert look.node._key_gate is None and look.node._look.config.key_mode is False
    assert any("key actions off" in text and why in text for text in look.look_warnings)


def test_a_failed_blank_calibration_turns_key_mode_off_and_the_look_goes_on(rig):
    vision = FakeVision(q1="比心", q3_heart="是")
    vision.score_error = VisionBackendError("npu")
    look = rig(vision, key_actions={**_KEY_ON, "calibrate": "blank"}).preload().mirror().run(1.5)
    _key_went_off(look, "blank calibration failed")
    assert look.pools_said == ["hand.heart"]
    assert vision.scored == ["key.4.fwd"]


def test_a_refused_key_question_turns_key_mode_off_and_the_look_goes_on(rig):
    vision = FakeVision(refused={"key.4.fwd": "over budget"}, q1="比心", q3_heart="是")
    look = rig(vision, key_actions=_KEY_ON).preload().mirror().run(1.5)
    _key_went_off(look, "over budget")
    assert look.pools_said == ["hand.heart"]
    assert vision.scored == []


def test_a_vision_without_the_choice_surface_turns_the_look_off(rig):
    class Old(FakeVision):
        score = None

    look = rig(Old(), key_actions=_KEY_ON).preload().mirror().run(0.5)
    assert look.said == []
    assert any("the vision has no score" in text for text in look.look_warnings)
