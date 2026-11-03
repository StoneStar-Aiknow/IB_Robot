"""The entity ground-truth streams (#205 grasp stage D3).

Unit tests for the gz ``Pose_V`` parser (quaternion order, the same link
name under different models, model-relative composition, the missing
entity; the model-name resolution for single-model worlds — review round 1
item 17) and for the MuJoCo link kinematics (a fixture model with a known
pose, a tilted base, a missing joint, an extension-declaring model, the
ancestor-chain floating joint of a free-jointed sub-body — review round 1
items 5/17). No simulator and no ROS graph: the parsers and the kinematics
run on text fixtures and tiny MJCF files.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from robot_runtime_sim.testing.entity_ground_truth import (
    GroundTruthPose,
    _record_samples,
    open_entity_ground_truth,
    parse_entity_pose_messages,
)


def _message(
    entries: list[tuple[str, int | None, tuple[float, float, float], tuple[float, float, float, float]]],
) -> str:
    """One gz Pose_V text message: a stamp then flat pose entries."""
    lines = ["header {", "  stamp {", "    sec: 100", "    nsec: 500000000", "  }", "}"]
    for name, entity_id, position, quat in entries:
        lines.append(f'pose {{ name: "{name}"')
        if entity_id is not None:
            lines.append(f"  id: {entity_id}")
        lines.append(f"  position {{ x: {position[0]} y: {position[1]} z: {position[2]} }}")
        lines.append(f"  orientation {{ x: {quat[0]} y: {quat[1]} z: {quat[2]} w: {quat[3]} }}")
        lines.append("}")
    return "\n".join(lines) + "\n"


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def test_the_parser_returns_the_entity_world_pose_in_xyzw_order():
    # a 90-deg YAW quaternion has the value in the z slot (xyzw order)
    message = _message([("cube", 8, (1.0, 2.0, 3.0), (0.0, 0.0, 0.7071067811865476, 0.7071067811865476))])
    # final=True: the trailing message is the last one, so it parses too
    samples, tail = parse_entity_pose_messages(
        message + _message([("cube", 8, (1.1, 2.0, 3.0), IDENTITY)]), "cube", None, {"cube"}, final=True
    )
    assert tail == ""
    assert len(samples) == 2
    assert samples[0].stamp_sec == pytest.approx(100.5)
    assert samples[0].position == pytest.approx((1.0, 2.0, 3.0))
    # a 90-deg yaw quaternion parsed in xyzw order: yaw_from_quaternion
    # reproduces pi/2 only when x/y/z/w keep their slots
    from robot_runtime_sim.testing.sim_ground_truth import yaw_from_quaternion

    assert yaw_from_quaternion(*samples[0].quat_xyzw) == pytest.approx(math.pi / 2)
    assert samples[0].quat_xyzw[3] == pytest.approx(0.7071067811865476)  # w last


def test_the_parser_composes_the_model_relative_link_pose():
    """The link entry is MODEL-relative: link world = model ∘ link (one
    composition), proven with a rotated model and an offset link."""
    yaw90 = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    # the real broadcaster layout: every MODEL first (ids ascending), then
    # every LINK (the ground plane's link id 3 < the robot's 4 marks the
    # descent that splits the blocks)
    message = _message(
        [
            ("ground_plane", 2, (0.0, 0.0, 0.0), IDENTITY),
            ("robot", 4, (1.0, 0.0, 0.05), yaw90),
            ("link", 3, (0.0, 0.0, 0.0), IDENTITY),
            ("base", 5, (0.0, 0.0, 0.0), IDENTITY),
            ("gripper", 7, (0.2, 0.0, 0.1), IDENTITY),
        ]
    )
    samples, _tail = parse_entity_pose_messages(
        message + message, "robot", "gripper", {"ground_plane", "robot"}, final=True
    )
    assert len(samples) == 2
    # the model's +90-deg yaw rotates the link's (0.2, 0, 0.1) offset
    assert samples[0].position == pytest.approx((1.0, 0.2, 0.15), abs=1e-9)
    assert samples[0].quat_xyzw == pytest.approx(yaw90, abs=1e-9)


# ---------------------------------------------------------------------------
# the model-name resolution (review round 1, item 17: GT API generality)
# ---------------------------------------------------------------------------


def test_the_parser_resolves_a_single_model_world_by_name():
    """A SINGLE-model world has no id descent between the model and its
    links: the models are therefore exactly the NAMED entries and the link
    scope comes from the named-model ids — an id-prefix heuristic would eat
    the links as models and never find the link."""
    message = _message(
        [
            ("robot", 2, (1.0, 0.0, 0.05), IDENTITY),
            ("base", 3, (0.0, 0.0, 0.0), IDENTITY),
            ("gripper", 5, (0.2, 0.0, 0.1), IDENTITY),
        ]
    )
    doubled = message + message
    samples, _ = parse_entity_pose_messages(doubled, "robot", "gripper", {"robot"}, final=True)
    assert len(samples) == 2
    assert samples[0].position == pytest.approx((1.2, 0.0, 0.15), abs=1e-9)


def test_the_parser_scopes_the_same_link_name_through_the_model_name_set():
    """The two-model `tip` case, by name: the link scope is computed from
    the NAMED models — beta's tip must not leak into alpha's scope and
    vice versa (the same result the prefix heuristic produces on the
    broadcaster's model-first layout)."""
    message = _message(
        [
            ("alpha", 4, (1.0, 0.0, 0.0), IDENTITY),
            ("beta", 10, (-1.0, 0.0, 0.0), IDENTITY),
            ("base", 5, (0.0, 0.0, 0.0), IDENTITY),
            ("tip", 7, (0.5, 0.0, 0.0), IDENTITY),
            ("base", 11, (0.0, 0.0, 0.0), IDENTITY),
            ("tip", 13, (0.0, 0.25, 0.0), IDENTITY),
        ]
    )
    doubled = message + message
    names = {"alpha", "beta"}
    alpha, _ = parse_entity_pose_messages(doubled, "alpha", "tip", names, final=True)
    beta, _ = parse_entity_pose_messages(doubled, "beta", "tip", names, final=True)
    assert alpha[0].position == pytest.approx((1.5, 0.0, 0.0), abs=1e-9)
    assert beta[0].position == pytest.approx((-1.0, 0.25, 0.0), abs=1e-9)


def test_the_model_name_set_rejects_an_entity_outside_it():
    """With ``model_names`` given, the requested entity must be among the
    named models — an entity outside the set yields no samples (as a
    missing entity does); naming it puts it back."""
    message = _message([("robot", 4, (0.0, 0.0, 0.0), IDENTITY), ("cube", 9, (1.0, 2.0, 3.0), IDENTITY)])
    samples, _ = parse_entity_pose_messages(message + message, "cube", None, {"robot"}, final=True)
    assert samples == []
    samples, _ = parse_entity_pose_messages(message + message, "cube", None, {"robot", "cube"}, final=True)
    assert samples[0].position == pytest.approx((1.0, 2.0, 3.0))


def test_the_parser_returns_no_samples_for_a_missing_entity():
    message = _message([("cube", 8, (1.0, 2.0, 3.0), IDENTITY)])
    samples, tail = parse_entity_pose_messages(message + message, "sphere", None, {"cube"}, final=True)
    assert samples == [] and tail == ""
    samples, _ = parse_entity_pose_messages(message + message, "robot", "gripper", {"robot"}, final=True)
    assert samples == []


def test_the_parser_keeps_the_trailing_partial_message_in_the_tail():
    partial = _message([("cube", 8, (1.0, 2.0, 3.0), IDENTITY)])
    complete = _message([("cube", 8, (0.0, 0.0, 0.0), IDENTITY)])
    samples, tail = parse_entity_pose_messages(complete + partial, "cube", None, {"cube"})
    assert len(samples) == 1
    assert "cube" in tail  # the trailing chunk waits for the next header
    samples, tail = parse_entity_pose_messages(complete + partial, "cube", None, {"cube"}, final=True)
    assert len(samples) == 2 and tail == ""


def test_the_sample_store_restarts_the_history_on_a_rewound_clock():
    """A world reset rewinds the simulator clock: the pre-reset history is
    dropped, because every reader must see a strictly time-ordered span —
    and the reset is COUNTED (review r1 item 8): a silent history restart
    must never present the post-reset span as the whole run."""
    history: list[GroundTruthPose] = []
    assert (
        _record_samples(
            [GroundTruthPose(10.0, (1.0, 0.0, 0.0), IDENTITY), GroundTruthPose(10.04, (1.2, 0.0, 0.0), IDENTITY)],
            history,
        )
        == 0
    )
    assert [s.stamp_sec for s in history] == [10.0, 10.04]
    assert _record_samples([GroundTruthPose(0.5, (-1.0, 0.0, 0.0), IDENTITY)], history) == 1  # one reset
    assert [s.stamp_sec for s in history] == [0.5]
    # a batch that rewinds twice within itself counts both epochs (each
    # rewind is beyond the hiccup margin — a real reset each time)
    assert (
        _record_samples(
            [
                GroundTruthPose(0.3, (0.0, 0.0, 0.0), IDENTITY),
                GroundTruthPose(0.1, (0.1, 0.0, 0.0), IDENTITY),
                GroundTruthPose(0.15, (0.2, 0.0, 0.0), IDENTITY),
            ],
            history,
        )
        == 2
    )
    assert [s.stamp_sec for s in history] == [0.1, 0.15]


def test_the_sample_store_drops_publisher_hiccups_without_a_reset():
    """A rewind WITHIN the margin is a publisher hiccup, not a world reset
    (measured on the GUI screenshot runs: a capture-stalled publisher
    re-emits a duplicate or briefly reordered stamp): the non-advancing
    sample is dropped and the run is not voided — only a rewind beyond the
    margin counts as a reset."""
    history: list[GroundTruthPose] = []
    assert (
        _record_samples(
            [GroundTruthPose(10.0, (1.0, 0.0, 0.0), IDENTITY), GroundTruthPose(10.04, (1.2, 0.0, 0.0), IDENTITY)],
            history,
        )
        == 0
    )
    # the frozen-clock duplicate: the same stamp re-emitted — dropped
    assert _record_samples([GroundTruthPose(10.04, (1.2, 0.0, 0.0), IDENTITY)], history) == 0
    assert [s.stamp_sec for s in history] == [10.0, 10.04]
    # a briefly reordered stamp within the publish period — dropped
    assert _record_samples([GroundTruthPose(10.01, (1.2, 0.0, 0.0), IDENTITY)], history) == 0
    assert [s.stamp_sec for s in history] == [10.0, 10.04]
    # the stream continues advancing afterwards
    assert _record_samples([GroundTruthPose(10.08, (1.3, 0.0, 0.0), IDENTITY)], history) == 0
    assert [s.stamp_sec for s in history] == [10.0, 10.04, 10.08]


def test_the_gazebo_stream_derives_its_model_names_from_the_scene():
    """Item 17: the Gazebo stream hands the parser a NAME-derived model set
    — the scene layout's static_geometry + objects names (one definition,
    the same layout.yaml the world templates generate from), the
    world-template scaffolding (ground_plane/sun) and the entity itself —
    so a live stream never leans on the ascending-id-prefix assumption that
    a single-model world defeats."""
    stream = open_entity_ground_truth(
        "gazebo", world="demo", scene="nav_grasp_room", entity="lekiwi_robot", link="gripper", env=None
    )
    assert "lekiwi_robot" in stream._model_names  # the requested entity itself
    assert "grasp_object" in stream._model_names  # the layout's dynamic object
    assert "table" in stream._model_names and "wall_north" in stream._model_names  # static geometry
    assert {"ground_plane", "sun"} <= stream._model_names  # the world-template scaffolding
    # an unknown scene contributes only the scaffolding + the entity
    stream = open_entity_ground_truth("gazebo", world="demo", scene="no_such_scene", entity="cube", env=None)
    assert stream._model_names == {"ground_plane", "sun", "cube"}


# ---------------------------------------------------------------------------
# the MuJoCo link kinematics (fixture models, no ROS graph)
# ---------------------------------------------------------------------------

FixtureModel = """
<mujoco model="fixture">
  <worldbody>
    <body name="base" pos="{base_pos}" quat="{base_quat}">
      <freejoint name="base_free"/>
      <geom name="base_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="arm" pos="0.1 0 0">
        <joint name="arm_joint" type="hinge" axis="0 0 1" ref="0"/>
        <geom name="arm_geom" type="box" size="0.02 0.02 0.02"/>
        <body name="gripper" pos="0.1 0 0">
          <geom name="gripper_geom" type="box" size="0.01 0.01 0.01"/>
        </body>
      </body>
      <body name="wheel" pos="-0.05 0 0">
        <joint name="wheel_joint" type="hinge" axis="0 1 0" ref="0"/>
        <geom name="wheel_geom" type="sphere" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def kinematics(tmp_path):
    pytest.importorskip("mujoco")
    from robot_runtime_sim.testing.mujoco_link_ground_truth import _MjcfKinematics

    path = tmp_path / "fixture.xml"
    path.write_text(FixtureModel.format(base_pos="0 0 0", base_quat="1 0 0 0"))
    return _MjcfKinematics(str(path))


def test_the_kinematics_computes_the_known_link_pose(kinematics):
    """At the zero configuration the gripper sits at base+(0.2, 0, 0); a
    90-deg hinge rotates it to base+(0, 0.2, 0)."""
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    pose = kinematics.link_pose("gripper", identity, {"arm_joint": 0.0})
    assert pose.position == pytest.approx((0.2, 0.0, 0.0), abs=1e-9)
    pose = kinematics.link_pose("gripper", identity, {"arm_joint": math.pi / 2})
    # the hinge turns the gripper's (0.1, 0, 0) offset ABOUT THE ARM's origin
    assert pose.position == pytest.approx((0.1, 0.1, 0.0), abs=1e-9)


def test_the_kinematics_applies_the_full_floating_base_quaternion(kinematics):
    """A pitched base (30 deg about y) moves the gripper's +x offset into
    -z; the body quaternion comes back in xyzw order (MuJoCo ``xquat`` is
    w-x-y-z, so the conversion is pinned by the order check)."""
    base = ((0.5, -0.2, 0.1), (0.0, math.sin(math.pi / 12), 0.0, math.cos(math.pi / 12)))
    pose = kinematics.link_pose("gripper", base, {"arm_joint": 0.0})
    # the gripper offset (0.2, 0, 0) under a 30-deg pitch about y:
    # x' = 0.2*cos30, z' = -0.2*sin30
    assert pose.position[0] == pytest.approx(0.5 + 0.2 * math.cos(math.pi / 6), abs=1e-9)
    assert pose.position[2] == pytest.approx(0.1 - 0.2 * math.sin(math.pi / 6), abs=1e-9)
    # the orientation: the pitch quaternion itself — in xyzw slots
    assert pose.quat_xyzw == pytest.approx(base[1], abs=1e-9)
    assert pose.quat_xyzw[3] == pytest.approx(math.cos(math.pi / 12))  # w LAST


def test_the_kinematics_rejects_a_missing_joint(kinematics):
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert kinematics.link_pose("gripper", identity, {}) is None
    assert kinematics.link_pose("gripper", identity, {"other_joint": 0.0}) is None


def test_the_kinematics_rejects_unknown_bodies(kinematics):
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(KeyError, match="no body named 'wrist'"):
        kinematics.link_pose("wrist", identity, {"arm_joint": 0.0})


def test_the_kinematics_writes_the_live_joint_positions_with_a_moved_base(kinematics):
    """Regression (found on the MuJoCo reach test, 2026-10-30): the live
    joint values must reach qpos — the bug left the arm at the model
    default (all zeros) and the ground-truth gripper pose read 158 mm off
    while compute_fk agreed with tf. Non-zero free joint AND a non-zero
    arm joint: the 90-deg hinge puts the gripper at base-local
    (0.1, 0.1, 0), the 90-deg base yaw maps it to (-0.1, 0.1, 0)."""
    base = ((1.0, -0.5, 0.2), (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    pose = kinematics.link_pose("gripper", base, {"arm_joint": math.pi / 2})
    assert pose.position == pytest.approx((1.0 - 0.1, -0.5 + 0.1, 0.2), abs=1e-9)


def test_the_kinematics_requires_only_the_link_chain_joints(kinematics):
    """A joint OFF the link's chain (the wheel hangs under the base, not on
    the base->gripper path) may be absent — /joint_states (the arm-only
    broadcaster) carries no wheel joints, and such a joint cannot move the
    link; a joint ON the chain must be present or the state is incomplete."""
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    pose = kinematics.link_pose("gripper", identity, {"arm_joint": 0.0})
    assert pose.position == pytest.approx((0.2, 0.0, 0.0), abs=1e-9)
    assert kinematics.link_pose("gripper", identity, {"wheel_joint": 0.3}) is None


def test_the_kinematics_returns_the_reference_body_relative_pose(kinematics):
    """With ``reference_body`` the pose is returned in that body's frame
    (the reach test reads the gripper relative to the arm root ``base``):
    the base pose cancels — a moved base changes the world pose but not the
    relative one, and the relative orientation is the frame's inverse
    composed with the link's."""
    moved = ((1.0, -0.5, 0.2), (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    pose = kinematics.link_pose("gripper", moved, {"arm_joint": 0.0}, reference_body="base")
    assert pose.position == pytest.approx((0.2, 0.0, 0.0), abs=1e-9)
    assert pose.quat_xyzw == pytest.approx(IDENTITY, abs=1e-9)


def test_the_kinematics_loads_extension_declaring_models(tmp_path):
    """A model declaring an engine extension loads only after
    load_mujoco_engine_plugins — the same path the provider takes."""
    pytest.importorskip("mujoco")
    from robot_runtime_sim.testing.mujoco_link_ground_truth import _MjcfKinematics
    from sim_models.scene_compiler import mjcf_engine_extensions

    path = tmp_path / "extension.xml"
    path.write_text(
        FixtureModel.format(base_pos="0 0 0", base_quat="1 0 0 0").replace(
            "<worldbody>",
            '<extension><plugin plugin="robot_runtime::testing::no_such_plugin"/></extension><worldbody>',
        )
    )
    assert mjcf_engine_extensions(path) == ["robot_runtime::testing::no_such_plugin"]
    with pytest.raises(RuntimeError, match="no_such_plugin"):
        _MjcfKinematics(str(path))  # the unregistered extension is named, not silently ignored


# ---------------------------------------------------------------------------
# the ancestor-chain floating joint (review round 1, item 17)
# ---------------------------------------------------------------------------

#: A `fixed_arm` (a hinge — joint 0) plus a free-jointed top-level `float`
#: carrying `tool`: MuJoCo only allows a free joint on a TOP-LEVEL body, so
#: "the free joint is not joint 0" is realized as a floating body that is
#: not the model's first body — the shape the old joint-0 lookup misread as
#: "no floating base" (e.g. a composed scene whose dynamic objects precede
#: the robot). `fixed_arm` doubles as the no-free-ancestor link.
SUBBODY_MODEL = """
<mujoco model="subbody">
  <worldbody>
    <body name="fixed_arm" pos="0 0 0.2">
      <joint name="arm_hinge" type="hinge" axis="0 0 1" ref="0"/>
      <geom name="fixed_arm_geom" type="box" size="0.01 0.01 0.01"/>
    </body>
    <body name="float" pos="0.1 0 0.05">
      <freejoint name="float_free"/>
      <geom name="float_geom" type="box" size="0.02 0.02 0.02"/>
      <body name="tool" pos="0.05 0 0">
        <geom name="tool_geom" type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def subbody(tmp_path):
    pytest.importorskip("mujoco")
    from robot_runtime_sim.testing.mujoco_link_ground_truth import _MjcfKinematics

    path = tmp_path / "subbody.xml"
    path.write_text(SUBBODY_MODEL, encoding="utf-8")
    return _MjcfKinematics(str(path))


def test_a_free_joint_below_joint_zero_resolves_from_the_ancestor_chain(subbody):
    """Item 17: the floating joint of this fixture is NOT joint 0 — the
    old joint-0 lookup read `None` (no floating base) and silently dropped
    the base pose; the ancestor-chain walk finds `float_free` for every
    link under `float` (MuJoCo restricts free joints to top-level bodies,
    so "not joint 0" means the floating body is not the model's first)."""
    mujoco = pytest.importorskip("mujoco")
    float_free = mujoco.mj_name2id(subbody.model, mujoco.mjtObj.mjOBJ_JOINT, "float_free")
    assert subbody.free_joint_for("tool") == int(subbody.model.jnt_qposadr[float_free])
    assert subbody.free_joint_for("float") == int(subbody.model.jnt_qposadr[float_free])
    # the base pose drives the free-jointed sub-body's links
    pose = subbody.link_pose("tool", ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)), {"arm_hinge": 0.0})
    assert pose.position == pytest.approx((1.05, 2.0, 3.0), abs=1e-9)
    # a chain WITH a free joint needs the base pose — never half-assembled
    assert subbody.link_pose("tool", None, {"arm_hinge": 0.0}) is None


def test_a_link_without_a_free_ancestor_uses_the_models_own_placement(subbody):
    """`fixed_arm` carries only a hinge — no free ancestor, so
    free_joint_for is None and the model's own root placement governs: a
    supplied base pose is ignored, never misapplied to another body's free
    joint (the old code would have written it into the root free joint of
    any model that HAD one)."""
    assert subbody.free_joint_for("fixed_arm") is None
    moved = subbody.link_pose("fixed_arm", ((9.0, 9.0, 9.0), (0.0, 0.0, 0.0, 1.0)), {"arm_hinge": 0.0})
    assert moved.position == pytest.approx((0.0, 0.0, 0.2), abs=1e-9)  # the MJCF placement, unmoved
    assert subbody.link_pose("fixed_arm", None, {"arm_hinge": 0.0}).position == pytest.approx((0.0, 0.0, 0.2), abs=1e-9)


def _joint_states_message(stamp: float, names: list, positions: list) -> SimpleNamespace:
    """One JointState as the MuJoCo LINK stream's callback sees it."""
    return SimpleNamespace(header=_stamp_namespace(stamp), name=names, position=positions)


def _stamp_namespace(stamp: float) -> SimpleNamespace:
    sec = int(stamp)
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9))))


def _odometry_message(stamp: float, x: float) -> SimpleNamespace:
    """One floating_base_state Odometry as the link stream's callback sees it."""
    return SimpleNamespace(
        header=_stamp_namespace(stamp),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
    )


def _free_joint_states_message(stamp: float, x: float) -> SimpleNamespace:
    """One FreeJointStateArray as the MuJoCo entity stream's callback sees it."""
    return SimpleNamespace(
        header=_stamp_namespace(stamp),
        free_joints=[
            SimpleNamespace(
                name="cube",
                pose=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=x, y=0.0, z=0.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ),
            )
        ],
    )


def test_the_free_joint_stream_restarts_its_history_on_a_rewound_clock():
    stream = open_entity_ground_truth("mujoco", world="demo", scene="s", entity="cube", env={})
    assert stream.reset_detected() is False  # a fresh stream has seen no reset
    stream._on_free_joint_states(_free_joint_states_message(10.0, 1.0))
    stream._on_free_joint_states(_free_joint_states_message(10.04, 1.2))
    assert [s.stamp_sec for s in stream.samples()] == [10.0, 10.04]
    assert stream.reset_detected() is False
    # the reset: the clock and the pose jump back — the history restarts AND
    # the reset is recorded (review r1 item 8: the consumer fails the run
    # explicitly instead of trusting the post-reset span)
    stream._on_free_joint_states(_free_joint_states_message(0.5, -1.0))
    assert [s.stamp_sec for s in stream.samples()] == [0.5]
    assert stream.reset_detected() is True
    stream._on_free_joint_states(_free_joint_states_message(0.54, -0.8))
    assert [s.stamp_sec for s in stream.samples()] == [0.5, 0.54]


def test_the_link_stream_pairs_the_base_and_restarts_on_a_rewind(tmp_path):
    """Item 17 at the stream level: `tool` (under a free-jointed body that
    is not joint 0) requires the base pairing and resolves it; `fixed_arm`
    (no free ancestor) samples with NO base message at all; a rewinding
    stamp restarts the history."""
    pytest.importorskip("mujoco")
    from robot_runtime_sim.testing.mujoco_link_ground_truth import MujocoLinkGroundTruthStream

    path = tmp_path / "subbody.xml"
    path.write_text(SUBBODY_MODEL, encoding="utf-8")
    tool = MujocoLinkGroundTruthStream(str(path), "m", "tool", env={})
    assert tool._free_joint_qposadr is not None
    assert tool.reset_detected() is False
    tool._on_joints(_joint_states_message(10.0, ["arm_hinge"], [0.0]))  # no base sample yet
    assert tool.samples() == []
    tool._on_base(_odometry_message(10.0, 1.0))
    tool._on_joints(_joint_states_message(10.0, ["arm_hinge"], [0.0]))
    assert [s.stamp_sec for s in tool.samples()] == [10.0]
    assert tool.samples()[0].position == pytest.approx((1.05, 0.0, 0.0), abs=1e-9)
    # the rewind: a new epoch — the history restarts and the reset is recorded
    tool._on_base(_odometry_message(0.5, -1.0))
    tool._on_joints(_joint_states_message(0.5, ["arm_hinge"], [0.0]))
    assert [s.stamp_sec for s in tool.samples()] == [0.5]
    assert tool.samples()[0].position == pytest.approx((-0.95, 0.0, 0.0), abs=1e-9)
    assert tool.reset_detected() is True

    fixed_arm = MujocoLinkGroundTruthStream(str(path), "m", "fixed_arm", env={})
    assert fixed_arm._free_joint_qposadr is None
    fixed_arm._on_joints(_joint_states_message(20.0, ["arm_hinge"], [0.0]))  # no base needed
    assert [s.stamp_sec for s in fixed_arm.samples()] == [20.0]
    assert fixed_arm.samples()[0].position == pytest.approx((0.0, 0.0, 0.2), abs=1e-9)


def test_the_floating_base_stream_filters_by_child_frame_and_restarts():
    """The MuJoCo robot BASE is read from /simulator/floating_base_state
    (the robot is not a scene dynamic object, so free_joint_states never
    carries it): messages of other bodies are ignored via child_frame_id,
    and a rewound stamp restarts the history."""
    from types import SimpleNamespace as NS

    from robot_runtime_sim.testing.entity_ground_truth import MujocoFloatingBaseGroundTruthStream

    stream = MujocoFloatingBaseGroundTruthStream("base_link", env={})
    odom = NS(
        child_frame_id="base_link",
        header=_stamp_namespace(10.0),
        pose=NS(pose=NS(position=NS(x=1.0, y=2.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
    )
    stream._on_odometry(odom)
    stream._on_odometry(NS(child_frame_id="other_body", header=_stamp_namespace(10.04), pose=odom.pose))
    assert [s.stamp_sec for s in stream.samples()] == [10.0]
    assert stream.latest().position == pytest.approx((1.0, 2.0, 0.0))
    stream._on_odometry(
        NS(
            child_frame_id="base_link",
            header=_stamp_namespace(0.5),
            pose=NS(pose=NS(position=NS(x=-1.0, y=2.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
        )
    )
    assert [s.stamp_sec for s in stream.samples()] == [0.5]  # the rewind restarts
    assert stream.reset_detected() is True  # ... and is recorded (review r1 item 8)
