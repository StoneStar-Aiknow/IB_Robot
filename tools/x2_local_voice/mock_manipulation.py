"""Local-only grasp/place backends: the real pipelines run, only perception and motion are fakes.

GraspPipelineNode / PlacePipelineNode, held custody, the Gateway and the Agent are the
production code. These factories replace what needs the robot: cameras, models, the arm
and the hand. Each step sleeps so progress speech has time to play, as it would on X2.

Behaviour can be changed per request with LOCAL_VOICE_PICK / LOCAL_VOICE_PLACE in the
environment of the launch: ``success`` (default) or ``empty`` (pick closes on nothing) /
``unseen`` (place cannot verify the object in the container).

The fakes are the manipulation_execution unit-test doubles, imported from the checkout's
test directory, so they follow the pipeline contracts they are tested against.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

_TESTS = Path(os.environ["WORKSPACE"]) / "src/manipulation_execution/test"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_grasp_pipeline import HOME, JOINTS, Effector, Executor, Proposer, Source, StreamingPlanner  # noqa: E402
from test_place_pipeline import fixture  # noqa: E402

from manipulation_execution.grasp_pipeline.contracts import GraspBackends, HeldObject, PlaceVerdict  # noqa: E402

FRAME_S = 0.5  # one fake trajectory frame (pick ~20 s, place ~12 s)
PERCEIVE_S = 3.0  # one fake detection / proposal
EFFECT_S = 1.0  # one fake hand action


class _SlowSource(Source):
    def capture(self, *, deadline, cancelled):
        time.sleep(0.3)
        return super().capture(deadline=deadline, cancelled=cancelled)


class _SlowProposer(Proposer):
    def propose(self, observation, query, *, deadline, cancelled):
        time.sleep(PERCEIVE_S)
        return super().propose(observation, query, deadline=deadline, cancelled=cancelled)


class _SlowExecutor(Executor):
    def execute(self, segment, *, carrying, halted, cancelled):
        frames = len(segment.positions)
        end = time.monotonic() + frames * FRAME_S
        while time.monotonic() < end:
            if cancelled():
                break
            time.sleep(0.02)
        return super().execute(segment, carrying=carrying, halted=halted, cancelled=cancelled)

    def hold(self, seconds, *, carrying, cancelled):
        end = time.monotonic() + min(float(seconds), 2.0)
        while time.monotonic() < end and not cancelled():
            time.sleep(0.02)
        return super().hold(seconds, carrying=carrying, cancelled=cancelled)

    def finish(self):
        # Custody handed the arm to placement, which ends with the arm back home.
        self.handed_over = True

    def measured(self, joint_names):
        return HOME.copy() if getattr(self, "handed_over", False) else super().measured(joint_names)


class _SlowEffector(Effector):
    def act(self, action, *, halted, cancelled):
        time.sleep(EFFECT_S)
        return super().act(action, halted=halted, cancelled=cancelled)

    def suspend(self):
        self.gripped = False

    def released(self):
        return not self.gripped


def grasp_backends(*, node, grasp_execution, description):
    scenario = os.environ.get("LOCAL_VOICE_PICK", "success")
    effector = _SlowEffector(held=[scenario != "empty"])
    node.get_logger().warning(f"LOCAL VOICE MOCK: grasp perception and motion are fakes (scenario={scenario})")

    def held_object(target, observation, proposals, plan):
        # The two fake executors live in different processes, unlike the one real arm. The handover
        # pose is the place executor's start (the fixture's transfer begins at zero).
        handover = (0.0, 0.0, 0.0)
        return HeldObject(
            f"local-{uuid.uuid4().hex[:8]}", target, "right", JOINTS, handover, (0.0,) * 3, 0.04, (0.0,) * 3, "{}"
        )

    return GraspBackends(
        _SlowSource(), _SlowProposer(), StreamingPlanner(), _SlowExecutor(), effector, held_object=held_object
    )


def place_backends(*, node, placement_execution, grasp_execution, description):
    scenario = os.environ.get("LOCAL_VOICE_PLACE", "success")
    verdict = PlaceVerdict(False, "PLACE_UNSEEN") if scenario == "unseen" else None
    _held, backends = fixture(verdict=verdict)
    slow = _SlowExecutor()
    slow.position = backends.executor.position
    proposer, planner, verifier = backends.proposer, backends.planner, backends.verifier

    class _Perceive:
        def propose(self, *args, **kwargs):
            time.sleep(PERCEIVE_S)
            return proposer.propose(*args, **kwargs)

    class _Verify:
        def verify(self, *args, **kwargs):
            time.sleep(PERCEIVE_S)
            return verifier.verify(*args, **kwargs)

    node.get_logger().warning(f"LOCAL VOICE MOCK: place perception and motion are fakes (scenario={scenario})")
    return replace(
        backends, observation=_SlowSource(), proposer=_Perceive(), planner=planner, verifier=_Verify(), executor=slow
    )
