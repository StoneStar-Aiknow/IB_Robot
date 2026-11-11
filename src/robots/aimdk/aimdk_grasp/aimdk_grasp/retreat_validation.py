"""Read-only X2 retreat validation against a newly captured scene."""

import time
from dataclasses import replace

import numpy as np

from manipulation_execution.grasp_pipeline.contracts import Effect, EffectorAction, GraspFailure, Move, SegmentKind

from .arm_gravity import require_executable
from .hand import is_open
from .joints import arm_joints
from .revalidation import require_fresh, require_posture


def _carried_scene(prepared, observation, model, feedback, held, config):
    """Use placement's hand/object envelope for self filtering, retaining all other obstacles."""
    from .container_perception import cover_reach
    from .place_request import placement_cover

    attached = placement_cover(model, feedback, held)
    palm = model.last_link_frames[0, model.link_indices["R_palm"]]
    centers = attached[:, :3] @ palm[:3, :3].T + palm[:3, 3]
    fine, coarse = config.validation_self_filter_margin_m, config.self_filter_margin_m
    reach = cover_reach(prepared.points, observation.base_from_camera, centers, attached[:, 3], max(fine, coarse))
    return replace(
        prepared,
        keep=prepared.keep & (reach > fine),
        planning_all=prepared.planning_all & (reach > coarse),
        spheres=np.vstack([prepared.spheres, np.column_stack([centers, attached[:, 3]])]),
    ), attached


def _require_support(scene, reference, model, feedback, held, contact):
    """A return-to-source release requires the same observed support under the original object."""
    from scipy.spatial import ConvexHull

    from .scene_estimation import plane_basis

    current = scene.support
    if (
        reference is None
        or np.dot(reference.normal, current.normal) < 0.99
        or abs(np.dot(current.point - reference.point, reference.normal)) > 0.01
    ):
        raise ValueError("original release support moved or is unavailable")
    model.spheres(feedback, np.asarray(contact)[None])
    palm = model.last_link_frames[0, model.link_indices["R_palm"]]
    points = np.asarray(held.points_in_palm).reshape(-1, 3) @ palm[:3, :3].T + palm[:3, 3]
    basis = plane_basis(current.normal)[:, :2]
    hull = ConvexHull((current.inliers - current.point) @ basis)
    projected = (np.median(points, axis=0) - current.point) @ basis
    if np.max(hull.equations[:, :2] @ projected + hull.equations[:, 2]) > 0.01:
        raise ValueError("original object position is no longer over observed support")


def validate_retreat(head, robot, steps, hand_positions, *, deadline, cancelled, held=None, support=None, contact=None):
    def check():
        if cancelled() or time.monotonic() >= deadline:
            raise GraspFailure("PICK_MANUAL_CHECK", "retreat validation interrupted or timed out")

    check()
    side, config = robot.side, robot.planning.revalidation
    release = held is not None
    moves = steps[:-1] if release else steps
    if (
        not steps
        or len(steps) > 32
        or any(not isinstance(step, Move) or step.refine_after for step in moves)
        or (release and steps[-1] != Effect(EffectorAction.RELEASE))
    ):
        raise ValueError("retreat accepts only bounded arm moves")
    hand = np.asarray(hand_positions, float)
    if hand.shape != (10,) or not np.isfinite(hand).all():
        raise ValueError("finite measured hand pose required")
    if not release and not is_open(hand, robot.hand.preset("open")):
        raise ValueError("automatic retreat requires an unchanged open hand")
    observation = head.camera.capture(deadline=deadline, cancelled=cancelled)
    live = observation.robot_state.live
    require_fresh(live, head.camera.clock(), config)
    feedback = live.feedback
    require_posture(live.observation.image_feedback, feedback, config, side=side)
    if np.max(np.abs(feedback.hand(side) - hand)) > config.start_tolerance_rad:
        raise ValueError("hand differs from the interrupted snapshot")
    previous = feedback.vector(arm_joints(side))
    if release and (
        held.arm_side != side
        or tuple(held.joint_names) != arm_joints(side)
        or np.max(np.abs(previous - held.joint_positions)) > 0.005
        or contact is None
        or np.asarray(contact).shape != (7,)
        or not np.isfinite(contact).all()
    ):
        raise ValueError("held geometry does not match this arm and contact")
    paths = []
    for step in moves:
        segment = step.segment
        if (
            segment.kind is not SegmentKind.TRAJECTORY
            or segment.joint_names != arm_joints(side)
            or len(segment.positions) > 10000
            or np.max(np.abs(segment.start - previous)) > 0.005
        ):
            raise ValueError("retreat start or segment continuity differs from measured arm")
        count = np.ceil(np.max(np.abs(np.diff(segment.positions, axis=0)), axis=1) / 0.01).astype(int)
        if int(count.sum()) + 1 + sum(map(len, paths)) > 10000:
            raise ValueError("retreat validation exceeds sample budget")
        dense = [segment.start]
        for a, b, n in zip(segment.positions[:-1], segment.positions[1:], count, strict=True):
            dense.extend(np.linspace(a, b, max(1, int(n)) + 1)[1:])
        paths.append(np.asarray(dense))
        previous = segment.end
    if sum(map(len, paths)) > 10000:
        raise ValueError("retreat validation exceeds sample budget")
    if release and np.max(np.abs(previous - contact)) > 0.02:
        raise ValueError("release must be at the original grasp contact")

    def source_fresh():
        now = head.camera.clock()
        age = now - live.observation.received_at
        if not np.isfinite(age) or not 0 <= age <= config.scene_max_age_s:
            raise ValueError("retreat source image expired during validation")

    from .head_camera import gpu_section
    from .proposer import prepare_capture_scene
    from .scene_estimation import complete_scene

    with gpu_section(head.gpu, deadline=deadline, cancelled=cancelled, code="PICK_MANUAL_CHECK"):
        check()
        source_fresh()
        prepared = prepare_capture_scene(observation, head.spheres, robot.perception.scene)
        attached = None
        if release:
            prepared, attached = _carried_scene(
                prepared, observation, head.spheres, feedback, held, robot.perception.scene
            )
        scene = complete_scene(prepared, None, lazy_validation=False)
        if release:
            _require_support(scene, support, head.spheres, feedback, held, contact)
        source_fresh()
        latest = head.camera.capture_state(deadline=deadline, cancelled=cancelled)
        require_fresh(latest, head.camera.clock(), config)
        require_posture(feedback, latest.feedback, config, side=side)
        from .place_request import _hand_cover, opening_cover
        from .planner import ATTACH_SLOTS

        if not release:
            attached = _hand_cover(head.spheres, feedback, side, feedback.hand(side), ATTACH_SLOTS)
            attached = attached if len(attached) else None
        with head.service.session(feedback, scene, attached=attached) as planner:
            check()
            source_fresh()
            for path in paths:
                if not np.asarray(planner.valid_positions(path)).all():
                    raise ValueError("retreat collision or joint constraint failed")
                check()
            require_executable({str(i): p for i, p in enumerate(paths)}, robot.limits, robot.gravity)
        if release:
            opening = opening_cover(head.spheres, feedback, held)
            with head.service.session(feedback, scene, attached=opening if len(opening) else None) as planner:
                if not np.asarray(planner.valid_positions(np.asarray(previous)[None])).all():
                    raise ValueError("opening at original contact is obstructed")
            source_fresh()
            check()
    current = head.camera.capture_state(deadline=deadline)
    require_fresh(current, head.camera.clock(), config)
    # Check the source scene age too, not merely the new feedback's image.
    source_fresh()
    require_posture(feedback, current.feedback, config, side=side)
    check()
    return {"image_stamp_ns": observation.stamp_ns, "checked_samples": sum(map(len, paths)), "arm_side": side}
