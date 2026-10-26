import threading
import time

import numpy as np
import pytest
from aimdk_grasp.target_watch import TargetWatch, WatchConfig, robot_pixel_mask, target_change
from aimdk_grasp.types import CameraIntrinsics

K = CameraIntrinsics(100.0, 100.0, 50.0, 40.0)
SHAPE = (80, 100)


def scene(target_at=(30, 40), seed=0):
    """Textured background at 1 m, a 12x12 textured box at 0.6 m."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
    depth = np.full(SHAPE, 1.0, np.float32)
    box = np.random.default_rng(99).integers(0, 256, (12, 12, 3), dtype=np.uint8)
    r, c = target_at
    rgb[r : r + 12, c : c + 12] = box
    depth[r : r + 12, c : c + 12] = 0.6
    mask = np.zeros(SHAPE, bool)
    mask[r : r + 12, c : c + 12] = True
    return rgb, depth, mask


NONE = np.zeros(SHAPE, bool)


def test_robot_pixel_mask_projects_spheres_and_covers_everything_at_the_camera():
    # Camera frame = torso frame. A 5 cm sphere at 1 m on the optical axis.
    covered = robot_pixel_mask([[0.0, 0.0, 1.0, 0.05]], np.eye(4), K, SHAPE)
    # Its exact projection extends 5.006 px from the centre; the conservative
    # bounds include every pixel there and at most one more.
    assert covered[40, 50] and covered[40, 55] and covered[35, 50] and not covered[40, 57] and not covered[0, 0]
    assert not robot_pixel_mask([[0.0, 0.0, -1.0, 0.05]], np.eye(4), K, SHAPE).any()
    # Crossing the camera plane in front of the lens covers everything; beside
    # the camera (the elbow passing the head, 30 cm to the side) nothing.
    assert robot_pixel_mask([[0.0, 0.0, 0.01, 0.05]], np.eye(4), K, SHAPE).all()
    assert not robot_pixel_mask([[0.3, 0.0, 0.01, 0.05]], np.eye(4), K, SHAPE).any()
    near_right = robot_pixel_mask([[0.03, 0.0, 0.03, 0.05]], np.eye(4), K, SHAPE)  # reaches in from the right
    assert near_right[:, -1].all() and not near_right[:, 0].any()
    # The torso pose is applied: camera 1 m along +x of the torso.
    torso_from_camera = np.eye(4)
    torso_from_camera[0, 3] = 1.0
    shifted = robot_pixel_mask([[1.0, 0.0, 1.0, 0.05]], torso_from_camera, K, SHAPE)
    assert shifted[40, 50]
    assert robot_pixel_mask([[0.0, 0.0, 1.0, 0.05]], np.eye(4), K, SHAPE, margin_m=0.05)[40, 59]


def test_robot_pixel_mask_covers_an_off_axis_sphere_projection():
    intrinsics = CameraIntrinsics(600.0, 600.0, 640.0, 360.0)
    shape = (720, 1280)
    centre = np.array([0.15, 0.0, 0.30])
    radius = 0.10
    pixel = (1210, 360)
    ray = np.array([(pixel[0] - intrinsics.cx) / intrinsics.fx, (pixel[1] - intrinsics.cy) / intrinsics.fy, 1.0])
    ray_distance = np.linalg.norm(np.cross(centre, ray)) / np.linalg.norm(ray)
    assert ray_distance < radius

    covered = robot_pixel_mask([[*centre, radius]], np.eye(4), intrinsics, shape, margin_m=0.02)
    assert covered[pixel[1], pixel[0]]


def test_a_still_target_votes_still_and_a_moved_one_votes_moved():
    rgb, depth, mask = scene()
    still = target_change(rgb, depth, rgb, depth, mask, NONE, NONE)
    assert still["vote"] == "still" and still["shift_px"] == [0, 0] and still["visible_fraction"] == 1.0
    moved_rgb, moved_depth, _ = scene(target_at=(30, 55))
    assert target_change(rgb, depth, moved_rgb, moved_depth, mask, NONE, NONE)["vote"] == "moved"


def test_camera_sway_within_the_shift_range_is_not_a_move():
    rgb, depth, mask = scene()
    swayed_rgb, swayed_depth = np.roll(rgb, (2, -3), axis=(0, 1)), np.roll(depth, (2, -3), axis=(0, 1))
    result = target_change(rgb, depth, swayed_rgb, swayed_depth, mask, NONE, NONE)
    assert result["vote"] == "still" and result["shift_px"] == [-3, 2]
    no_shift = target_change(rgb, depth, swayed_rgb, swayed_depth, mask, NONE, NONE, WatchConfig(shift_px=0))
    assert no_shift["vote"] == "moved"


def test_the_arm_in_front_of_the_target_is_excluded_not_counted_as_a_move():
    rgb, depth, mask = scene()
    arm_rgb, arm_depth = rgb.copy(), depth.copy()
    arm_rgb[25:34, 30:60], arm_depth[25:34, 30:60] = 0, 0.3  # hand covers the upper part
    arm = np.zeros(SHAPE, bool)
    arm[24:35, 29:61] = True
    assert target_change(rgb, depth, arm_rgb, arm_depth, mask, NONE, NONE)["vote"] == "moved"
    result = target_change(rgb, depth, arm_rgb, arm_depth, mask, NONE, arm)
    assert result["vote"] == "still" and result["visible_fraction"] < 1.0
    # Covering (nearly) all of it: unknown, never moved.
    everything = np.ones(SHAPE, bool)
    assert target_change(rgb, depth, arm_rgb, arm_depth, mask, NONE, everything)["vote"] == "unknown"


def test_moved_needs_enough_votes_in_the_window_and_unknown_frames_do_not_count():
    rgb, depth, mask = scene()
    moved_rgb, moved_depth, _ = scene(target_at=(30, 55))
    watch = TargetWatch(rgb, depth, mask, NONE, WatchConfig(moved_frames=3, window_frames=4))
    frames = [
        (moved_rgb, moved_depth, NONE),
        (rgb, depth, NONE),
        (moved_rgb, moved_depth, NONE),
        (moved_rgb, moved_depth, np.ones(SHAPE, bool)),
    ]
    assert [watch.update(*frame)["moved"] for frame in frames] == [False] * 4
    assert list(watch.votes) == ["moved", "still", "moved", "unknown"]
    assert not watch.update(moved_rgb, moved_depth, NONE)["moved"]  # still, moved, unknown, moved
    assert watch.update(moved_rgb, moved_depth, NONE)["moved"]  # moved, unknown, moved, moved
    assert watch.update(rgb, depth, NONE)["moved"]  # the decision latches


def test_invalid_configuration_and_mask_are_rejected():
    with pytest.raises(ValueError):
        WatchConfig(moved_frames=5, window_frames=4)
    rgb, depth, _ = scene()
    with pytest.raises(ValueError):
        target_change(rgb, depth, rgb, depth, NONE, NONE, NONE)


def watcher_inputs(frames, phases):
    """A planned request and a capture() that returns the given frames, phases advancing per capture."""
    from types import SimpleNamespace

    rgb, depth, mask = scene()
    healthy = SimpleNamespace(healthy=True)
    transform = SimpleNamespace(base_from_camera=np.eye(4), stamp_ns=10)
    request = SimpleNamespace(
        rgb=rgb, depth=depth, mask=mask, intrinsics=K, transform=transform, image_feedback=healthy, feedback=healthy
    )
    state = dict(index=0, phase=phases[0])

    def capture():
        index = state["index"]
        frame_rgb, frame_depth, feedback = frames[min(index, len(frames) - 1)]
        state["index"] = index + 1
        state["phase"] = phases[min(index + 1, len(phases) - 1)]
        return SimpleNamespace(
            observation=SimpleNamespace(
                frame=SimpleNamespace(rgb=frame_rgb, depth=frame_depth, intrinsics=K),
                transform=SimpleNamespace(base_from_camera=np.eye(4), stamp_ns=11 + index),
                image_feedback=feedback,
                received_at=0.0,
            )
        )

    return request, capture, lambda: state["phase"]


def far_spheres(feedback):
    return np.array([[0.0, 0.0, -1.0, 0.05]])  # behind the camera: covers nothing


@pytest.fixture(autouse=True)
def one_moving_sphere(monkeypatch):
    monkeypatch.setattr("aimdk_grasp.target_watch.moving_sphere_indices", lambda: np.array([0]))


def test_only_the_left_arm_below_the_shoulder_and_the_hand_are_masked(monkeypatch):
    monkeypatch.undo()  # the real sphere model
    from aimdk_grasp.robot_spheres import RobotSphereModel
    from aimdk_grasp.target_watch import moving_sphere_indices

    links = {RobotSphereModel().sphere_links[i] for i in moving_sphere_indices()}
    assert "R_palm" in links and "left_wrist_roll_link" in links and "left_elbow_link" in links
    assert not links & {"torso_link", "head_pitch_link", "left_shoulder_roll_link", "right_elbow_link"}


def test_execution_watcher_votes_during_arm_segments_and_reports_moved_once():
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher

    rgb, depth, _ = scene()
    moved_rgb, moved_depth, _ = scene(target_at=(30, 55))
    ok, bad = SimpleNamespace(healthy=True), SimpleNamespace(healthy=False)
    frames = [(rgb, depth, ok), (moved_rgb, moved_depth, bad)] + [(moved_rgb, moved_depth, ok)] * 4
    phases = ["approach"] * 4 + ["grasp"] * 3 + ["close"]
    request, capture, phase = watcher_inputs(frames, phases)
    calls = []
    watcher = ExecutionWatcher(
        request, capture, far_spheres, phase, on_moved=calls.append, keep_frames=True, clock=lambda: 0.0
    )
    watcher._run()  # synchronously: ends at the first phase after the watched ones
    report = watcher.report()
    assert watcher.error is None and report["frames"] == 7 and report["skipped"] == 1
    assert [r.get("vote") for r in report["records"]] == ["still", None] + ["moved"] * 5
    assert report["moved"] and len(calls) == 1 and calls[0] is report["moved_at"]
    assert report["moved_at"]["stamp_ns"] == 15  # third moved vote
    assert len(watcher.frames) == 1 + 6  # planned image + every voted frame
    assert watcher.frames[0]["rgb"].shape == (watcher.crop[1] - watcher.crop[0], watcher.crop[3] - watcher.crop[2], 3)


def test_execution_watcher_does_not_notify_after_capture_enters_close():
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher

    moved_rgb, moved_depth, _ = scene(target_at=(30, 55))
    request, capture, phase = watcher_inputs(
        [(moved_rgb, moved_depth, SimpleNamespace(healthy=True))], ["approach", "close"]
    )
    calls = []
    watcher = ExecutionWatcher(
        request,
        capture,
        far_spheres,
        phase,
        config=WatchConfig(moved_frames=1, window_frames=1),
        on_moved=calls.append,
        clock=lambda: 0.0,
    )

    watcher._run()

    assert watcher.watch.moved and watcher.notified is None and calls == []


def test_execution_watcher_does_not_notify_after_stop_during_capture():
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher

    moved_rgb, moved_depth, _ = scene(target_at=(30, 55))
    request, capture, phase = watcher_inputs([(moved_rgb, moved_depth, SimpleNamespace(healthy=True))], ["approach"])
    entered, release = threading.Event(), threading.Event()

    def blocked_capture():
        entered.set()
        assert release.wait(1.0)
        return capture()

    calls = []
    watcher = ExecutionWatcher(
        request,
        blocked_capture,
        far_spheres,
        phase,
        config=WatchConfig(moved_frames=1, window_frames=1),
        on_moved=calls.append,
        clock=lambda: 0.0,
    ).start()
    assert entered.wait(1.0)
    stopper = threading.Thread(target=watcher.stop)
    stopper.start()
    assert watcher._stop.wait(1.0)
    release.set()
    stopper.join(2.0)

    assert not stopper.is_alive() and not watcher._thread.is_alive()
    assert watcher.watch.moved and watcher.notified is None and calls == []


def test_a_record_ages_its_frame_on_the_clock_the_frame_is_stamped_on():
    """Frames carry source (PTP) seconds; the watch's own clock is monotonic."""
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher

    rgb, depth, _ = scene()
    ok = SimpleNamespace(healthy=True)
    request, capture, phase = watcher_inputs([(rgb, depth, ok)] * 2, ["approach"] * 2 + ["close"])
    watcher = ExecutionWatcher(request, capture, far_spheres, phase, clock=lambda: 5.0, source_clock=lambda: 0.25)
    watcher._run()
    assert [record["age_s"] for record in watcher.records] == [0.25, 0.25]


def test_execution_watcher_waits_for_the_first_segment_and_records_its_own_failure():
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher

    rgb, depth, _ = scene()
    request, capture, _ = watcher_inputs([(rgb, depth, SimpleNamespace(healthy=True))], ["approach"])
    # SOC0 refused before any segment: nothing is captured until stop().
    watcher = ExecutionWatcher(request, capture, far_spheres, lambda: "").start()
    time.sleep(0.1)
    watcher.stop()
    assert watcher.records == [] and not watcher._thread.is_alive()

    def broken():
        raise RuntimeError("capture service down")

    watcher = ExecutionWatcher(request, broken, far_spheres, lambda: "approach")
    watcher._run()
    assert watcher.error == "RuntimeError: capture service down" and not watcher.report()["moved"]


def test_orientation_buffer_interpolates_between_close_samples_only():
    from aimdk_grasp.target_watch import OrientationBuffer
    from scipy.spatial.transform import Rotation

    buffer = OrientationBuffer(seconds=1.0, max_gap_s=0.1)

    def turn(degrees):
        return Rotation.from_euler("z", degrees, degrees=True).as_quat()

    for stamp, degrees in ((0, 0.0), (50_000_000, 10.0), (300_000_000, 20.0)):
        buffer.add(stamp, turn(degrees))
    buffer.add(40_000_000, turn(99.0))  # out of order: ignored
    buffer.add(350_000_000, [0.0, 0.0, 0.0, 2.0])  # not a unit quaternion: ignored
    middle = Rotation.from_quat(buffer.at(25_000_000)).as_euler("zyx", degrees=True)[0]
    assert middle == pytest.approx(5.0)
    assert buffer.at(60_000_000) is None  # 240 ms to the next sample
    assert buffer.at(-1) is None and buffer.at(400_000_000) is None  # outside the samples
    buffer.add(2_000_000_000, turn(0.0))
    assert len(buffer._rows) == 1  # older than `seconds` dropped


def test_camera_shift_is_the_image_motion_of_a_still_point_when_the_body_turns():
    from aimdk_grasp.target_watch import camera_shift
    from scipy.spatial.transform import Rotation

    still = Rotation.identity().as_quat()
    # torso x forward, y left, z up; optical frame z forward, x right, y down.
    torso_from_camera = np.eye(4)
    torso_from_camera[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert camera_shift(still, still, torso_from_camera, K, (50.0, 40.0)) == pytest.approx((0.0, 0.0))
    # The body yaws 1 deg to the left: a still point moves right in the image.
    left = Rotation.from_euler("z", 1.0, degrees=True).as_quat()
    du, dv = camera_shift(still, left, torso_from_camera, K, (50.0, 40.0))
    assert du == pytest.approx(K.fx * np.tan(np.radians(1.0)), abs=1e-6) and dv == pytest.approx(0.0, abs=1e-6)
    # Pitching down (nose-down about torso y): the point moves up... i.e. to smaller v.
    down = Rotation.from_euler("y", 1.0, degrees=True).as_quat()
    du, dv = camera_shift(still, down, torso_from_camera, K, (50.0, 40.0))
    assert dv == pytest.approx(-K.fy * np.tan(np.radians(1.0)), abs=1e-6) and du == pytest.approx(0.0, abs=1e-6)
    # Only the change since the reference counts.
    assert camera_shift(left, left, torso_from_camera, K, (50.0, 40.0)) == pytest.approx((0.0, 0.0))


def test_a_predicted_shift_is_taken_out_and_only_a_small_search_remains():
    rgb, depth, mask = scene()
    swayed_rgb, swayed_depth = np.roll(rgb, (0, 14), axis=(0, 1)), np.roll(depth, (0, 14), axis=(0, 1))
    config = WatchConfig(shift_px=3, imu_shift_px=2)
    assert target_change(rgb, depth, swayed_rgb, swayed_depth, mask, NONE, NONE, config)["vote"] == "moved"
    result = target_change(rgb, depth, swayed_rgb, swayed_depth, mask, NONE, NONE, config, predicted_shift=(13.2, 0.3))
    assert result["vote"] == "still" and result["shift_px"] == [14, 0]
    # A prediction that is off by more than the remaining search does not hide a real difference.
    assert (
        target_change(rgb, depth, swayed_rgb, swayed_depth, mask, NONE, NONE, config, predicted_shift=(5.0, 0.0))[
            "vote"
        ]
        == "moved"
    )
    # No room for the shifted region in the image: unknown, not moved.
    assert (
        target_change(rgb, depth, swayed_rgb, swayed_depth, mask, NONE, NONE, config, predicted_shift=(60.0, 0.0))[
            "vote"
        ]
        == "unknown"
    )


def test_execution_watcher_takes_the_imu_predicted_shift_out_of_every_frame():
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher, camera_shift
    from scipy.spatial.transform import Rotation

    rgb, depth, _ = scene()
    ok = SimpleNamespace(healthy=True)

    # The body turns a little more in every frame; the image moves as predicted.
    def quaternion(stamp):
        return Rotation.from_euler("y", -1.2 * max(stamp - 10, 0), degrees=True).as_quat()

    shifts = [round(camera_shift(quaternion(10), quaternion(11 + i), np.eye(4), K, (45.5, 35.5))[0]) for i in range(5)]
    assert shifts[0] and shifts[-1] >= 10
    frames = [(np.roll(rgb, shift, axis=1), np.roll(depth, shift, axis=1), ok) for shift in shifts]
    request, capture, phase = watcher_inputs(frames, ["approach"] * 5 + ["close"])
    # The IMU alone: no background measurement (see the background test).
    config = WatchConfig(shift_px=3, imu_shift_px=1, background_patch_px=None)
    watcher = ExecutionWatcher(
        request, capture, far_spheres, phase, config=config, keep_frames=True, clock=lambda: 0.0, orientation=quaternion
    )
    watcher._run()
    report = watcher.report()
    assert report["imu"] and report["votes"] == {"still": 5, "moved": 0, "unknown": 0}
    assert [record["shift_px"][0] for record in report["records"]] == shifts
    assert [record["predicted_shift_px"][0] for record in report["records"]] == pytest.approx(
        [camera_shift(quaternion(10), quaternion(11 + i), np.eye(4), K, (45.5, 35.5))[0] for i in range(5)], abs=0.05
    )
    assert all(np.isfinite(frame["imu_quaternion"]).all() for frame in watcher.frames)
    # Without the IMU the same frames read moved once the shift passes shift_px.
    request, capture, phase = watcher_inputs(frames, ["approach"] * 5 + ["close"])
    blind = ExecutionWatcher(request, capture, far_spheres, phase, config=config, clock=lambda: 0.0)
    blind._run()
    assert blind.report()["moved"] and not blind.report()["imu"]
    # An IMU without a sample at an image time: that frame is compared without it.
    request, capture, phase = watcher_inputs(frames[:1], ["approach", "close"])
    gap = ExecutionWatcher(
        request,
        capture,
        far_spheres,
        phase,
        config=config,
        clock=lambda: 0.0,
        orientation=lambda stamp: quaternion(stamp) if stamp == 10 else None,
    )
    gap._run()
    assert "predicted_shift_px" not in gap.records[0]
    # Required (stop mode): a frame without an orientation is never voted, whatever it shows.
    request, capture, phase = watcher_inputs(frames, ["approach"] * 5 + ["close"])
    strict = ExecutionWatcher(
        request,
        capture,
        far_spheres,
        phase,
        config=config,
        keep_frames=True,
        clock=lambda: 0.0,
        orientation=lambda stamp: quaternion(stamp) if stamp <= 12 else None,
        require_orientation=True,
    )
    strict._run()
    report = strict.report()
    assert [record["vote"] for record in report["records"]] == ["still", "still"] + ["unknown"] * 3
    assert report["records"][-1]["unknown"] == "no camera shift (background or IMU)" and not report["moved"]
    assert len(strict.frames) == 1 + 5  # still kept for replay
    request, capture, phase = watcher_inputs(frames, ["approach"] * 5 + ["close"])
    no_imu = ExecutionWatcher(
        request, capture, far_spheres, phase, config=config, clock=lambda: 0.0, require_orientation=True
    )
    no_imu._run()
    assert no_imu.report()["votes"] == {"still": 0, "moved": 0, "unknown": 5}


def test_the_background_measures_the_camera_shift_the_imu_misses():
    # Live 2026-10-15 p_stop st6: the whole image drifted 36 px as the arm
    # reached, the IMU (rotation only) predicted 16, and the still bottle read moved.
    from types import SimpleNamespace

    from aimdk_grasp.target_watch import ExecutionWatcher, background_shift

    rgb, depth, mask = scene()
    config = WatchConfig(shift_px=3, imu_shift_px=2, background_search_px=16)
    watch = TargetWatch(rgb, depth, mask, NONE, config)
    assert len(watch.corners) >= config.background_min_patches
    swayed_rgb, swayed_depth = np.roll(rgb, (2, 14), axis=(0, 1)), np.roll(depth, (2, 14), axis=(0, 1))
    # The IMU predicted half of it: still, by the background's 14, 2.
    result = watch.update(swayed_rgb, swayed_depth, NONE, predicted_shift=(7.0, 1.0))
    assert result["vote"] == "still" and result["background_shift_px"] == [14.0, 2.0]
    assert result["shift_px"] == [14, 2] and result["predicted_shift_px"] == [7.0, 1.0]
    # Without the background the same frame reads moved.
    blind = TargetWatch(rgb, depth, mask, NONE, WatchConfig(shift_px=3, imu_shift_px=2, background_patch_px=None))
    assert blind.update(swayed_rgb, swayed_depth, NONE, predicted_shift=(7.0, 1.0))["vote"] == "moved"
    # The target moved, the background did not: moved.
    moved_rgb, moved_depth, _ = scene(target_at=(30, 52))
    result = watch.update(moved_rgb, moved_depth, NONE)
    assert result["vote"] == "moved" and result["background_shift_px"] == [0.0, 0.0]
    # Robot pixels at a patch's match are never used; too few patches left: the IMU's shift (or none).
    covered = np.ones(SHAPE, bool)
    assert background_shift(watch.reference_grey, watch.reference_grey, watch.corners, covered, config=config) == (
        None,
        0,
    )
    # A watcher in stop mode votes on the background's shift alone, without an IMU.
    ok = SimpleNamespace(healthy=True)
    request, capture, phase = watcher_inputs([(swayed_rgb, swayed_depth, ok)] * 3, ["approach"] * 3 + ["close"])
    watcher = ExecutionWatcher(
        request, capture, far_spheres, phase, config=config, clock=lambda: 0.0, require_orientation=True
    )
    watcher._run()
    assert watcher.report()["votes"] == {"still": 3, "moved": 0, "unknown": 0}
