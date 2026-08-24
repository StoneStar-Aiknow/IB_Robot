import numpy as np

from object_tracker.template_tracker import TemplateTracker


def _frame_with_target(shape, center, size=40, background=20, target=220, noise_seed=None):
    height, width = shape
    gray = np.full((height, width), background, dtype=np.uint8)
    cx, cy = center
    x_min = int(np.clip(cx - size / 2, 0, width - 1))
    x_max = int(np.clip(cx + size / 2, 0, width))
    y_min = int(np.clip(cy - size / 2, 0, height - 1))
    y_max = int(np.clip(cy + size / 2, 0, height))
    gray[y_min:y_max, x_min:x_max] = target
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, 4, gray.shape)
        gray = np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return gray


def _bbox_around(center, size=56):
    cx, cy = center
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def test_initialize_and_track_static_target():
    tracker = TemplateTracker()
    frame0 = _frame_with_target((240, 320), (160, 120))
    assert tracker.initialize(frame0, _bbox_around((160, 120)))

    frame1 = _frame_with_target((240, 320), (166, 124))
    update = tracker.update(frame1)

    assert update is not None
    assert update.match_score > 0.8
    center = update.center
    assert abs(center[0] - 166) <= 3
    assert abs(center[1] - 124) <= 3


def test_update_rejects_when_target_absent():
    tracker = TemplateTracker()
    frame0 = _frame_with_target((240, 320), (160, 120))
    assert tracker.initialize(frame0, _bbox_around((160, 120)))

    blank = np.full((240, 320), 20, dtype=np.uint8)
    assert tracker.update(blank) is None


def test_track_with_noise_and_gradual_motion():
    tracker = TemplateTracker()
    centers = [(160, 120), (168, 124), (177, 129), (187, 135)]
    first = _frame_with_target((240, 320), centers[0], noise_seed=1)
    assert tracker.initialize(first, _bbox_around(centers[0]))

    for index, center in enumerate(centers[1:], start=1):
        frame = _frame_with_target((240, 320), center, noise_seed=index + 10)
        update = tracker.update(frame)
        assert update is not None
        assert abs(update.center[0] - center[0]) <= 4
        assert abs(update.center[1] - center[1]) <= 4


def test_wide_search_reacquires_after_jump():
    tracker = TemplateTracker()
    first = _frame_with_target((240, 320), (100, 120))
    assert tracker.initialize(first, _bbox_around((100, 120)))

    moved = _frame_with_target((240, 320), (150, 120))
    assert tracker.update(moved, search_radius_px=90.0) is not None


def test_scale_jump_limit_rejects_runaway_template():
    def striped_frame(shape, center, size, period=5):
        height, width = shape
        gray = np.full((height, width), 20, dtype=np.uint8)
        cx, cy = center
        x_min, x_max = int(cx - size / 2), int(cx + size / 2)
        y_min, y_max = int(cy - size / 2), int(cy + size / 2)
        patch = np.tile(np.arange(size) // period % 2 * 180 + 40, (size, 1)).astype(np.uint8)
        gray[y_min:y_max, x_min:x_max] = patch
        return gray

    tracker = TemplateTracker(scales=(1.18,), scale_jump_limit=1.1)
    first = striped_frame((240, 320), (160, 120), 40)
    assert tracker.initialize(first, _bbox_around((160, 120), size=48))

    larger = striped_frame((240, 320), (160, 120), 48)
    update = tracker.update(larger)
    assert update is None


def test_initialize_rejects_degenerate_boxes():
    tracker = TemplateTracker()
    frame = _frame_with_target((240, 320), (160, 120))
    assert not tracker.initialize(frame, (10.0, 10.0, 10.0, 10.0))
    assert not tracker.initialize(frame, (5.0, 5.0, 100.0, 6.0))
