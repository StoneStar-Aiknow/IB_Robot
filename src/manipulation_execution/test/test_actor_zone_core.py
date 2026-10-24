"""The standing-zone actor core, rule by rule (dev plan A2, B1-B23, C1-C15).

``ZoneCore`` decides once per YOLOX refresh which person standing inside the
floor zone the mirror follows. The expectations here are written from the
design (x2_actor_select_design.md sections 5-8), not read off the code: each
test builds one scenario of the plan's table out of synthetic boxes and checks
what the design says the answer must be.

The zone is built in the test, not read from a file. It is a rectangle in the
1344x972 half stream so that zone coordinates can be worked out by hand:
x 472..872 and y 682..932, so u = (x - 472) / 400 and v = (932 - y) / 250. The
image band is x 200..1144. A person is a 400 px tall, 160 px wide box whose
feet are at the box's bottom centre, so b = 0.12 * 400 = 48 px and a body
core is 96 px wide. Refreshes are 1.0 s apart in camera time.

Where a scenario depends on the sticky foot x (design 6.2: the previous foot x
clamped into [x1 + b, x2 - b] of the new box), the comment next to the box
gives the foot that rule produces.
"""

import dataclasses
import random

import pytest

from manipulation_execution import actor_zone_core
from manipulation_execution.actor_zone_config import TuningConfig, ZoneConfig
from manipulation_execution.actor_zone_core import (
    ACQUIRING,
    GRACE,
    HOLD,
    IDLE,
    LOCKED,
    NO_ACTOR,
    Box,
    Homography,
    ZoneCore,
    in_zone,
)

SIZE = (1344, 972)
LEFT, RIGHT, NEAR, FAR = 472.0, 872.0, 932.0, 682.0
ZONE_W, ZONE_D = RIGHT - LEFT, NEAR - FAR
BAND = (200.0, 1144.0)
ZONE = ZoneConfig(
    path="/test/actor_zone.yaml",
    sha256="z" * 64,
    image_size=SIZE,
    crop_x=BAND,
    floor_quad=((LEFT, NEAR), (RIGHT, NEAR), (RIGHT, FAR), (LEFT, FAR)),
    near_edge_margin_px=30,
    note="test",
)
DEFAULTS = TuningConfig()
THRESHOLD = 0.30
H = 400.0
W = 160.0
ZERO_BOX = (0.0, 0.0, 0.0, 0.0)


def person(x, y2, h=H, w=W, conf=0.9):
    """A standing person with feet at (x, y2)."""
    return Box("person", conf, (x - w / 2.0, y2 - h, x + w / 2.0, y2))


def at(u, v, **kwargs):
    """A standing person with feet at zone coordinates (u, v)."""
    return person(LEFT + u * ZONE_W, NEAR - v * ZONE_D, **kwargs)


def answer(core):
    return (core.state, core.present, core.actor_id, core.bbox, core.confidence)


class Feed:
    """A ZoneCore fed one refresh per call, camera stamps 1.0 s apart."""

    def __init__(self, tuning=DEFAULTS, zone=ZONE, start=100.0):
        self.core = ZoneCore(zone, tuning, THRESHOLD)
        self.stamp = start - 1.0
        self.answers = []

    def __call__(self, *boxes, dt=1.0):
        self.stamp += dt
        report = self.core.step(self.stamp, SIZE, list(boxes))
        assert report is not None, f"stamp {self.stamp} was not new"
        self.answers.append(answer(self.core))
        return report


def lock_on(feed, *boxes):
    """Two refreshes of the same people: ACQUIRING, then LOCKED (acquire_updates = 2)."""
    feed(*boxes)
    assert feed.core.state == ACQUIRING
    feed(*boxes)
    assert feed.core.state == LOCKED


def target_entry(report):
    return next(t for t in report["tracks"] if t["track"] == report["target_track"])


def other_track(report):
    """The one track in the report that is not the target's."""
    (track,) = [t["track"] for t in report["tracks"] if t["track"] != report["target_track"]]
    return track


# ---------------------------------------------------------------- A2: the homography

def test_trapezoid_corners_map_to_the_unit_square_corners():
    quad = ((452.0, 932.0), (892.0, 932.0), (812.0, 682.0), (532.0, 682.0))
    h = Homography(quad)
    for (x, y), expected in zip(quad, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
        u, v = h.to_zone(x, y)
        assert u == pytest.approx(expected[0], abs=1e-9)
        assert v == pytest.approx(expected[1], abs=1e-9)


def test_trapezoid_points_inside_map_into_the_square_and_points_outside_do_not():
    quad = ((452.0, 932.0), (892.0, 932.0), (812.0, 682.0), (532.0, 682.0))
    h = Homography(quad)
    # The trapezoid is symmetric about x = 672, so that line is u = 0.5.
    for y in (700.0, 800.0, 900.0):
        assert h.to_zone(672.0, y)[0] == pytest.approx(0.5, abs=1e-9)
    # At y = 800 the left edge is at x 494.2 and the right edge at 849.8.
    for x, y in ((672.0, 807.0), (500.0, 800.0), (840.0, 800.0), (672.0, 690.0), (672.0, 925.0)):
        assert in_zone(h.to_zone(x, y)), (x, y)
    for x, y in ((440.0, 800.0), (900.0, 800.0), (672.0, 660.0), (672.0, 950.0)):
        assert not in_zone(h.to_zone(x, y)), (x, y)


def test_rectangle_zone_of_these_tests_is_linear_in_x_and_y():
    # Pins the hand-worked u, v every other test relies on.
    h = Homography(ZONE.floor_quad)
    for x, y in ((472.0, 932.0), (672.0, 807.0), (512.0, 922.0), (800.0, 700.0), (412.0, 650.0)):
        u, v = h.to_zone(x, y)
        assert u == pytest.approx((x - LEFT) / ZONE_W, abs=1e-9)
        assert v == pytest.approx((NEAR - y) / ZONE_D, abs=1e-9)


# ---------------------------------------------------------------- B: picking and holding on

def test_same_stamp_twenty_times_changes_nothing_and_reports_once():
    core = ZoneCore(ZONE, DEFAULTS, THRESHOLD)
    a = at(0.5, 0.5)
    assert core.step(100.0, SIZE, [a]) is not None
    before = (core.state, core.track_count, core.last_stamp)
    assert core.state == ACQUIRING
    for _ in range(19):
        assert core.step(100.0, SIZE, [a]) is None
    assert (core.state, core.track_count, core.last_stamp) == before

    # Locked, the same stamp with nobody in it must not count bad refreshes either.
    assert core.step(101.0, SIZE, [a]) is not None
    assert core.state == LOCKED
    for _ in range(20):
        assert core.step(101.0, SIZE, []) is None
    assert core.state == LOCKED
    assert core.step(102.0, SIZE, []) is not None
    assert core.state == GRACE
    assert core.step(103.0, SIZE, []) is not None
    assert core.state == GRACE


def test_one_person_entering_is_acquiring_then_locked_with_actor_id_one():
    feed = Feed()
    a = at(0.5, 0.5)
    assert feed.core.actor_id == NO_ACTOR
    feed(a)
    assert feed.core.state == ACQUIRING
    assert not feed.core.present
    assert feed.core.actor_id == NO_ACTOR
    assert feed.core.bbox == ZERO_BOX
    assert feed.core.confidence == 0.0
    assert feed.core.reason == "zone_acquiring"
    feed(a)
    assert feed.core.state == LOCKED
    assert feed.core.present
    assert feed.core.actor_id == 1
    assert feed.core.bbox == a.bbox
    assert feed.core.confidence == pytest.approx(0.9)
    assert feed.core.reason == "zone_locked"


def test_acquiring_restarts_the_count_when_someone_else_becomes_the_best():
    feed = Feed()
    p1 = at(0.5, 0.9)  # score 0.6 * 1.0 + 0.4 * 0.1 = 0.64
    p2_back = at(0.25, 0.5)  # 0.6 * 0.5 + 0.4 * 0.5 = 0.50
    p2_front = at(0.25, 0.04)  # 0.6 * 0.5 + 0.4 * 0.96 = 0.684
    feed(p1, p2_back)
    assert feed.core.state == ACQUIRING
    feed(p1, p2_front)
    assert feed.core.state == ACQUIRING, "p2 has been the best once, not twice"
    feed(p1, p2_front)
    assert feed.core.state == LOCKED
    assert feed.core.bbox == p2_front.bbox


def test_first_pick_prefers_the_centred_person_over_the_one_in_front():
    centred = at(0.5, 0.8)  # 0.6 * 1.0 + 0.4 * 0.2 = 0.68
    in_front = at(0.1, 0.04)  # 0.6 * 0.2 + 0.4 * 0.96 = 0.504
    feed = Feed()
    lock_on(feed, centred, in_front)
    assert feed.core.bbox == centred.bbox

    # The same two people with only the front weight: now the one in front wins.
    feed = Feed(TuningConfig(center_weight=0.0, front_weight=1.0))
    lock_on(feed, centred, in_front)
    assert feed.core.bbox == in_front.bbox


def test_locked_actor_is_kept_when_a_higher_scoring_person_enters_the_zone():
    feed = Feed()
    a = at(0.3, 0.5)  # score 0.56
    lock_on(feed, a)
    b = at(0.55, 0.2)  # score 0.86; cores [544, 640] and [644, 740] do not overlap
    for _ in range(10):
        feed(a, b)
        assert feed.core.state == LOCKED
        assert feed.core.actor_id == 1
        assert feed.core.bbox == a.bbox


def test_eight_people_moving_outside_the_zone_or_the_band_do_not_change_the_actor():
    feed = Feed()
    a = at(0.5, 0.5)
    lock_on(feed, a)
    for i in range(10):
        s = 20.0 if i % 2 else -20.0
        others = [
            # In the band, behind the zone (v > 1).
            person(350.0 + s, 600.0, h=300.0),
            person(560.0 - s, 590.0, h=300.0),
            person(790.0 + s, 600.0, h=300.0),
            person(1000.0 - s, 610.0, h=300.0),
            # In the band, beside the zone.
            person(300.0 + s, 850.0),
            person(1050.0 - s, 850.0),
            # Outside the band.
            person(100.0 + s / 2.0, 850.0),
            person(1260.0 - s / 2.0, 850.0),
        ]
        feed(a, *others)
        assert feed.core.state == LOCKED
        assert feed.core.actor_id == 1
        assert feed.core.bbox == a.bbox
        assert feed.core.persons_count == 9


def test_people_entirely_outside_the_band_get_no_track_and_occlude_nobody():
    # Band edge on the zone's left edge, so a box can be just outside the band
    # and still cover the target's core.
    zone = dataclasses.replace(ZONE, crop_x=(LEFT, BAND[1]))
    a = at(0.02, 0.5)  # feet (480, 807), core [432, 528]
    # 90 px wide, narrower than 2b, so its foot is the midline: core [377, 473]
    # or [380, 476], overlapping a's core by 41 or 44 px (> 0.3 * 96), and 93 px lower.
    outside = Box("person", 0.9, (380.0, 500.0, 470.0, 900.0))
    touching = Box("person", 0.9, (383.0, 500.0, 473.0, 900.0))

    feed = Feed(zone=zone)
    lock_on(feed, a)
    for _ in range(3):
        report = feed(a, outside)
        assert feed.core.state == LOCKED
        assert feed.core.track_count == 1
        assert feed.core.persons_count == 2
        assert [t["box"] for t in report["tracks"]] == [0]

    # The same person one pixel into the band is kept, and stands in front of a.
    feed = Feed(zone=zone)
    lock_on(feed, a)
    report = feed(a, touching)
    assert feed.core.track_count == 2
    assert target_entry(report)["occluded_by"] == other_track(report)
    assert feed.core.state == GRACE


def step_across(feed):
    """B8: one step from the zone's left side to its right, a bystander standing outside."""
    a_left = at(0.1, 0.5)  # feet (512, 807)
    bystander = person(960.0, 807.0)  # u 1.22: outside the zone, inside the band
    lock_on(feed, a_left, bystander)
    # Box [752, 912]: the foot 512 is pushed to x1 + b = 800, u 0.82, still inside.
    a_right = at(0.9, 0.5)
    return [feed(a_right, bystander) for _ in range(3)], a_right


def test_one_step_from_left_to_right_inside_the_zone_keeps_the_actor():
    feed = Feed()
    reports, a_right = step_across(feed)
    for report in reports:
        assert report["state"] == LOCKED
        assert report["actor_id"] == 1
        assert not report["handoff_suspected"]
        assert not target_entry(report)["handoff_suspected"]
    assert feed.core.bbox == a_right.bbox


@pytest.mark.parametrize("dy", [-60.0, 60.0])
def test_a_step_forward_or_back_inside_the_zone_keeps_the_actor(dy):
    feed = Feed()
    lock_on(feed, person(672.0, 807.0))
    moved = person(672.0, 807.0 + dy)  # 0.15 h, y2 747 or 867, both inside
    for _ in range(3):
        feed(moved)
        assert feed.core.state == LOCKED


def test_opening_both_arms_near_the_zone_side_is_not_a_bad_refresh():
    feed = Feed()
    normal = person(504.0, 807.0)  # u 0.08
    lock_on(feed, normal)
    # As wide as the person is tall, top 0.3 h higher, bottom unchanged: box
    # height 520, b 62.4, the foot 504 stays inside [366.4, 641.6].
    arms_open = Box("person", 0.9, (304.0, 287.0, 704.0, 807.0))
    for box in (arms_open, normal, arms_open, arms_open):
        report = feed(box)
        assert feed.core.state == LOCKED
        assert report["events"] == []
        assert target_entry(report)["foot"][0] == pytest.approx(504.0)


def test_one_arm_out_stretches_the_box_but_not_the_sticky_foot():
    feed = Feed()
    a = person(512.0, 807.0)  # u 0.1, box [432, 592]
    lock_on(feed, a)
    # Left arm out 0.4 h: box [272, 592]. Its midline 432 would be u -0.1; the
    # foot 512 stays inside [320, 544].
    one_arm = Box("person", 0.9, (272.0, 407.0, 592.0, 807.0))
    for _ in range(3):
        report = feed(one_arm)
        entry = target_entry(report)
        assert entry["foot"][0] == pytest.approx(512.0)
        assert entry["in_zone"]
        assert feed.core.state == LOCKED


def test_raising_hands_and_bending_over_are_not_bad_refreshes():
    feed = Feed()
    lock_on(feed, person(672.0, 807.0))
    hands_up = person(672.0, 807.0, h=1.4 * H)
    bent = person(672.0, 807.0, h=0.7 * H)
    for box in (hands_up, bent, hands_up, person(672.0, 807.0), bent):
        report = feed(box)
        assert feed.core.state == LOCKED
        assert report["events"] == []


def test_missing_once_or_twice_is_grace_with_the_last_box_then_locked_again():
    feed = Feed()
    bystander = person(300.0, 650.0, h=300.0)  # in the band, outside the zone
    lock_on(feed, at(0.5, 0.5), bystander)
    last = person(676.0, 807.0, conf=0.8)
    feed(last, bystander)
    assert feed.core.bbox == last.bbox

    for _ in range(2):
        feed(bystander)
        assert feed.core.state == GRACE
        assert feed.core.present
        assert feed.core.actor_id == 1
        assert feed.core.bbox == last.bbox
        assert feed.core.confidence == pytest.approx(0.8)
        assert feed.core.reason == "zone_grace"

    back = person(672.0, 807.0, conf=0.7)
    feed(back, bystander)
    assert feed.core.state == LOCKED
    assert feed.core.bbox == back.bbox
    assert feed.core.confidence == pytest.approx(0.7)
    # The count was cleared: two more misses are grace again, not a release.
    feed(bystander)
    feed(bystander)
    assert feed.core.state == GRACE


def test_missing_three_times_in_a_row_releases_to_hold():
    feed = Feed()
    lock_on(feed, at(0.5, 0.5))
    feed()
    feed()
    assert feed.core.state == GRACE
    feed()
    assert feed.core.state == HOLD
    assert not feed.core.present
    assert feed.core.actor_id == NO_ACTOR
    assert feed.core.bbox == ZERO_BOX
    assert feed.core.confidence == 0.0
    assert feed.core.reason == "zone_hold"


def test_an_empty_yolox_result_is_one_bad_refresh():
    feed = Feed()
    lock_on(feed, at(0.5, 0.5))
    report = feed()
    assert feed.core.state == GRACE
    assert report["bad"] == 1
    assert feed.core.persons_count == 0


def test_two_refreshes_outside_then_back_keeps_the_actor_and_clears_the_count():
    feed = Feed()
    a = person(672.0, 807.0)
    lock_on(feed, a)
    # Box [300, 460]: the foot is pushed to x2 - b = 412, u -0.15.
    out_left = person(380.0, 807.0)
    for _ in range(2):
        feed(out_left)
        feed(out_left)
        assert feed.core.state == GRACE
        feed(a)
        assert feed.core.state == LOCKED
        assert feed.core.actor_id == 1


@pytest.mark.parametrize(
    "outside",
    [
        pytest.param(person(380.0, 807.0), id="left"),  # foot pushed to 412, u -0.15
        pytest.param(person(964.0, 807.0), id="right"),  # foot pushed to 932, u 1.15
        pytest.param(person(672.0, 650.0), id="far"),  # v 1.128
        pytest.param(person(672.0, 950.0), id="near"),  # v -0.072, feet still in the frame
    ],
)
def test_three_refreshes_outside_releases_in_every_direction(outside):
    feed = Feed()
    lock_on(feed, person(672.0, 807.0))
    feed(outside)
    assert feed.core.state == GRACE
    feed(outside)
    assert feed.core.state == GRACE
    feed(outside)
    assert feed.core.state == HOLD
    assert not feed.core.present


def test_feet_cut_off_by_the_frame_bottom_are_outside_the_zone():
    feed = Feed()
    lock_on(feed, person(672.0, 807.0))
    report = feed(person(672.0, 965.0))
    assert not target_entry(report)["in_zone"]
    assert feed.core.state == GRACE


def test_a_neighbour_in_the_zone_is_not_taken_for_the_missed_actor():
    feed = Feed()
    a = at(0.45, 0.5)  # score 0.74, core [604, 700]
    b = at(0.8, 0.5)  # score 0.44, core [744, 840]
    lock_on(feed, a, b)
    assert feed.core.bbox == a.bbox
    feed(b)
    assert feed.core.state == GRACE, "b has its own track; a was missed"
    assert feed.core.bbox == a.bbox
    feed(a, b)
    assert feed.core.state == LOCKED
    assert feed.core.bbox == a.bbox
    assert feed.core.actor_id == 1


def test_a_person_behind_the_actor_outside_the_zone_changes_nothing():
    feed = Feed()
    a = person(672.0, 807.0)
    behind = person(672.0, 640.0, h=320.0, w=128.0)  # v 1.168, directly behind
    lock_on(feed, a, behind)
    assert feed.core.bbox == a.bbox
    for _ in range(5):
        report = feed(a, behind)
        assert feed.core.state == LOCKED
        assert target_entry(report)["occluded_by"] is None


def test_a_duplicate_upper_body_box_neither_occludes_nor_replaces_the_actor():
    feed = Feed()
    a = person(672.0, 807.0)
    lock_on(feed, a)
    upper_body = Box("person", 0.6, (602.0, 407.0, 742.0, 627.0))
    for _ in range(5):
        report = feed(a, upper_body)
        assert feed.core.state == LOCKED
        assert feed.core.bbox == a.bbox
        assert target_entry(report)["occluded_by"] is None


@pytest.mark.parametrize(
    "start_u, end_u",
    [pytest.param(0.2, 0.8, id="left_to_right"), pytest.param(0.8, 0.2, id="right_to_left")],
)
@pytest.mark.parametrize("gate_dx", [0.3, None])
def test_gate_dx_refuses_a_long_step_and_no_gate_accepts_it(start_u, end_u, gate_dx):
    feed = Feed(TuningConfig(gate_dx=gate_dx))
    start = at(start_u, 0.5)
    lock_on(feed, start)
    # 240 px between the box centres; the sticky foot moves 208 px = 0.52 h.
    end = at(end_u, 0.5)
    feed(end)
    if gate_dx is None:
        assert feed.core.state == LOCKED
        assert feed.core.bbox == end.bbox
    else:
        assert feed.core.state == GRACE
        assert feed.core.bbox == start.bbox


def test_grace_outside_the_zone_reports_this_refresh_box_and_confidence():
    feed = Feed()
    lock_on(feed, person(672.0, 807.0, conf=0.9))
    out_left = person(380.0, 807.0, conf=0.6)
    feed(out_left)
    assert feed.core.state == GRACE
    assert feed.core.bbox == out_left.bbox
    assert feed.core.confidence == pytest.approx(0.6)


# ---------------------------------------------------------------- C: occlusion, release, hold

def legs_blocked(feed):
    """C1: someone steps right in front of the actor and hides their legs."""
    a = person(672.0, 807.0)
    lock_on(feed, a)
    # a's bottom is 0.2 h higher (box height 320, core [633.6, 710.4]); the
    # blocker's core [624, 720] covers it, its bottom 173 px lower.
    a_cut = Box("person", 0.9, (592.0, 407.0, 752.0, 727.0))
    blocker = person(672.0, 900.0)
    return [feed(a_cut, blocker) for _ in range(3)]


def test_rule_o_someone_in_front_hiding_the_legs_releases_after_three():
    feed = Feed()
    reports = legs_blocked(feed)
    for report in reports[:2]:
        assert report["state"] == GRACE
        assert target_entry(report)["occluded_by"] == other_track(report)
        # The reference stays at the last good position (y2 807).
        assert target_entry(report)["dy"] == pytest.approx(-0.2)
    assert reports[2]["state"] == HOLD
    assert not feed.core.present


def test_rule_o_a_child_right_in_front_occludes():
    feed = Feed()
    a = person(672.0, 807.0)
    lock_on(feed, a)
    child = person(672.0, 900.0, h=0.6 * H, w=100.0)  # core [643.2, 700.8]
    report = feed(a, child)
    assert target_entry(report)["occluded_by"] == other_track(report)
    assert feed.core.state == GRACE
    feed(a, child)
    feed(a, child)
    assert feed.core.state == HOLD


def test_rule_o_a_child_held_in_the_arms_does_not_occlude_and_nobody_is_swapped():
    feed = Feed()
    mother = person(672.0, 807.0)
    # Inside the mother's box, bottom 47 px higher, its feet still inside the zone.
    child = Box("person", 0.8, (632.0, 560.0, 712.0, 760.0))
    lock_on(feed, mother, child)
    assert feed.core.bbox == mother.bbox
    for i in range(10):
        s = 4.0 if i % 2 else -4.0
        m = Box("person", 0.9, (592.0 + s, 407.0, 752.0 + s, 807.0))
        c = Box("person", 0.8, (632.0 - s, 560.0, 712.0 - s, 760.0))
        report = feed(m, c)
        assert target_entry(report)["occluded_by"] is None
        assert feed.core.state == LOCKED
        assert feed.core.actor_id == 1
        assert feed.core.bbox == m.bbox


def test_rule_o_a_neighbour_opening_arms_over_the_actor_does_not_occlude():
    feed = Feed()
    actor = person(652.0, 807.0)  # core [604, 700]
    neighbour = person(772.0, 830.0)  # 23 px lower than the actor, core [724, 820]
    lock_on(feed, actor, neighbour)
    assert feed.core.bbox == actor.bbox
    # Arms open, covering the actor's whole box: height 520, b 62.4, the
    # neighbour's foot stays at 772, core [709.6, 834.4].
    arms_open = Box("person", 0.9, (572.0, 310.0, 972.0, 830.0))
    for _ in range(3):
        report = feed(actor, arms_open)
        assert target_entry(report)["occluded_by"] is None
        assert feed.core.state == LOCKED


def follow_the_box(feed):
    """C5: b jumps from beside a to right in front of a within one refresh."""
    a = person(672.0, 807.0)
    lock_on(feed, a)
    b_side = person(780.0, 830.0)  # core [732, 828], no overlap with a's [624, 720]
    feed(a, b_side)
    feed(a, b_side)
    # Only b's box is left. a's track: dx 0, dy 73 -> 0.18; b's track: foot
    # 780 -> 704, dx -76, dy 50 -> 0.23. a's track is nearer and takes it.
    b_front = person(672.0, 880.0)
    return [feed(b_front), feed(b_front)], b_front


def test_following_the_box_when_someone_covers_the_actor_entirely():
    feed = Feed()
    reports, b_front = follow_the_box(feed)
    for report in reports:
        assert report["state"] == LOCKED
        assert report["actor_id"] == 1
    assert feed.core.bbox == b_front.bbox
    assert reports[0]["handoff_suspected"]
    assert target_entry(reports[0])["handoff_suspected"]


def slow_cover(feed):
    """C6: b walks in front of a over four refreshes."""
    a = person(672.0, 807.0)
    lock_on(feed, a)
    reports = [feed(a, person(840.0, 880.0))]
    # b's foot: 840 -> 822 -> 772 -> 722. At 722 its core [674, 770] overlaps
    # a's [624, 720] by 46 px (> 28.8): rule O.
    for x in (790.0, 740.0, 690.0):
        reports.append(feed(a, person(x, 880.0)))
    # Then a is hidden: b's box stays with b's own track (foot 722 -> 704).
    b_front = person(672.0, 880.0)
    reports.append(feed(b_front))
    reports.append(feed(b_front))
    return reports


def test_slowly_covering_the_actor_is_occlusion_then_release():
    feed = Feed()
    reports = slow_cover(feed)
    for report in reports[:3]:
        assert report["state"] == LOCKED
    both_visible = reports[3]
    assert len(both_visible["tracks"]) == 2
    assert target_entry(both_visible)["occluded_by"] == other_track(both_visible)
    assert both_visible["state"] == GRACE
    assert reports[4]["state"] == GRACE
    assert reports[5]["state"] == HOLD
    assert not feed.core.present


def test_target_reference_moves_only_on_good_refreshes():
    feed = Feed()
    a = at(0.9, 0.5)  # feet (832, 807)
    lock_on(feed, a)
    # a is missed and a never-seen c stands just outside the right edge: box
    # [880, 1040], a's foot 832 pushed to 928, u 1.14. It is a's only candidate.
    c = person(960.0, 807.0)
    report = feed(c)
    assert target_entry(report)["box"] == 0
    assert not target_entry(report)["in_zone"]
    assert feed.core.state == GRACE
    # Next refresh both are there; a's reference is still (832, 807), so a's
    # own box is at distance 0 and goes to a's track.
    for _ in range(3):
        report = feed(a, c)
        entry = target_entry(report)
        assert entry["box"] == 0
        assert entry["dx"] == pytest.approx(0.0)
        assert entry["dy"] == pytest.approx(0.0)
        assert feed.core.state == LOCKED
        assert feed.core.bbox == a.bbox
    assert all(state != HOLD for state, *_ in feed.answers)


def test_occlusion_rule_switched_off_keeps_the_legs_blocked_actor_locked():
    feed = Feed(TuningConfig(occlude_overlap=None))
    reports = legs_blocked(feed)
    for report in reports:
        assert report["state"] == LOCKED
        assert target_entry(report)["occluded_by"] is None


SCENARIOS = {
    "follow_the_box": lambda feed: follow_the_box(feed)[0],
    "slow_cover": slow_cover,
    "step_across": lambda feed: step_across(feed)[0],
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("forced", [True, False])
def test_handoff_suspected_is_only_a_record(monkeypatch, scenario, forced):
    original = Feed()
    SCENARIOS[scenario](original)

    monkeypatch.setattr(actor_zone_core, "handoff_suspected", lambda target_obs, unmatched: forced)
    patched = Feed()
    reports = SCENARIOS[scenario](patched)

    assert patched.answers == original.answers
    # The replacement was really in effect wherever the target was seen.
    seen = [
        r
        for r in reports
        if r["target_track"] is not None and any(t["track"] == r["target_track"] for t in r["tracks"])
    ]
    assert seen
    assert all(r["handoff_suspected"] is forced for r in seen)


def test_someone_crossing_in_front_for_one_refresh_is_one_grace():
    feed = Feed()
    a = person(672.0, 807.0)
    lock_on(feed, a)
    # The crosser's foot: 480 -> 640 (core [592, 688], overlaps a's by 64) -> 832.
    states = []
    for x in (480.0, 672.0, 864.0, 1000.0):
        feed(a, person(x, 900.0))
        states.append(feed.core.state)
    assert states == [LOCKED, GRACE, LOCKED, LOCKED]
    assert feed.core.actor_id == 1


def test_an_occluded_person_cannot_be_picked():
    feed = Feed()
    back = person(672.0, 782.0)  # score 0.6 + 0.4 * 0.4 = 0.76
    # 75 px lower, core [684, 780] overlapping back's [624, 720] by 36 px.
    front = person(732.0, 857.0)  # score 0.6 * 0.7 + 0.4 * 0.7 = 0.70
    lock_on(feed, back, front)
    assert feed.core.bbox == front.bbox


def leave_with_b_waiting(feed):
    a = at(0.5, 0.5)
    b = at(0.15, 0.5)  # score 0.38, core [484, 580]
    lock_on(feed, a, b)
    assert feed.core.bbox == a.bbox
    for _ in range(3):
        feed(b)
    assert feed.core.state == HOLD
    return b


def test_after_a_release_nobody_is_picked_until_the_hold_ends_then_b_gets_a_new_id():
    feed = Feed()
    b = leave_with_b_waiting(feed)
    for dt in (1.0, 1.0, 0.9):  # 2.9 s of camera time after the release
        feed(b, dt=dt)
        assert feed.core.state == HOLD
        assert not feed.core.present
        assert feed.core.actor_id == NO_ACTOR
    feed(b, dt=0.6)  # 3.5 s
    assert feed.core.state == ACQUIRING
    assert not feed.core.present
    feed(b)
    assert feed.core.state == LOCKED
    assert feed.core.actor_id == 2
    assert feed.core.bbox == b.bbox


def test_with_no_hold_b_is_picked_from_the_next_refresh():
    feed = Feed(TuningConfig(release_hold_s=0.0))
    b = leave_with_b_waiting(feed)
    feed(b)
    assert feed.core.state == ACQUIRING
    feed(b)
    assert feed.core.state == LOCKED
    assert feed.core.actor_id == 2


def test_the_hold_runs_on_camera_time_so_a_repeated_stamp_never_ends_it():
    feed = Feed()
    lock_on(feed, at(0.5, 0.5))
    feed()
    feed()
    feed()
    assert feed.core.state == HOLD
    released_at = feed.stamp
    for _ in range(1000):
        assert feed.core.step(released_at, SIZE, [at(0.5, 0.5)]) is None
    assert feed.core.state == HOLD
    # A frame much later without new detections is not a new stamp either.
    feed.core.unknown(released_at + 100.0)
    assert feed.core.state == HOLD
    feed(at(0.5, 0.5), dt=0.5)
    assert feed.core.state == HOLD
    feed(at(0.5, 0.5), dt=3.0)
    assert feed.core.state == ACQUIRING


def test_camera_time_going_back_clears_tracks_and_restarts_the_hold():
    feed = Feed()
    a = at(0.5, 0.5)
    bystander = person(300.0, 650.0, h=300.0)
    lock_on(feed, a, bystander)
    assert feed.core.track_count == 2

    feed.stamp = 49.0
    feed(a)  # stamp 50, earlier than 101
    assert feed.core.state == HOLD
    assert not feed.core.present
    assert feed.core.track_count == 1, "the bystander's track is gone"
    feed(a)
    feed(a)
    assert feed.core.state == HOLD
    feed(a, dt=1.5)  # 3.5 s after 50
    assert feed.core.state == ACQUIRING
    feed(a)
    assert feed.core.state == LOCKED
    assert feed.core.actor_id == 2


def _random_box(rng):
    h = rng.uniform(80.0, 600.0)
    w = rng.uniform(0.2, 1.0) * h
    x = rng.uniform(0.0, SIZE[0])
    y2 = rng.uniform(300.0, SIZE[1])
    label = "person" if rng.random() < 0.9 else "chair"
    return Box(label, rng.uniform(0.0, 1.0), (x - w / 2.0, y2 - h, x + w / 2.0, y2))


def _tracked(box):
    """Gets a track per design 5: a person, conf >= 0.30, touching the band."""
    x1, _, x2, _ = box.bbox
    return box.label == "person" and box.confidence >= THRESHOLD and x2 >= BAND[0] and x1 <= BAND[1]


def test_tracks_do_not_leak_over_ten_thousand_random_refreshes():
    rng = random.Random(20261024)
    core = ZoneCore(ZONE, DEFAULTS, THRESHOLD)
    keep = DEFAULTS.track_keep_updates
    stamp = 0.0
    window = []
    for i in range(10_000):
        if i % 500 == 499:
            # keep + 1 empty refreshes drop every non-target track and release
            # the target; the old target's track goes after at most keep + 1 more.
            for _ in range(DEFAULTS.release_updates + keep + 1):
                stamp += 1.0
                core.step(stamp, SIZE, [])
            assert core.track_count == 0
            window = []
        if rng.random() < 0.01:
            core.unknown(stamp + 0.5)
        if rng.random() < 0.005:
            stamp -= rng.uniform(1.0, 50.0)
            window = []
        else:
            stamp += rng.uniform(0.5, 1.5)
        boxes = [_random_box(rng) for _ in range(rng.randint(0, 10))]
        assert core.step(stamp, SIZE, boxes) is not None

        # Every surviving non-target track was matched or created in one of the
        # last keep + 1 refreshes, and each refresh feeds one box per track.
        window = (window + [sum(1 for b in boxes if _tracked(b))])[-(keep + 1):]
        assert core.track_count <= sum(window) + 1
        assert core.present == (core.state in (LOCKED, GRACE))
        if not core.present:
            assert core.actor_id == NO_ACTOR
            assert core.bbox == ZERO_BOX
            assert core.confidence == 0.0


def test_known_limitation_a_stranger_appearing_as_the_actor_leaves_is_taken_for_the_actor():
    feed = Feed()
    lock_on(feed, person(672.0, 807.0))
    b = person(752.0, 850.0)  # never seen; a's foot 672 -> 720, u 0.62
    feed(b)
    assert feed.core.state == LOCKED
    assert feed.core.actor_id == 1
    assert feed.core.bbox == b.bbox


def test_known_limitation_two_people_swapping_places_in_one_refresh_swap_the_actor():
    feed = Feed()
    a = at(0.4, 0.5)  # score 0.68
    b = at(0.8, 0.5)  # score 0.44
    lock_on(feed, a, b)
    assert feed.core.bbox == a.bbox
    b_now_left = Box("person", 0.8, a.bbox)
    a_now_right = Box("person", 0.9, b.bbox)
    feed(a_now_right, b_now_left)
    assert feed.core.state == LOCKED
    assert feed.core.actor_id == 1
    assert feed.core.bbox == b_now_left.bbox
    assert feed.core.confidence == pytest.approx(0.8), "b's detection, at a's old place"
