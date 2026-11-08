"""The upper-body crop: geometry, borders, fallbacks and coordinate scaling."""

import numpy as np
import pytest

from visual_interaction.session.look_crop import (
    MIN_BOX_PX,
    SIDE,
    crop_upper_body,
    upper_body_square,
    whole_frame,
)


def frame(width=1344, height=972, value=200):
    return np.full((height, width, 3), value, np.uint8)


def marked(width=1344, height=972):
    """Each pixel's column in channel 0 and row in channel 1 (mod 256)."""
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0] = (np.arange(width) % 256)[None, :]
    image[..., 1] = (np.arange(height) % 256)[:, None]
    image[..., 2] = 255
    return image


class TestGeometry:
    def test_a_standing_person_gives_a_square_over_the_upper_body(self):
        # 200 wide, 600 tall: region is 260 wide, 0.70 * 600 = 420 tall.
        x0, y0, side = upper_body_square((500.0, 200.0, 700.0, 800.0), (1344, 972))
        assert side == 420
        assert y0 == 170  # 5 % of 600 above the top
        assert y0 + side == 590  # 65 % of 600 below the top
        assert x0 + side / 2 == pytest.approx(600, abs=1)  # centred on the box

    def test_a_wide_pose_is_padded_vertically_around_the_region(self):
        # Arms open: 600 wide, 400 tall. Region 780 wide, 280 tall.
        x0, y0, side = upper_body_square((300.0, 300.0, 900.0, 700.0), (1344, 972))
        assert side == 780
        assert x0 == pytest.approx(300 - 90, abs=1)
        centre_y = (280.0 + 560.0) / 2
        assert y0 + side / 2 == pytest.approx(centre_y, abs=1)

    def test_the_crop_shows_the_frame_where_the_square_is_inside_it(self):
        image = marked()
        crop = crop_upper_body(image, (500.0, 200.0, 700.0, 800.0))
        assert crop.fallback is False
        assert crop.image.shape == (SIDE, SIDE, 3) and crop.image.dtype == np.uint8
        assert crop.square == (390, 170, 420)
        # Fully inside the frame, so no black: channel 2 is 255 everywhere.
        assert crop.image[..., 2].min() > 250

    def test_a_frame_that_is_already_448_with_a_matching_square_is_not_resampled(self):
        image = marked(448, 448)
        crop = crop_upper_body(image, (0.0, 0.0, 448.0, 448.0))
        assert crop.fallback is False and crop.image.shape == (SIDE, SIDE, 3)


class TestBorders:
    @pytest.mark.parametrize(
        "bbox, black",
        [
            ((0.0, 100.0, 150.0, 700.0), "left"),
            ((1194.0, 100.0, 1344.0, 700.0), "right"),
            ((500.0, 0.0, 700.0, 600.0), "top"),
        ],
    )
    def test_a_box_at_the_edge_is_padded_black_not_shifted(self, bbox, black):
        image = frame(value=200)
        crop = crop_upper_body(image, bbox)
        assert crop.fallback is False
        x0, y0, side = crop.square
        if black == "left":
            assert x0 < 0
            assert crop.image[SIDE // 2, :5].max() == 0
            assert crop.image[SIDE // 2, -5:].min() > 150
        elif black == "right":
            assert x0 + side > 1344
            assert crop.image[SIDE // 2, -5:].max() == 0
            assert crop.image[SIDE // 2, :5].min() > 150
        else:
            assert y0 < 0
            assert crop.image[:3, SIDE // 2].max() == 0
            assert crop.image[-5:, SIDE // 2].min() > 150

    def test_a_box_larger_than_the_frame_still_crops(self):
        crop = crop_upper_body(frame(640, 480), (-100.0, -50.0, 800.0, 900.0))
        assert crop.fallback is False and crop.image.shape == (SIDE, SIDE, 3)

    def test_a_box_whose_upper_body_is_above_the_frame_falls_back(self):
        # Only the legs are in view: the upper 65 % is all above row 0.
        crop = crop_upper_body(frame(), (500.0, -1500.0, 700.0, 200.0))
        assert crop.fallback is True


class TestFallback:
    @pytest.mark.parametrize(
        "bbox",
        [
            None,
            (0.0, 0.0, 0.0, 0.0),  # ActorState zeros: no actor
            (100.0, 100.0, 100.0 + MIN_BOX_PX - 1, 400.0),  # too narrow
            (100.0, 100.0, 400.0, 100.0 + MIN_BOX_PX - 1),  # too short
            (400.0, 400.0, 100.0, 100.0),  # inverted
            (float("nan"), 0.0, 100.0, 100.0),
            (0.0, 0.0, float("inf"), 100.0),
            (2000.0, 100.0, 2400.0, 600.0),  # right of the frame
            (-500.0, 100.0, -10.0, 600.0),  # left of the frame
            (1330.0, 100.0, 1400.0, 600.0),  # only 14 px inside
            (1.0, 2.0, 3.0),  # wrong length
            ("a", "b", "c", "d"),
        ],
    )
    def test_an_unusable_box_gives_the_whole_frame(self, bbox):
        image = marked()
        crop = crop_upper_body(image, bbox)
        assert crop.fallback is True and crop.square is None
        assert np.array_equal(crop.image, whole_frame(image))

    def test_a_box_of_exactly_the_minimum_is_used(self):
        crop = crop_upper_body(frame(), (100.0, 100.0, 100.0 + MIN_BOX_PX, 100.0 + 2 * MIN_BOX_PX))
        assert crop.fallback is False

    def test_the_whole_frame_is_stretched_like_the_demo(self):
        from PIL import Image

        image = marked(640, 360)
        expected = np.asarray(Image.fromarray(image).resize((SIDE, SIDE), Image.BICUBIC))
        assert np.array_equal(whole_frame(image), expected)

    @pytest.mark.parametrize(
        "bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8), np.zeros((10, 10, 3), np.float32), None]
    )
    def test_a_frame_that_is_not_an_image_is_an_error(self, bad):
        with pytest.raises(ValueError):
            crop_upper_body(bad, (0.0, 0.0, 5.0, 5.0))


class TestNonSquareFrames:
    @pytest.mark.parametrize("size", [(1344, 972), (640, 480), (480, 640), (1280, 200)])
    def test_every_aspect_gives_a_448_square(self, size):
        width, height = size
        image = frame(width, height)
        bbox = (0.3 * width, 0.1 * height, 0.6 * width, 0.95 * height)
        crop = crop_upper_body(image, bbox)
        assert crop.image.shape == (SIDE, SIDE, 3)
        assert whole_frame(image).shape == (SIDE, SIDE, 3)

    def test_a_box_from_a_full_resolution_stream_is_scaled_to_the_half_frame(self):
        half = marked(1344, 972)
        full_box = (1000.0, 400.0, 1400.0, 1600.0)  # measured on 2688x1944
        scaled = crop_upper_body(half, full_box, source_size=(2688, 1944))
        direct = crop_upper_body(half, (500.0, 200.0, 700.0, 800.0))
        assert scaled.square == direct.square
        assert np.array_equal(scaled.image, direct.image)

    def test_a_bad_source_size_falls_back(self):
        crop = crop_upper_body(frame(), (500.0, 200.0, 700.0, 800.0), source_size=(0, 972))
        assert crop.fallback is True
