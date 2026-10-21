"""Resolving the camera, and refusing everything else."""

import pytest

from visual_interaction.session.endpoints import (
    CAMERA_COMPRESSED,
    CAMERA_STEREO_LEFT_COMPRESSED,
    FORBIDDEN_PREFIXES,
    EndpointBindingError,
    assert_observation_only,
    resolve_camera,
)

CAMERA_ENDPOINT = "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"


def description(**overrides):
    """A bound runtime description with one camera, as the runtime publishes it."""
    camera = {
        "kind": "topic",
        "direction": "publish",
        "message_type": "sensor_msgs/msg/CompressedImage",
        "endpoint": CAMERA_ENDPOINT,
        "qos": {"reliability": "reliable", "history": "keep_last", "depth": 5},
    }
    camera.update(overrides)
    return {
        "robot": {"id": "x2", "type": "x2_ultra_omnihand", "runtime_name": "aimdk", "runtime_version": "1"},
        "interfaces": {
            CAMERA_COMPRESSED: camera,
            # A motion interface the runtime really does publish: resolving it
            # must still be refused.
            "joint.arm_stream": {
                "kind": "topic",
                "direction": "subscribe",
                "message_type": "std_msgs/msg/Float64MultiArray",
                "endpoint": "/aima/hal/joint/arm_stream",
            },
        },
    }


class TestResolveCamera:
    def test_the_declared_camera_resolves_to_its_endpoint_and_qos(self):
        endpoint = resolve_camera(description())

        assert endpoint.endpoint == CAMERA_ENDPOINT
        assert endpoint.qos["reliability"] == "reliable"
        assert endpoint.qos["depth"] == 5

    def test_the_default_is_the_half_resolution_relay(self):
        """Measured on the robot: 30.1 Hz at ~80 KB, against 10.0 Hz at ~296 KB.

        The stereo stream the magic mirror watches declares 30 Hz and delivers
        10, at a quarter of this one's frame rate and four times its bytes,
        full-resolution 2064x1552 that a vision-language model immediately
        scales to 448x448 anyway.
        """
        assert CAMERA_COMPRESSED == "camera.rgb_head_front_center.color_half_compressed"

    def test_the_stereo_stream_stays_resolvable(self):
        """A deployment may still choose it, and the mirror reads it."""
        stereo_endpoint = "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed"
        bound = {
            "robot": {"id": "x2", "type": "x2_ultra_omnihand", "runtime_name": "aimdk", "runtime_version": "1"},
            "interfaces": {
                CAMERA_STEREO_LEFT_COMPRESSED: {
                    "kind": "topic",
                    "direction": "publish",
                    "message_type": "sensor_msgs/msg/CompressedImage",
                    "endpoint": stereo_endpoint,
                    "qos": {"reliability": "reliable", "durability": "transient_local", "depth": 5},
                }
            },
        }

        assert resolve_camera(bound, CAMERA_STEREO_LEFT_COMPRESSED).endpoint == stereo_endpoint

    def test_an_unbound_description_is_refused_by_code(self):
        for value in [{}, {"interfaces": None}, "not a mapping", None]:
            with pytest.raises(EndpointBindingError) as error:
                resolve_camera(value)
            assert error.value.code == "description_required"

    def test_a_missing_interface_names_the_id(self):
        broken = description()
        del broken["interfaces"][CAMERA_COMPRESSED]

        with pytest.raises(EndpointBindingError) as error:
            resolve_camera(broken)

        assert error.value.code == "unknown_interface"
        assert error.value.path == CAMERA_COMPRESSED

    @pytest.mark.parametrize(
        "override",
        [
            {"kind": "service"},
            {"direction": "subscribe"},
            {"message_type": "sensor_msgs/msg/Image"},
        ],
        ids=["wrong-kind", "wrong-direction", "uncompressed"],
    )
    def test_a_mismatched_shape_is_refused(self, override):
        with pytest.raises(EndpointBindingError) as error:
            resolve_camera(description(**override))

        assert error.value.code == "invalid_interface"

    @pytest.mark.parametrize("endpoint", ["relative/topic", "", None, 42])
    def test_a_non_absolute_endpoint_is_refused(self, endpoint):
        with pytest.raises(EndpointBindingError) as error:
            resolve_camera(description(endpoint=endpoint))

        assert error.value.code == "invalid_interface"

    def test_an_unknown_camera_id_is_refused_before_the_description_is_read(self):
        """A typo must not become a silently unsubscribed node."""
        with pytest.raises(EndpointBindingError) as error:
            resolve_camera(description(), "camera.made_up.color_compressed")

        assert error.value.code == "unknown_interface"

    def test_the_withdrawn_centre_camera_is_not_resolvable(self):
        """The vendor withdrew it from the open interface set in v0.8.1.

        The board demo used it, so naming it has to fail here rather than
        resolve into a topic nothing promises.
        """
        with pytest.raises(EndpointBindingError):
            resolve_camera(description(), "camera.rgb_head_front_center.color_compressed")

    def test_another_declared_camera_may_be_chosen(self):
        both = description()
        both["interfaces"]["camera.rgbd_head_front.color_compressed"] = {
            "kind": "topic",
            "direction": "publish",
            "message_type": "sensor_msgs/msg/CompressedImage",
            "endpoint": "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed",
        }

        resolved = resolve_camera(both, "camera.rgbd_head_front.color_compressed")

        assert resolved.endpoint.endswith("/rgbd_head_front/rgb_image/compressed")


class TestObservationOnly:
    @pytest.mark.parametrize(
        "interface_id",
        [
            "joint.arm_stream",
            "joint.hand_stream",
            "joint.state",
            "runtime.set_mode",
            "motion.execute",
            "base.velocity",
            "arm.move",
            "hand.grasp",
            "gripper.open",
            "nav.goal",
        ],
    )
    def test_every_motion_family_is_refused(self, interface_id):
        """The runtime exposes a control plane whatever this feature does.

        "We did not start an executor" is not evidence of no side effect;
        refusing to resolve the interface is.
        """
        with pytest.raises(EndpointBindingError) as error:
            assert_observation_only((interface_id,))

        assert error.value.code == "forbidden_interface"
        assert error.value.path == interface_id

    def test_resolving_a_motion_interface_as_a_camera_is_refused(self):
        with pytest.raises(EndpointBindingError) as error:
            resolve_camera(description(), "joint.arm_stream")

        assert error.value.code == "forbidden_interface"

    def test_the_camera_passes(self):
        assert_observation_only((CAMERA_COMPRESSED,))

    def test_reading_runtime_status_is_not_a_motion_interface(self):
        """Only set_mode and the control plane are forbidden, not observation."""
        assert_observation_only(("runtime.status", "runtime.get_status"))

    def test_the_forbidden_list_covers_the_families_that_move_the_robot(self):
        for prefix in ("joint.", "motion.", "base.", "arm.", "hand.", "gripper.", "nav."):
            assert prefix in FORBIDDEN_PREFIXES

    def test_audio_focus_is_not_control_authority(self):
        """Both are acquired; a future speech coordinator needs the first.

        This is why the check names what it forbids instead of matching on a
        verb like "acquire".
        """
        assert_observation_only(("audio.focus", "audio.play"))
