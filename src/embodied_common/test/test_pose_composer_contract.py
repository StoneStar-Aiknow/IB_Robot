"""Read-only pose lookup has its own strict, default-on configuration."""

import pytest

from embodied_common.pose_composer_contract import lookup_enabled


def test_lookup_defaults_on_independently_of_writable_memory():
    assert lookup_enabled({})
    assert lookup_enabled({"memory": {"enabled": False}})
    assert lookup_enabled({"lookup": {"enabled": True}})
    assert not lookup_enabled({"lookup": {"enabled": False}, "memory": {"enabled": True}})


@pytest.mark.parametrize("lookup", [None, True, [], {"enabled": "false"}, {"enabled": 0}, {"other": True}])
def test_invalid_lookup_configuration_is_refused(lookup):
    with pytest.raises(ValueError, match="pose_composer.lookup"):
        lookup_enabled({"lookup": lookup})
