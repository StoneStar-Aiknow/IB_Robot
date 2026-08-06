from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("ibrobot_msgs")

from skill_catalog.consumer import CatalogIdentity
from skill_catalog.ros_consumer import CatalogViewSynchronizer


class _Client:
    def service_is_ready(self):
        return False


class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def test_new_desired_identity_clears_stale_current_before_snapshot_is_available():
    synchronizer = object.__new__(CatalogViewSynchronizer)
    synchronizer._lock = __import__("threading").RLock()
    old = CatalogIdentity("epoch", 1, "a" * 64)
    new = CatalogIdentity("epoch", 2, "b" * 64)
    synchronizer._current = SimpleNamespace(identity=old)
    synchronizer._desired = old
    synchronizer._snapshot_in_flight = None
    synchronizer._snapshot_client = _Client()

    synchronizer._select(new)

    assert synchronizer.current is None
    assert synchronizer._desired == new


def test_delayed_old_status_cannot_regress_newer_desired_generation():
    synchronizer = object.__new__(CatalogViewSynchronizer)
    synchronizer._lock = __import__("threading").RLock()
    desired = CatalogIdentity("epoch", 2, "b" * 64)
    synchronizer._current = None
    synchronizer._desired = desired
    synchronizer._status_in_flight = True
    synchronizer._snapshot_in_flight = None
    synchronizer._snapshot_client = _Client()
    stale = SimpleNamespace(
        schema_version=1,
        registry_epoch="epoch",
        registry_generation=1,
        registry_digest="a" * 64,
    )

    synchronizer._handle_status(_DoneFuture(stale))

    assert synchronizer._desired == desired
