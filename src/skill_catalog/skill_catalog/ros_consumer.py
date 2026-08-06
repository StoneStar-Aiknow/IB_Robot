"""Asynchronous exact-snapshot synchronization for planner-side consumers."""

from __future__ import annotations

from threading import RLock

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import SkillRegistryEvent
from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot
from skill_catalog.consumer import CatalogConsumerError, CatalogIdentity, VerifiedCatalogView, verify_snapshot_response


class CatalogViewSynchronizer:
    """Keep a verified current catalog view without blocking business callbacks."""

    def __init__(
        self,
        node,
        *,
        status_service: str,
        snapshot_service: str,
        event_topic: str,
        sync_period_sec: float = 1.0,
    ) -> None:
        self._node = node
        self._lock = RLock()
        self._current: VerifiedCatalogView | None = None
        self._desired: CatalogIdentity | None = None
        self._status_in_flight = False
        self._snapshot_in_flight: tuple[str, int] | None = None
        self._status_client = node.create_client(GetSkillGatewayStatus, status_service)
        self._snapshot_client = node.create_client(GetSkillSnapshot, snapshot_service)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(SkillRegistryEvent, event_topic, self._handle_event, qos)
        node.create_timer(sync_period_sec, self._sync_status)

    @property
    def current(self) -> VerifiedCatalogView | None:
        with self._lock:
            return self._current

    def _sync_status(self) -> None:
        with self._lock:
            if self._status_in_flight or not self._status_client.service_is_ready():
                return
            self._status_in_flight = True
        request = GetSkillGatewayStatus.Request()
        request.schema_version = 1
        try:
            future = self._status_client.call_async(request)
        except Exception:
            with self._lock:
                self._status_in_flight = False
            return
        future.add_done_callback(self._handle_status)

    def _handle_status(self, future) -> None:
        with self._lock:
            self._status_in_flight = False
        try:
            response = future.result()
        except Exception:
            return
        if (
            response is None
            or response.schema_version != 1
            or not response.registry_epoch
            or response.registry_generation <= 0
            or not response.registry_digest
        ):
            return
        identity = CatalogIdentity(
            str(response.registry_epoch),
            int(response.registry_generation),
            str(response.registry_digest),
        )
        with self._lock:
            desired = self._desired
            if desired is not None:
                if identity.registry_epoch != desired.registry_epoch:
                    return
                if identity.registry_epoch == desired.registry_epoch and identity.generation < desired.generation:
                    return
                if (
                    identity.registry_epoch == desired.registry_epoch
                    and identity.generation == desired.generation
                    and identity.registry_digest != desired.registry_digest
                ):
                    self._current = None
                    return
        self._select(identity)

    def _handle_event(self, event: SkillRegistryEvent) -> None:
        if (
            event.schema_version != 1
            or not event.registry_epoch
            or event.new_generation <= 0
            or not event.registry_digest
        ):
            return
        identity = CatalogIdentity(str(event.registry_epoch), int(event.new_generation), str(event.registry_digest))
        with self._lock:
            desired = self._desired
            if desired is not None:
                if identity.registry_epoch == desired.registry_epoch and identity.generation <= desired.generation:
                    return
                if identity.registry_epoch != desired.registry_epoch:
                    self._current = None
        self._select(identity)

    def _select(self, identity: CatalogIdentity) -> None:
        with self._lock:
            current = self._current
            if current is not None and current.identity == identity:
                self._desired = identity
                return
            self._desired = identity
            # A newer identity is not usable until its exact snapshot has
            # been verified. Never leave stale capabilities visible.
            self._current = None
            key = (identity.registry_epoch, identity.generation)
            if self._snapshot_in_flight == key or not self._snapshot_client.service_is_ready():
                return
            self._snapshot_in_flight = key
        request = GetSkillSnapshot.Request()
        request.schema_version = 1
        request.registry_epoch = identity.registry_epoch
        request.generation = identity.generation
        try:
            future = self._snapshot_client.call_async(request)
        except Exception:
            with self._lock:
                self._snapshot_in_flight = None
            return
        future.add_done_callback(lambda completed, expected=identity: self._handle_snapshot(expected, completed))

    def _handle_snapshot(self, expected: CatalogIdentity, future) -> None:
        with self._lock:
            self._snapshot_in_flight = None
            desired = self._desired
        if desired != expected:
            return
        try:
            view = verify_snapshot_response(future.result(), expected)
        except (CatalogConsumerError, Exception):
            return
        with self._lock:
            if self._desired == expected:
                self._current = view
