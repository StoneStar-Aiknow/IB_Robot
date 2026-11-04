"""The ``UnitreeSdk`` facade: factory lifecycle, resources, cleanup.

The vendor channel factory is a process-wide singleton
(``ChannelFactory.Init`` silently ignores re-initialization), so this facade
makes the singleton contract explicit:

- the first :meth:`UnitreeSdk.initialize` wins,
- re-initializing with the same arguments is a no-op,
- re-initializing with different arguments raises
  :class:`~unitree_g1_adapter.errors.UnitreeSdkStateError` instead of silently
  keeping the old domain/interface,

and every channel/client created here is tracked so :meth:`UnitreeSdk.close`
releases them in one place. The vendor RPC clients expose no close; only the
channels own native resources.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from unitree_g1_adapter import vendor as vendor_module
from unitree_g1_adapter.channels import CommandWriter, StateFeed
from unitree_g1_adapter.clients import (
    ArmActionClient,
    AudioClient,
    LocoClient,
    MotionSwitcherClient,
    RobotStateClient,
)
from unitree_g1_adapter.errors import UnitreeSdkInitError, UnitreeSdkStateError


class UnitreeSdk:
    """Owns the vendor channel factory and everything created through it."""

    def __init__(
        self,
        *,
        vendor: Mapping[str, Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._vendor = vendor if vendor is not None else vendor_module.default_vendor()
        self._now = now if now is not None else time.monotonic
        self._domain_id: int | None = None
        self._network_interface: str | None = None
        self._initialized = False
        self._feeds: list[StateFeed] = []
        self._writers: list[CommandWriter] = []
        self._clients: dict[str, Any] = {}
        self._closed = False

    # --- lifecycle ----------------------------------------------------------

    def initialize(self, domain_id: int = 0, network_interface: str | None = None) -> None:
        """Initialize the vendor channel factory once per process.

        ``network_interface`` binds the vendor DDS tier to one NIC (a
        deployment fact, like the G1's wired link); ``None`` lets the vendor
        auto-determine it. Arguments are remembered so a conflicting second
        call fails loudly instead of being silently ignored by the vendor
        singleton.
        """
        if self._closed:
            raise UnitreeSdkStateError("this UnitreeSdk instance is closed")
        domain_id = int(domain_id)
        network_interface = str(network_interface) if network_interface else None
        if self._initialized:
            if (self._domain_id, self._network_interface) != (domain_id, network_interface):
                raise UnitreeSdkStateError(
                    "the vendor channel factory is a process-wide singleton and is already "
                    f"initialized with domain_id={self._domain_id}, "
                    f"network_interface={self._network_interface!r}; refusing the conflicting "
                    f"domain_id={domain_id}, network_interface={network_interface!r}"
                )
            return
        try:
            self._vendor["channel_factory_initialize"](domain_id, network_interface)
        except Exception as exc:  # noqa: BLE001 - vendor raises a bare Exception for every failure
            raise UnitreeSdkInitError(domain_id=domain_id, network_interface=network_interface, cause=exc) from exc
        self._domain_id = domain_id
        self._network_interface = network_interface
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _require_initialized(self) -> None:
        if self._closed:
            raise UnitreeSdkStateError("this UnitreeSdk instance is closed")
        if not self._initialized:
            raise UnitreeSdkStateError(
                "call UnitreeSdk.initialize(domain_id, network_interface) before creating channels or clients"
            )

    # --- resources ------------------------------------------------------------

    def state_feed(
        self,
        topic: str,
        type_key: str,
        *,
        handler: Callable[[Any], None] | None = None,
        queue_len: int = 0,
    ) -> StateFeed:
        """A cached subscriber on a vendor topic (e.g. ``rt/lowstate``)."""
        self._require_initialized()
        feed = StateFeed(
            topic,
            type_key,
            vendor=self._vendor,
            handler=handler,
            queue_len=queue_len,
            now=self._now,
        )
        self._feeds.append(feed)
        return feed

    def command_writer(self, topic: str, type_key: str) -> CommandWriter:
        """A command publisher on a vendor topic (e.g. ``rt/arm_sdk``)."""
        self._require_initialized()
        writer = CommandWriter(topic, type_key, vendor=self._vendor)
        self._writers.append(writer)
        return writer

    def _client(self, key: str, factory: Callable[[], Any]) -> Any:
        self._require_initialized()
        if key not in self._clients:
            self._clients[key] = factory()
        return self._clients[key]

    def loco(self, timeout_s: float) -> LocoClient:
        return self._client("loco", lambda: LocoClient(vendor=self._vendor, timeout_s=timeout_s))

    def arm_action(self, timeout_s: float) -> ArmActionClient:
        return self._client("arm_action", lambda: ArmActionClient(vendor=self._vendor, timeout_s=timeout_s))

    def audio(self, timeout_s: float) -> AudioClient:
        return self._client("audio", lambda: AudioClient(vendor=self._vendor, timeout_s=timeout_s))

    def motion_switcher(self, timeout_s: float) -> MotionSwitcherClient:
        return self._client("motion_switcher", lambda: MotionSwitcherClient(vendor=self._vendor, timeout_s=timeout_s))

    # --- teardown ---------------------------------------------------------------

    def robot_state(self, timeout_s: float) -> RobotStateClient:
        return self._client("robot_state", lambda: RobotStateClient(vendor=self._vendor, timeout_s=timeout_s))

    def close(self) -> None:
        """Close every tracked feed and writer; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for feed in self._feeds:
            try:
                feed.close()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                errors.append(exc)
        for writer in self._writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                errors.append(exc)
        self._feeds.clear()
        self._writers.clear()
        self._clients.clear()
        self._initialized = False
        if errors:
            raise errors[0]
