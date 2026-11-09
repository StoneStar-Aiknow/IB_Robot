"""Errors raised by the Unitree SDK adapter."""

from __future__ import annotations


class UnitreeSdkUnavailableError(RuntimeError):
    """The official unitree_sdk2_python package is not importable.

    The vendor SDK is installed by ``scripts/setup/install_unitree_sdk.sh`` at
    a pinned revision (it is not on PyPI). The G1 runtime fails closed on this
    error instead of pretending the vendor tier is reachable.
    """

    def __init__(self, import_error: str = "") -> None:
        self.import_error = import_error
        super().__init__(
            "unitree_sdk2py is not available: run ./scripts/setup.sh (full profile) to install the "
            "pinned Unitree SDK, or set UNITREE_SDK2_PYTHON_HOME to an existing checkout. "
            f"Import error: {import_error}"
        )


class UnitreeSdkInitError(RuntimeError):
    """The vendor channel factory failed to initialize.

    The official SDK raises a bare ``Exception("channel factory init error.")``
    for every failure — most commonly a ``network_interface`` that does not
    exist on the host. The adapter wraps it with the arguments and the usual
    causes instead of letting that string surface alone.
    """

    def __init__(self, *, domain_id: int, network_interface: str | None, cause: Exception) -> None:
        self.domain_id = domain_id
        self.network_interface = network_interface
        super().__init__(
            f"the vendor channel factory failed to initialize (domain_id={domain_id}, "
            f"network_interface={network_interface!r}): {cause}. "
            "Check that the interface exists (ip link) and that no other UnitreeSdk in this "
            "process already initialized a different domain."
        )


class UnitreeSdkStateError(RuntimeError):
    """The adapter was used outside its lifecycle contract.

    Raised for double initialization with conflicting arguments (the vendor
    channel factory is a process-wide singleton that would otherwise silently
    ignore the new configuration) and for channel creation before
    :meth:`unitree_g1_adapter.runtime.UnitreeSdk.initialize`.
    """


class SdkCallError(RuntimeError):
    """A vendor RPC client answered with a non-zero status code.

    ``code`` is the vendor status (or client-side error) value; ``timed_out``
    marks the vendor client-side timeout sentinel, which callers must treat as
    "fate unknown" — a motion request in that state is never retried (design
    decision recorded in AtomGit Issue #171, finalized in PR 2).
    """

    def __init__(self, *, api: str, code: int, timed_out: bool, detail: str = "") -> None:
        self.api = api
        self.code = code
        self.timed_out = timed_out
        self.detail = detail
        suffix = f" ({detail})" if detail else ""
        state = "timed out" if timed_out else f"failed with code {code}"
        super().__init__(f"vendor RPC {api} {state}{suffix}")
