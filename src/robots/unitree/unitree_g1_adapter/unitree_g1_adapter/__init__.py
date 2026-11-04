"""ROS-free Unitree G1 SDK adapter (AtomGit Issue #171, PR 1).

This package is the single dependency boundary between IB_Robot and the
official ``unitree_sdk2_python`` vendor SDK:

- :mod:`unitree_g1_adapter.vendor` is the ONLY module that imports a vendor symbol;
  everything else consumes its injectable surface, so the package imports and
  its tests run on hosts where the vendor SDK is not installed.
- :mod:`unitree_g1_adapter.channels` wraps the vendor pub/sub channels with
  last-sample caching and explicit close semantics.
- :mod:`unitree_g1_adapter.clients` wraps the vendor high-level RPC clients and
  maps their mixed return shapes onto one :class:`~unitree_g1_adapter.clients.SdkCallResult`.
- :mod:`unitree_g1_adapter.runtime` is the ``UnitreeSdk`` facade that owns factory
  initialization (the vendor channel factory is a process-wide singleton) and
  resource cleanup.

Deliberately out of scope here (they belong to the G1 runtime, PR 2): topic
names, joint index mappings, control weight ramps, stop/mode policies and any
Dex3/Revo2 hand semantics. This package exposes verified vendor primitives
only.
"""
