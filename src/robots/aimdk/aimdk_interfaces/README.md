# X2 Interfaces

Repository-owned extensions for the X2 MC bridge and standalone X2 MuJoCo
simulation. These are not cross-platform Robot Runtime contracts and are not
the vendor's externally supplied `aimdk_msgs` package.

- `GetMotionState`: raw X2 MC action and status identifiers.
- `LocomotionProfile` / `GetLocomotionProfiles`: X2 MC speed modes and bounds.
  Names correspond to the vendor response fields `none`, `low`, `medium`,
  `high`, and `step`, not a portable speed-profile enumeration.
- `ExecuteSimulationPolicy`: the X2 simulation's BMIMIC, MC-RL velocity, and
  recovery controller lifecycle. It is not a hardware execution action.

The bridge exposes queries under `/aimdk/mc/`; standalone simulation uses
`/simulation/`. Neither extension adds a generic `motion.*` capability.
