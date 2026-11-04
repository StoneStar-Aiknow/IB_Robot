# aimdk_description — AgiBot X2 robot description

Official AgiBot X2 URDF and meshes, taken byte-for-byte from
[AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf) at a
pinned commit and licensed under the **Mulan Permissive Software License v2**
(`LICENSE`). `manifest.yaml` records the commit and every file's SHA-256.
Never edit these files: regenerate them.

```bash
scripts/sync_x2_description.py            # fetch the pinned commit and rewrite the package
scripts/sync_x2_description.py --check    # verify the files against manifest.yaml
```

## Contents

| File | Upstream | Why |
|---|---|---|
| `urdf/x2_ultra.urdf` | `X2_URDF-v1.3.0/x2_ultra.urdf` | X2 Ultra body. Its kinematics match the motion-control unit's own model on the verified robot exactly (same 40 joint origins); only 12 joint limits and the pelvis mass differ. |
| `urdf/x2_ultra_omnihand_v1.4.0.urdf` | `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf` | The only official OmniHand model. v1.3.0 ships no hand; the v1.3.0 and v1.4.0 wrist frames agree within a millimetre. |
| `meshes/v1.3.0/` | v1.3.0 meshes of the upper body and the left leg | pelvis, waist, torso, both arms, head; left hip and knee, which the grasp planner keeps the left hand away from |
| `meshes/v1.4.0/` | v1.4.0 meshes of the left OmniHand and both wrist roll links | palm, thumb, fingers; the v1.3.0 wrist roll meshes include the dummy hand fitted when no OmniHand is mounted |

The right leg, the feet and the IMU and camera housings are left out: grasp
planning does not use them. The two URDFs are kept verbatim for provenance and
still reference the upstream `./meshes/` layout, so they are not loaded directly:
the grasp planning model is generated from them by `aimdk_grasp`.

## Model version

The robot must be an **X2 Ultra** (nameplate without "-N"). An X2 Ultra -N uses
`X2_URDF-v1.4.0` throughout and needs its own description.

## Known deviation: OmniHand thumb mount

The official OmniHand places the palm with an extra ~90° about the wrist Z axis
relative to the AimDK SDK hand model (`agibot_x2_hand`). The palm and four
fingers of the official model coincide with the observation-corrected hand used
by the X2 grasp pipeline (median 1.3 mm, p95 4.9 mm), so that correction is the
official geometry. The thumb does not: its roll axis differs by ~75° between
the two models, and the SDK mount is the one the operator confirmed on the
robot. Consumers keep the files verbatim and apply the SDK thumb mount when
they generate a model (`aimdk_grasp.planning_model`).
