# Troubleshooting

## Nodes Cannot Discover Each Other

Confirm every process uses identical `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION`. For distributed
operation, also confirm `ROS_LOCALHOST_ONLY=0` and network reachability.

## `ModuleNotFoundError: lerobot`

- Ubuntu/openEuler: source `.shrc_local` in the same shell.
- OpenHarmony: source `/data/roboframe/scripts/robooh_1.0.1.env`; if the staged LeRobot tree is
  absent, rebuild and redeploy the official release rather than copying source manually.

## Package or Launch File Not Found

- Ubuntu/openEuler: build first, then source `install/setup.zsh` in the launch shell.
- OpenHarmony: verify `/data/roboframe/install` and the package with `ros2 pkg list`. Rebuild with
  the default `oh-build-roboframe` package set if it is missing.

## OpenHarmony Native Process Crashes or RKNN Import Fails

Confirm the board env was loaded, the artifact is aarch64/musl compatible, the deployment exists in
`inference_manifest.json`, and the required NPU runtime is installed. For RKNN, verify:

```sh
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

## Controller or Hardware Device Missing

Check the selected control mode, YAML paths, calibration, permissions, `/dev/ttyACM*`, camera
devices, and required OpenHarmony kernel configuration before changing launch logic.
