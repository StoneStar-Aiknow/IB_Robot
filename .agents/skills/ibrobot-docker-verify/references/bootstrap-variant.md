# Bootstrap Variant for ROS Installation Changes

The default desktop-full flow verifies dependency convergence and the complete
workspace build, but it cannot exercise the missing-ROS branch in
`install_ros.sh`. For changes to ROS installation, repository, or GPG-key
logic, repeat the same procedure with these differences:

```bash
IMAGE=ubuntu:22.04
```

- Keep the selected Phase 3 source mode and host pip cache mount.
- The host CUDA toolkit mount from Phase 2 is optional: without GPU
  passthrough, GraspGen and `pointnet2_ops` install from audited wheels, and
  `--gpus` is not required for setup+build verification.
- Let `setup.sh --yes --accel cuda` install ROS without manual intervention.
  Keep `--accel cuda` for the same reason as Phase 4: detection reads
  `nvidia-smi`, absent in a container without `--gpus`.
- Report the bootstrap result separately from the faster ROS-ready result.
- For release validation, optionally run with an empty pip cache directory to
  measure a true cold start.
