# Bootstrap Variant for ROS Installation Changes

The default desktop-full flow verifies dependency convergence and the complete
workspace build, but it cannot exercise the missing-ROS branch in
`install_ros.sh`. For changes to ROS installation, repository, or GPG-key
logic, repeat the same procedure with these differences:

```bash
IMAGE=ubuntu:22.04
```

- Keep the selected Phase 3 source mode and host pip cache mount.
- **Keep the host CUDA toolkit mount** from Phase 2 (if available). The
  bootstrap variant can still compile `pointnet2_ops` when `nvcc` is mounted,
  but `--gpus` is not required for setup+build verification. If no host
  CUDA toolkit is available, setup.sh will skip grasp install gracefully.
- Let `setup.sh --yes` install ROS without manual intervention.
- Report the bootstrap result separately from the faster ROS-ready result.
- For release validation, optionally run with an empty pip cache directory to
  measure a true cold start.
