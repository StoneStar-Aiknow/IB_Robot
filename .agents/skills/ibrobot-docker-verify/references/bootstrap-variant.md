# Bootstrap Variant for ROS Installation Changes

The default desktop-full flow verifies dependency convergence and the complete
workspace build, but it cannot exercise the missing-ROS branch in
`install_ros.sh`. For changes to ROS installation, repository, or GPG-key
logic, repeat the same procedure with these differences:

```bash
IMAGE=ubuntu:22.04
```

- Keep the selected Phase 3 source mode and host pip cache mount.
- **Leave the host CUDA toolkit unmounted**, as in Phase 2, unless its CUDA
  version matches `torch.version.cuda`. A visible `nvcc` makes setup prefer a
  source build of `pointnet2_ops` and suppresses the Torch ABI pin, so a
  mismatched toolkit aborts setup; with none mounted the pin runs and the
  audited wheel installs. `--gpus` is not required either way.
- Let `setup.sh --yes --accel cuda` install ROS without manual intervention.
  Keep `--accel cuda` for the same reason as Phase 4: detection reads
  `nvidia-smi`, absent in a container without `--gpus`.
- Report the bootstrap result separately from the faster ROS-ready result.
- For release validation, optionally run with an empty pip cache directory to
  measure a true cold start.
