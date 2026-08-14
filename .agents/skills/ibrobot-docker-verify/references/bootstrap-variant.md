# Bootstrap Variant for ROS Installation Changes

The default desktop-full flow verifies dependency convergence and the complete
workspace build, but it cannot exercise the missing-ROS branch in
`install_ros.sh`. For changes to ROS installation, repository, or GPG-key
logic, repeat the same procedure with these differences:

```bash
IMAGE=ubuntu:22.04
```

- Keep the selected Phase 3 source mode and host pip cache mount.
- Let `setup.sh --yes` install ROS without manual intervention.
- Report the bootstrap result separately from the faster ROS-ready result.
- For release validation, optionally run with an empty pip cache directory to
  measure a true cold start.
