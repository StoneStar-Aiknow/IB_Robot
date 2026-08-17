# Quick-Run (Iterative Testing)

When only scripts changed (ROS 2 + system deps already installed), copy
updated files and re-run without recreating the container:

```bash
# Copy changed files into rootfs
docker cp scripts/setup.sh verify-oee:/root/openeuler_rootfs/root/IB_Robot/scripts/
docker cp scripts/setup/platforms/openeuler-embedded-24.03.sh \
  verify-oee:/root/openeuler_rootfs/root/IB_Robot/scripts/setup/platforms/
docker cp scripts/setup/lerobot_patches.sh \
  verify-oee:/root/openeuler_rootfs/root/IB_Robot/scripts/setup/

# Clean and re-run
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
docker exec -d verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "cd /root/IB_Robot && IBR_LEROBOT_FORCE_REBUILD=1 bash scripts/setup.sh --yes --no-sudo > /tmp/setup.log 2>&1"'
```
