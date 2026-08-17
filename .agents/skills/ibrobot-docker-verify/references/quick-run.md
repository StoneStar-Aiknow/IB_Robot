# Quick-Run One-Liner (for Iterative Testing)

> **Prerequisite**: The container must have been created via the full Phase 2
> `docker run`. If a host CUDA toolkit was detected in Phase 0, it is mounted
> read-only and `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` are set so GraspGen
> `pointnet2_ops` can be compiled. If no CUDA toolkit is available, setup.sh
> will skip grasp install gracefully and the verification continues.

When iterating locally, repeat the matching Phase 3 source preparation rather
than changing source modes implicitly:

```bash
# Default local-copy mode: replace the previous container copy.
docker exec verify-ubuntu2204 rm -rf /home/testuser/IB_Robot
docker cp "${PROJECT_ROOT}" verify-ubuntu2204:/home/testuser/IB_Robot
docker exec verify-ubuntu2204 \
  chown -R "${HOST_UID}:${HOST_GID}" /home/testuser/IB_Robot

# Both modes: remove host or previous-run artifacts before setup.
docker exec verify-ubuntu2204 bash -c \
  'rm -rf /home/testuser/IB_Robot/{venv,build,install,log}'

# Re-run
docker exec -u testuser -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/setup.sh --yes \
    > /tmp/setup.log 2>&1'
```
