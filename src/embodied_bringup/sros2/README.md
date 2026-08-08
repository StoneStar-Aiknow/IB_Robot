# Hermes runtime caller policy

Authorized embodied deployments require ROS 2 DDS Security in `Enforce` mode. Generate a deployment-specific keystore
from `caller_policy.xml`; do not commit the generated private keys.

```bash
ros2 security create_keystore /secure/ibrobot_keystore
ros2 security generate_artifacts \
  --keystore /secure/ibrobot_keystore \
  --policy caller_policy.xml
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
export ROS_SECURITY_KEYSTORE=/secure/ibrobot_keystore
```

Start Hermes/`robot-skill` with `ROS_SECURITY_ENCLAVE_OVERRIDE=/hermes_cli`. Catalog reload is an operator-only action;
run it from a separate process with `ROS_SECURITY_ENCLAVE_OVERRIDE=/operator`. Runtime nodes receive their enclave from
`embodied_pipeline.launch.py`.

The policy intentionally does not change standalone `robot_moveit` launches. Those remain engineering/debug entry
points and must not share an authorized production ROS domain.
