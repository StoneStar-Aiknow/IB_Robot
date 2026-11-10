# robot_client

Reusable client foundation for workflow preparation and ROS transport. The
package intentionally has no dependencies on `robot_skill_cli`, `skill_catalog`,
`robot_config`, or any compiler CLI: callers resolve the deployment themselves
and pass the result in.

## Deployment binding

The client has no discovery, selection or credential step. A caller that owns a
deployment resolves the robot configuration at its own boundary and builds one
neutral transport for the whole process lifetime:

```python
from robot_client import RobotClient, gateway_transport

transport = gateway_transport({"skill_gateway_status_service": "/lab/status"},
                              robot_id="lab_g1", robot_name="lab_g1")
with RobotClient(transport) as client:
    status = client.get_status()
```

`gateway_transport()` performs no file IO; it only applies the documented
endpoint defaults to an already-loaded `embodied` mapping. `GatewayTransport`
exposes the normalized endpoint map, a canonical `identity_digest()`, and the
robot/caller identity. The client derives its `target_key` from that digest plus
the local ROS environment scope (`ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, security
settings, ...), so two deployments that resolve to different endpoints, or that
share endpoints on different ROS graphs, stay distinguishable.

A request never silently retargets: saved execution references carry
`target_key`, `robot_id` and `caller_scope`, and a reference from another
deployment is rejected before it reaches the Gateway. The digest is a routing
identity, not cryptographic authentication; keep using the deployment's ROS
security and trusted network boundary.

## Caller scope

The standard local operator uses the trusted ROS graph without a credential file
or token environment variable: `caller_scope="robot_skill_cli"`, an empty caller
token, and `read_only=false`. When the deployment leaves
`execution_caller_tokens_json` without a `robot_skill_cli` entry, the Gateway
keeps the empty token for this scope, so stale operator environment tokens do
not grant additional access. If the deployment configures an explicit
`robot_skill_cli` token, the Gateway keeps enforcing it and the CLI forwards the
documented `IBROBOT_EXECUTION_TOKEN_ROBOT_SKILL_CLI` value per session.

Integrations that need a different caller scope pass one explicitly to
`gateway_transport(caller_scope=...)` and rely on the Gateway's configured
authorization for that scope. Never put a token in a transport, selection file,
execution reference, browser storage, or command-line argument. Binding a
transport does not authorize motion.

## Preparation and execution

`prepare_workflow()` accepts typed `workflow_steps`, an audit `raw_command`, and
an optional caller-owned `request_id`. It returns an exact preview. The caller
must display it before calling `submit_prepared()` with a `presenter` callback
and a durable `persist_reference` callback. Both callbacks must complete before
any goal is sent. A callback exception prevents transmission. Do not retry a
submission whose outcome is unknown; query its saved reference instead.

The preview records the Gateway's business identity (`robot_name`,
`registry_epoch`, `registry_generation`, digests). `submit_prepared()` re-reads
it: a Gateway restart (new registry epoch) or a catalog change (new generation or
digest) invalidates the preview before anything is displayed or sent. The plan
itself is stored inside the Gateway process, so a restart also makes the prepared
plan unconfirmable.

`get_execution()`, `get_execution_events()`, and `cancel_execution()` use saved
references, not the skill catalog. References bind the deployment target, caller
scope, and Program request ID. Legacy action task IDs are not interchangeable
with Program request IDs. Accepted cancellation is only a request; only the
Gateway ledger establishes a terminal result.

Closing the client never cancels or replays a task. An unresolved task blocks new
preparations until it is observed or cancelled. A Gateway restart invalidates
previews, not saved execution references.

## Verification

```bash
source .shrc_local
colcon test --merge-install --packages-select robot_client
colcon test-result --test-result-base build/robot_client --verbose
```

Tests bind explicit transports and mock services. They do not send hardware
motion requests. Real robot acceptance is a separate operator-controlled task.
