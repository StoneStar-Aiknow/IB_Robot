# object_tracker

This package owns single-target RGB-D tracking and collision-aware Nav2 following.

## Interfaces

- `target_tracker_node` publishes `ibrobot_msgs/msg/TrackState` and exposes the
  tracking lifecycle services.
- `dynamic_target_follower_node` consumes actionable `TrackState` messages in
  `odom`, transforms them to `map` using the localization TF contract, and
  follows a stand-off pose through `ComputePathToPose` and `FollowPath`.
- `EgoCompensatedMotionClassifier` classifies independent target motion from a
  window of timestamped `odom` positions. Camera-image motion caused only by
  robot translation or rotation remains stationary after timestamped TF
  compensation; covariance, displacement, speed, and confirmation windows gate
  the `MOVING` state.

The follower does not publish `/cmd_vel`. Nav2 remains the motion owner, so its
planner, costmaps, controller, and LeKiwi velocity bridge remain in the path.

The online semantic mapper may consume `/object_tracker/track_state`. Only a
known persistent semantic `object_id`, measured tracker state, acceptable age
and covariance, and repeated stable movement geometry can update the semantic
database. Stationary states refresh an already confirmed position. This updates
the semantic object layer and object version, not the SLAM map.
Dynamic obstacle marking and clearing remain owned by Nav2's existing `/scan`
source generated from aligned depth; tracker state is not written into the SLAM
occupancy map.

Both nodes are disabled by default. The follower also remains fail-closed when
the target is stale, uncertain, not actionable, or when either Nav2 action is
unavailable.

## Temporary integration mock

`mock_slam_nav_interfaces` provides a replaceable integration fixture until the
SLAM and navigation readiness contracts are finalized. It exposes
`/slam/readiness` and `/navigation/readiness` as `std_srvs/srv/Trigger`, and
publishes deterministic `map -> odom`, `odom -> base_link`, and camera TF data.
Enable `require_slam_readiness` and `require_navigation_readiness` on the
follower to exercise the fail-closed readiness gate. The service names are
parameters so a later 311 adapter can replace them without changing following
logic.
