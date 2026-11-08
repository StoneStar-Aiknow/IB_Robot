#pragma once

#include "cbf/proxy_model.hpp"

#include <array>

namespace x2::cbf {

struct RevoluteJointFrame {
  Vec3 origin_world{};
  Vec3 axis_world{1.0, 0.0, 0.0};
};

struct KinematicsInput {
  std::array<double, 14> q_arm{};
  std::array<double, 6> q_thumb_urdf{};
  bool thumb_pose_valid{false};
  std::array<bool, 2> hands_installed{true, true};
};

struct KinematicsOutput {
  std::array<Transform, 12> body_transforms{};
  std::array<RevoluteJointFrame, 14> arm_joint_frames{};
  std::array<ProxyPrimitive, kProxyPrimitiveCount> primitives{};
  bool valid{false};
  std::array<bool, 2> hands_installed{true, true};
};

KinematicsOutput forward_kinematics(const KinematicsInput& input);
bool proxy_pair_enabled(const KinematicsOutput& kinematics,
                        std::size_t pair_index);
PrimitiveDistanceReport evaluate_proxy_distances(
    const KinematicsOutput& kinematics);
PrimitiveDistanceReport evaluate_proxy_distances(const KinematicsInput& input);

}  // namespace x2::cbf
