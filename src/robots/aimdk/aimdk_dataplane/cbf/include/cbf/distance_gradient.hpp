#pragma once

#include "cbf/kinematics.hpp"

#include <array>

namespace x2::cbf {

using ArmGradient = std::array<double, 14>;
using PointJacobian = std::array<Vec3, 14>;

struct PairDistanceGradient {
  DistanceResult distance{};
  ArmGradient gradient{};
  double gradient_squared_norm{0.0};
  bool valid{false};
};

struct DistanceGradientReport {
  KinematicsOutput kinematics{};
  PrimitiveDistanceReport distances{};
  std::array<PairDistanceGradient, kPrimitivePairCount> pairs{};
  bool valid{false};
};

PointJacobian primitive_point_jacobian(const KinematicsOutput& kinematics,
                                       PrimitiveId primitive,
                                       const Vec3& point_world);

DistanceGradientReport evaluate_distance_gradients(
    const KinematicsInput& input);

}  // namespace x2::cbf
