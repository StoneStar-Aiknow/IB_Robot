#include "cbf/distance_gradient.hpp"

#include <cmath>

namespace x2::cbf {
namespace {

struct JointRange {
  int first{-1};
  int last{-2};
};

JointRange ancestor_joint_range(PrimitiveId primitive) {
  switch (primitive) {
    case PrimitiveId::LeftShoulder:
      return {0, 1};
    case PrimitiveId::LeftForearm:
      return {0, 3};
    case PrimitiveId::LeftPalm:
    case PrimitiveId::LeftThumb:
      return {0, 6};
    case PrimitiveId::RightShoulder:
      return {7, 8};
    case PrimitiveId::RightForearm:
      return {7, 10};
    case PrimitiveId::RightPalm:
    case PrimitiveId::RightThumb:
      return {7, 13};
    default:
      return {};
  }
}

bool finite_vec3(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

}  // namespace

PointJacobian primitive_point_jacobian(const KinematicsOutput& kinematics,
                                       PrimitiveId primitive,
                                       const Vec3& point_world) {
  PointJacobian jacobian{};
  const JointRange range = ancestor_joint_range(primitive);
  for (int joint = range.first; joint <= range.last; ++joint) {
    const RevoluteJointFrame& frame =
        kinematics.arm_joint_frames[static_cast<std::size_t>(joint)];
    jacobian[static_cast<std::size_t>(joint)] =
        cross(frame.axis_world, point_world - frame.origin_world);
  }
  return jacobian;
}

DistanceGradientReport evaluate_distance_gradients(
    const KinematicsInput& input) {
  DistanceGradientReport report;
  report.kinematics = forward_kinematics(input);
  report.distances = evaluate_proxy_distances(report.kinematics);
  report.valid = report.kinematics.valid;

  for (std::size_t pair_index = 0; pair_index < kPrimitivePairCount;
       ++pair_index) {
    const PrimitivePair pair = kPrimitivePairs[pair_index];
    PairDistanceGradient& pair_report = report.pairs[pair_index];
    pair_report.distance = report.distances.distances[pair_index];
    if (!proxy_pair_enabled(report.kinematics, pair_index)) {
      pair_report.valid = true;
      continue;
    }
    const PointJacobian first_jacobian = primitive_point_jacobian(
        report.kinematics, pair.first, pair_report.distance.point_a);
    const PointJacobian second_jacobian = primitive_point_jacobian(
        report.kinematics, pair.second, pair_report.distance.point_b);

    pair_report.valid = pair_report.distance.valid &&
                        finite_vec3(pair_report.distance.normal);
    for (std::size_t joint = 0; joint < pair_report.gradient.size(); ++joint) {
      const Vec3 relative_velocity =
          second_jacobian[joint] - first_jacobian[joint];
      const double value = dot(pair_report.distance.normal, relative_velocity);
      pair_report.gradient[joint] = value;
      pair_report.gradient_squared_norm += value * value;
      pair_report.valid = pair_report.valid && std::isfinite(value);
    }
    report.valid = report.valid && pair_report.valid;
  }
  return report;
}

}  // namespace x2::cbf
