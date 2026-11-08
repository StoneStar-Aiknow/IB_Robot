#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace x2::cbf {

inline constexpr std::array<std::size_t, 4> kRigidThumbIndices{0, 1, 3, 4};
inline constexpr std::array<std::array<double, 2>, 4> kRigidThumbLimits{{
    {0.03490658, 1.27409035},
    {-1.6424, 0.045379},
    {-0.5235987, 0.69813170},
    {-1.464235482612666, 0.04537856055185257},
}};
inline constexpr std::array<double, 6> kRigidThumbBaselineUrdf{
    0.654498465, -0.7985105, 0.0,
    0.0872665, -0.7094284610304067, 0.0,
};

inline std::array<double, 6> rigid_thumb_pose(std::array<double, 6> pose) {
  pose[2] = 0.0;
  pose[5] = 0.0;
  return pose;
}

inline bool valid_rigid_thumb_pose(
    const std::array<double, 6>& pose,
    const std::array<std::array<double, 2>, 4>& bounds = kRigidThumbLimits) {
  for (std::size_t axis = 0; axis < kRigidThumbIndices.size(); ++axis) {
    const double angle = pose[kRigidThumbIndices[axis]];
    const double lower = bounds[axis][0];
    const double upper = bounds[axis][1];
    const double span = upper - lower;
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper ||
        !std::isfinite(span) || span > 6.283185307179586 ||
        !std::isfinite(angle) || angle < lower - 1e-12 || angle > upper + 1e-12) {
      return false;
    }
  }
  return true;
}

}
