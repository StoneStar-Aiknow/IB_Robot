// Copyright 2026 IB_Robot Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LEKIWI__OMNI_BASE_HPP_
#define LEKIWI__OMNI_BASE_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "feetech/bus.hpp"
#include "feetech/types.hpp"

namespace lekiwi
{

/// Wheel mounting and geometry (the proven LeKiwi values live in robot
/// configuration; the SDK takes them as parameters).
struct BaseGeometry
{
  /// Wheel radius in meters.
  double wheel_radius = 0.05;
  /// Distance from base center to wheel contact point, meters.
  double base_radius = 0.125;
  /// Maximum wheel angular velocity, rad/s.
  double max_wheel_radps = 4.602;
  /// Wheel mount angles around the base, radians, CCW from base +X.
  std::array<double, 3> mount_angles = {
    (240.0 * M_PI / 180.0) - (M_PI / 2.0),
    0.0 - (M_PI / 2.0),
    (120.0 * M_PI / 180.0) - (M_PI / 2.0)};
};

/// Odometry pose estimate (planar).
struct BasePose
{
  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;
};

/// Pure kinematics (no bus): body velocity (vx, vy, vtheta) -> wheel angular
/// velocities (rad/s), scaled down proportionally when any wheel would exceed
/// geometry.max_wheel_radps so the commanded direction is preserved.
std::array<double, 3> body_to_wheel_velocities(
  double vx, double vy, double vtheta, const BaseGeometry & geometry);

/// Pure kinematics (no bus): wheel angular deltas or velocities (rad or rad/s)
/// -> body displacement or velocity (dx, dy, dtheta) via the least-squares
/// pseudo-inverse of the wheel Jacobian. Returns false when the geometry is
/// degenerate.
bool wheel_deltas_to_body(
  const std::array<double, 3> & wheel_deltas, const BaseGeometry & geometry,
  double & dx, double & dy, double & dtheta);

/// Pure kinematics (no bus): integrate a body-frame displacement into a
/// world-frame pose.
void integrate_pose(BasePose & pose, double dx, double dy, double dtheta);

/// Omni-wheel base on a Feetech bus: velocity commands in rad/s, wheel
/// feedback, and wheel-derived odometry.
class OmniBase
{
public:
  /// Construct on a caller-owned bus. `wheel_ids` are the three wheel motor
  /// ids in mount-angle order (left, back, right by convention).
  OmniBase(
    feetech::Bus & bus, const std::vector<std::uint8_t> & wheel_ids,
    BaseGeometry geometry);

  /// Configure the wheel motors (wheel mode). Fails with the offending motor
  /// on any configuration failure (the bus performs the rollback).
  bool activate();

  /// Command wheel velocities in rad/s (mount-angle order). Values beyond
  /// the configured maximum are scaled down proportionally (preserving the
  /// commanded direction).
  bool set_wheel_velocities(const std::array<double, 3> & radps);

  /// Body-frame velocity command (vx, vy [m/s], vtheta [rad/s]) converted to
  /// wheel velocities by the configured geometry.
  bool set_body_velocity(double vx, double vy, double vtheta);

  /// Read wheel feedback (position [rad accumulated], velocity [rad/s]).
  /// Returns false on bus failure (stale values never presented as fresh).
  bool read_wheels(std::array<double, 3> & positions, std::array<double, 3> & velocities);

  /// Integrate wheel feedback into `pose` since the previous update. Returns
  /// false when the underlying wheel read fails (pose is not advanced).
  bool update_odometry(BasePose & pose);

  /// Command zero wheel velocities.
  bool stop();

  const BaseGeometry & geometry() const;

private:
  feetech::Bus & bus_;
  std::vector<std::uint8_t> wheel_ids_;
  BaseGeometry geometry_;
  std::array<double, 3> last_positions_{};
  bool odometry_seeded_ = false;
};

}  // namespace lekiwi

#endif  // LEKIWI__OMNI_BASE_HPP_
