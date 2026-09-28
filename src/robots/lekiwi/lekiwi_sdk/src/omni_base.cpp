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

#include "lekiwi/omni_base.hpp"

#include <algorithm>
#include <cmath>

namespace lekiwi
{

std::array<double, 3> body_to_wheel_velocities(
  double vx, double vy, double vtheta, const BaseGeometry & geometry)
{
  // Matches the proven cmd_vel bridge kinematics.
  std::array<double, 3> wheels{};
  for (std::size_t i = 0; i < 3; ++i) {
    wheels[i] = (vx * std::cos(geometry.mount_angles[i]) +
                 vy * std::sin(geometry.mount_angles[i]) + vtheta * geometry.base_radius) /
                geometry.wheel_radius;
  }
  const double max_abs = std::max({std::abs(wheels[0]), std::abs(wheels[1]), std::abs(wheels[2])});
  if (max_abs > geometry.max_wheel_radps && max_abs > 0.0) {
    const double scale = geometry.max_wheel_radps / max_abs;
    for (double & wheel : wheels) {
      wheel *= scale;
    }
  }
  return wheels;
}

bool wheel_deltas_to_body(
  const std::array<double, 3> & deltas, const BaseGeometry & geometry,
  double & dx, double & dy, double & dtheta)
{
  // Least-squares pseudo-inverse of the wheel Jacobian (rows
  // [cos(a)/k, sin(a)/k, r/k]) — the proven cmd_vel bridge formulation.
  const double a0 = geometry.mount_angles[0];
  const double a1 = geometry.mount_angles[1];
  const double a2 = geometry.mount_angles[2];
  const double k = geometry.wheel_radius;
  const double r = geometry.base_radius;

  const double c0 = std::cos(a0), s0 = std::sin(a0);
  const double c1 = std::cos(a1), s1 = std::sin(a1);
  const double c2 = std::cos(a2), s2 = std::sin(a2);
  const double j11 = (c0 * c0 + c1 * c1 + c2 * c2) / (k * k);
  const double j12 = (c0 * s0 + c1 * s1 + c2 * s2) / (k * k);
  const double j13 = r * (c0 + c1 + c2) / (k * k);
  const double j22 = (s0 * s0 + s1 * s1 + s2 * s2) / (k * k);
  const double j23 = r * (s0 + s1 + s2) / (k * k);
  const double j33 = 3.0 * r * r / (k * k);

  const double b1 = (c0 * deltas[0] + c1 * deltas[1] + c2 * deltas[2]) / k;
  const double b2 = (s0 * deltas[0] + s1 * deltas[1] + s2 * deltas[2]) / k;
  const double b3 = r * (deltas[0] + deltas[1] + deltas[2]) / k;

  const double det = j11 * (j22 * j33 - j23 * j23) - j12 * (j12 * j33 - j23 * j13) +
                     j13 * (j12 * j23 - j22 * j13);
  if (std::abs(det) < 1e-12) {
    return false;
  }
  dx = (b1 * (j22 * j33 - j23 * j23) - j12 * (b2 * j33 - j23 * b3) +
        j13 * (b2 * j23 - j22 * b3)) / det;
  dy = (j11 * (b2 * j33 - j23 * b3) - b1 * (j12 * j33 - j23 * j13) +
        j13 * (j12 * b3 - b2 * j13)) / det;
  dtheta = (j11 * (j22 * b3 - b2 * j23) - j12 * (j12 * b3 - b1 * j23) +
            b1 * (j12 * j23 - j22 * j13)) / det;
  return true;
}

void integrate_pose(BasePose & pose, double dx, double dy, double dtheta)
{
  const double cos_t = std::cos(pose.theta);
  const double sin_t = std::sin(pose.theta);
  pose.x += dx * cos_t - dy * sin_t;
  pose.y += dx * sin_t + dy * cos_t;
  pose.theta = std::atan2(std::sin(pose.theta + dtheta), std::cos(pose.theta + dtheta));
}

OmniBase::OmniBase(
  feetech::Bus & bus, const std::vector<std::uint8_t> & wheel_ids, BaseGeometry geometry)
: bus_(bus), wheel_ids_(wheel_ids), geometry_(geometry)
{
}

bool OmniBase::activate()
{
  // Scoped to the wheels. The bus is shared with the arm on a full LeKiwi, and
  // apply_all_configs() would re-energize arm motors the arm had deliberately
  // released -- behind the arm SDK's back, leaving its lifecycle state
  // disagreeing with the hardware.
  const auto result = bus_.apply_configs(wheel_ids_);
  return result.ok;
}

bool OmniBase::set_wheel_velocities(const std::array<double, 3> & radps)
{
  // Scale down proportionally when any wheel exceeds the configured maximum
  // so the commanded direction is preserved.
  std::array<double, 3> commanded = radps;
  const double max_abs = std::max({std::abs(commanded[0]), std::abs(commanded[1]),
    std::abs(commanded[2])});
  if (max_abs > geometry_.max_wheel_radps && max_abs > 0.0) {
    const double scale = geometry_.max_wheel_radps / max_abs;
    for (double & wheel : commanded) {
      wheel *= scale;
    }
  }

  std::vector<feetech::MotorTarget> targets;
  targets.reserve(wheel_ids_.size());
  for (std::size_t i = 0; i < wheel_ids_.size(); ++i) {
    feetech::MotorTarget target;
    target.id = wheel_ids_[i];
    target.velocity = commanded[i];
    targets.push_back(target);
  }
  return bus_.sync_write_velocities(targets).ok;
}

bool OmniBase::set_body_velocity(double vx, double vy, double vtheta)
{
  return set_wheel_velocities(body_to_wheel_velocities(vx, vy, vtheta, geometry_));
}

bool OmniBase::read_wheels(
  std::array<double, 3> & positions, std::array<double, 3> & velocities)
{
  std::vector<feetech::MotorSample> samples;
  // Scope the read to the wheel motors. The bus is shared with the arm on a
  // full LeKiwi, and an unresponsive arm motor must not take the base's
  // odometry down with it.
  if (!bus_.sync_read(samples, wheel_ids_).ok) {
    return false;
  }
  // Map samples by motor id (registry order may differ from mount order).
  for (std::size_t i = 0; i < wheel_ids_.size(); ++i) {
    const std::uint8_t id = wheel_ids_[i];
    const auto sample = std::find_if(samples.begin(), samples.end(),
      [id](const feetech::MotorSample & s) { return s.id == id; });
    if (sample == samples.end()) {
      return false;
    }
    positions[i] = sample->position;
    velocities[i] = sample->velocity;
  }
  return true;
}

bool OmniBase::update_odometry(BasePose & pose)
{
  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  if (!read_wheels(positions, velocities)) {
    return false;
  }
  if (!odometry_seeded_) {
    last_positions_ = positions;
    odometry_seeded_ = true;
    return true;
  }

  std::array<double, 3> deltas{};
  for (std::size_t i = 0; i < 3; ++i) {
    deltas[i] = positions[i] - last_positions_[i];
  }
  last_positions_ = positions;

  double dx = 0.0, dy = 0.0, dtheta = 0.0;
  if (!wheel_deltas_to_body(deltas, geometry_, dx, dy, dtheta)) {
    return false;
  }
  integrate_pose(pose, dx, dy, dtheta);
  return true;
}

bool OmniBase::stop()
{
  return set_wheel_velocities({0.0, 0.0, 0.0});
}

const BaseGeometry & OmniBase::geometry() const { return geometry_; }

}  // namespace lekiwi
