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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "feetech/bus.hpp"
#include "lekiwi/omni_base.hpp"

namespace
{

lekiwi::BaseGeometry test_geometry()
{
  lekiwi::BaseGeometry geometry;
  geometry.wheel_radius = 0.05;
  geometry.base_radius = 0.125;
  geometry.max_wheel_radps = 10.0;
  return geometry;
}

feetech::BusOptions wheel_bus_options()
{
  feetech::BusOptions options;
  options.simulated = true;
  for (const std::uint8_t id : {7, 8, 9}) {
    feetech::MotorConfig motor;
    motor.id = id;
    motor.mode = feetech::Mode::Wheel;
    options.motors.push_back(motor);
  }
  return options;
}

// A full LeKiwi: six arm motors and three wheel motors on ONE bus.
feetech::BusOptions shared_bus_options()
{
  feetech::BusOptions options;
  options.simulated = true;
  for (std::uint8_t id = 1; id <= 6; ++id) {
    feetech::MotorConfig motor;
    motor.id = id;
    motor.mode = feetech::Mode::Position;
    options.motors.push_back(motor);
  }
  for (const std::uint8_t id : {7, 8, 9}) {
    feetech::MotorConfig motor;
    motor.id = id;
    motor.mode = feetech::Mode::Wheel;
    options.motors.push_back(motor);
  }
  return options;
}

TEST(OmniBase, VelocityCommandRoundTrip)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());

  const std::array<double, 3> commanded = {1.0, -0.5, 0.25};
  ASSERT_TRUE(base.set_wheel_velocities(commanded));

  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  ASSERT_TRUE(base.read_wheels(positions, velocities));
  // Feedback reflects the commanded directions and magnitudes after settling
  // (the simulation advances each wheel by its commanded speed per read).
  // Tick quantisation bounds the error to one step (~0.0015 rad/s).
  EXPECT_NEAR(velocities[0], 1.0, 0.002);
  EXPECT_NEAR(velocities[1], -0.5, 0.002);
  EXPECT_NEAR(velocities[2], 0.25, 0.002);
}

// Regression: the base must read only its own motors. On a full LeKiwi the arm
// shares this bus, and an unresponsive arm motor used to fail the whole group
// read, taking wheel feedback and odometry down with it.
TEST(OmniBase, SharedBusReadIgnoresUnresponsiveArmMotor)
{
  feetech::Bus bus(shared_bus_options());
  ASSERT_TRUE(bus.open());
  const std::vector<std::uint8_t> wheels = {7, 8, 9};
  ASSERT_TRUE(bus.apply_configs(wheels).ok);
  lekiwi::OmniBase base(bus, wheels, test_geometry());

  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  ASSERT_TRUE(base.read_wheels(positions, velocities));

  // An arm motor drops off the bus entirely.
  bus.sim().set_responsive(3, false);
  EXPECT_TRUE(base.read_wheels(positions, velocities))
    << "an unresponsive arm motor must not fail the base's scoped read";

  // A wheel motor dropping off still fails, as it must.
  bus.sim().set_responsive(8, false);
  EXPECT_FALSE(base.read_wheels(positions, velocities));
}

// Regression: activating the base must not re-energize the arm. On a shared
// bus apply_all_configs() would write torque-enable to every registered motor,
// silently undoing an arm TorqueOff behind the arm SDK's back.
TEST(OmniBase, ActivateLeavesArmMotorsReleased)
{
  feetech::Bus bus(shared_bus_options());
  ASSERT_TRUE(bus.open());
  const std::vector<std::uint8_t> wheels = {7, 8, 9};
  const std::vector<std::uint8_t> arm = {1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(bus.apply_all_configs().ok);
  ASSERT_TRUE(bus.emergency_release(arm).ok);
  for (const std::uint8_t id : arm) {
    ASSERT_FALSE(bus.sim().torque_enabled(id)) << "precondition: arm motor " << +id << " released";
  }

  lekiwi::OmniBase base(bus, wheels, test_geometry());
  ASSERT_TRUE(base.activate());

  for (const std::uint8_t id : arm) {
    EXPECT_FALSE(bus.sim().torque_enabled(id))
      << "base activation re-enabled torque on arm motor " << +id;
  }
  for (const std::uint8_t id : wheels) {
    EXPECT_TRUE(bus.sim().torque_enabled(id))
      << "base activation did not enable wheel motor " << +id;
  }
}

TEST(OmniBase, StopCommandsZeroVelocities)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());

  ASSERT_TRUE(base.set_wheel_velocities({2.0, 2.0, 2.0}));
  ASSERT_TRUE(base.stop());
  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  ASSERT_TRUE(base.read_wheels(positions, velocities));
  EXPECT_NEAR(velocities[0], 0.0, 1e-9);
  EXPECT_NEAR(velocities[1], 0.0, 1e-9);
  EXPECT_NEAR(velocities[2], 0.0, 1e-9);
}

TEST(OmniBase, OverspeedCommandScalesProportionally)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  lekiwi::BaseGeometry geometry = test_geometry();
  geometry.max_wheel_radps = 4.0;
  lekiwi::OmniBase base(bus, {7, 8, 9}, geometry);

  // 2x over the limit on one wheel: all wheels scale by 0.5, direction kept.
  ASSERT_TRUE(base.set_wheel_velocities({8.0, 4.0, -2.0}));
  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  ASSERT_TRUE(base.read_wheels(positions, velocities));
  EXPECT_NEAR(velocities[0], 4.0, 0.002);
  EXPECT_NEAR(velocities[1], 2.0, 0.002);
  EXPECT_NEAR(velocities[2], -1.0, 0.002);
}

TEST(OmniBase, BodyVelocityUsesConfiguredGeometry)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  const lekiwi::BaseGeometry geometry = test_geometry();
  lekiwi::OmniBase base(bus, {7, 8, 9}, geometry);

  const auto expected_wheel = [&](double vx, double vy, double vtheta) {
    std::array<double, 3> expected{};
    for (std::size_t i = 0; i < 3; ++i) {
      expected[i] = (vx * std::cos(geometry.mount_angles[i]) +
                     vy * std::sin(geometry.mount_angles[i]) + vtheta * geometry.base_radius) /
                    geometry.wheel_radius;
    }
    return expected;
  };

  // Pure forward motion: wheel speeds follow each mount angle's projection.
  ASSERT_TRUE(base.set_body_velocity(0.1, 0.0, 0.0));
  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  ASSERT_TRUE(base.read_wheels(positions, velocities));
  const auto forward = expected_wheel(0.1, 0.0, 0.0);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(velocities[i], forward[i], 0.002);
  }

  // Pure rotation: every wheel spins at vtheta * base_radius / wheel_radius.
  ASSERT_TRUE(base.set_body_velocity(0.0, 0.0, 0.5));
  ASSERT_TRUE(base.read_wheels(positions, velocities));
  const auto spin = expected_wheel(0.0, 0.0, 0.5);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(velocities[i], spin[i], 0.002);
  }
}

TEST(OmniBase, OdometryAdvancesConsistentlyWithWheelFeedback)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  const lekiwi::BaseGeometry geometry = test_geometry();
  lekiwi::OmniBase base(bus, {7, 8, 9}, geometry);

  // Pure forward drive: pose must advance along +x after odometry seeding.
  // The pseudo-inverse recovers the commanded body velocity per read step
  // (each simulated read advances the wheels by one step of the commanded
  // speed), so one integrated step equals vx.
  ASSERT_TRUE(base.set_body_velocity(0.2, 0.0, 0.0));
  lekiwi::BasePose pose;
  ASSERT_TRUE(base.update_odometry(pose));  // seed read
  ASSERT_TRUE(base.update_odometry(pose));  // first integrated step

  // One integrated step recovers the commanded body velocity, bounded by the
  // one-tick command quantisation (~1.5e-3 rad/s per wheel).
  EXPECT_GT(pose.x, 0.0);
  EXPECT_NEAR(pose.x, 0.2, 1e-3);
  EXPECT_NEAR(pose.y, 0.0, 1e-3);
  EXPECT_NEAR(pose.theta, 0.0, 1e-3);
}

TEST(OmniBase, OdometryPureSpinAdvancesThetaOnly)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  const lekiwi::BaseGeometry geometry = test_geometry();
  lekiwi::OmniBase base(bus, {7, 8, 9}, geometry);

  ASSERT_TRUE(base.set_body_velocity(0.0, 0.0, 0.5));
  lekiwi::BasePose pose;
  ASSERT_TRUE(base.update_odometry(pose));  // seed
  ASSERT_TRUE(base.update_odometry(pose));  // step
  EXPECT_NEAR(pose.x, 0.0, 1e-3);
  EXPECT_NEAR(pose.y, 0.0, 1e-3);
  EXPECT_NEAR(pose.theta, 0.5, 1e-3);
}

TEST(OmniBase, ReadFailureIsExplicitAndPoseNotAdvanced)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());

  lekiwi::BasePose pose;
  ASSERT_TRUE(base.update_odometry(pose));  // seed
  const lekiwi::BasePose before = pose;

  // Persistent unresponsiveness (a one-shot injection would be consumed by
  // the first failing read).
  bus.sim().set_responsive(7, false);
  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  EXPECT_FALSE(base.read_wheels(positions, velocities));
  EXPECT_FALSE(base.update_odometry(pose));
  EXPECT_DOUBLE_EQ(pose.x, before.x);
  EXPECT_DOUBLE_EQ(pose.y, before.y);
  EXPECT_DOUBLE_EQ(pose.theta, before.theta);
}

}  // namespace

TEST(OmniBaseKinematics, PureFunctionsRoundTripWithoutBus)
{
  lekiwi::BaseGeometry geometry;
  // Pure forward: body velocity -> wheels; pure inverse: wheels -> body.
  const auto wheels = lekiwi::body_to_wheel_velocities(0.2, -0.1, 0.3, geometry);
  double vx = 0.0, vy = 0.0, vtheta = 0.0;
  ASSERT_TRUE(lekiwi::wheel_deltas_to_body(wheels, geometry, vx, vy, vtheta));
  EXPECT_NEAR(vx, 0.2, 1e-9);
  EXPECT_NEAR(vy, -0.1, 1e-9);
  EXPECT_NEAR(vtheta, 0.3, 1e-9);

  // Overspeed scaling preserves direction.
  const auto fast = lekiwi::body_to_wheel_velocities(10.0, 0.0, 0.0, geometry);
  double max_abs = 0.0;
  for (double w : fast) {
    max_abs = std::max(max_abs, std::abs(w));
  }
  EXPECT_NEAR(max_abs, geometry.max_wheel_radps, 1e-9);

  // Pose integration rotates body displacement into the world frame.
  lekiwi::BasePose pose;
  pose.theta = M_PI / 2.0;
  lekiwi::integrate_pose(pose, 1.0, 0.0, 0.0);
  EXPECT_NEAR(pose.x, 0.0, 1e-9);
  EXPECT_NEAR(pose.y, 1.0, 1e-9);
}
