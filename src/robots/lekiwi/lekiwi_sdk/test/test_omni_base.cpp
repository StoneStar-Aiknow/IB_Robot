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
#include <chrono>
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

// Drive `base` at a fixed body velocity for `seconds` of simulated time,
// integrating odometry every `period`. Returns the x after every step.
std::vector<double> drive(
  feetech::Bus & bus, lekiwi::OmniBase & base, lekiwi::BasePose & pose,
  double vx, double vy, double vtheta, double seconds, std::chrono::milliseconds period)
{
  bus.sim().use_manual_clock();
  EXPECT_TRUE(base.set_body_velocity(vx, vy, vtheta));
  EXPECT_TRUE(base.update_odometry(pose));  // seed read, time origin
  std::vector<double> xs;
  const int steps = static_cast<int>(std::lround(seconds * 1000.0 / period.count()));
  for (int i = 0; i < steps; ++i) {
    bus.sim().advance_clock(period);
    EXPECT_TRUE(base.update_odometry(pose));
    xs.push_back(pose.x);
  }
  return xs;
}

TEST(OmniBase, OdometryAdvancesConsistentlyWithWheelFeedback)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());

  // A quarter second of pure forward drive at the runtime's 100 Hz read rate:
  // the pose advances by vx * t, bounded by the one-step-per-second command
  // quantisation of each wheel. (Short enough that no wheel crosses the
  // encoder wrap; OdometryIsContinuousAcrossEncoderWrap covers that.)
  lekiwi::BasePose pose;
  drive(bus, base, pose, 0.2, 0.0, 0.0, 0.25, std::chrono::milliseconds(10));
  EXPECT_NEAR(pose.x, 0.05, 1e-3);
  EXPECT_NEAR(pose.y, 0.0, 1e-3);
  EXPECT_NEAR(pose.theta, 0.0, 1e-3);
}

TEST(OmniBase, OdometryPureSpinAdvancesThetaOnly)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());

  lekiwi::BasePose pose;
  drive(bus, base, pose, 0.0, 0.0, 0.5, 0.25, std::chrono::milliseconds(10));
  EXPECT_NEAR(pose.x, 0.0, 1e-3);
  EXPECT_NEAR(pose.y, 0.0, 1e-3);
  EXPECT_NEAR(pose.theta, 0.125, 1e-3);
}

// Regression: odometry differenced raw encoder angles, so every wheel
// revolution injected a spurious full turn and the pose jumped. Five seconds
// at 0.2 m/s turns the wheels through several revolutions, from a start tick
// close to the wrap point so the first crossing comes early.
TEST(OmniBase, OdometryIsContinuousAcrossEncoderWrap)
{
  feetech::Bus bus(wheel_bus_options());
  ASSERT_TRUE(bus.open());
  ASSERT_TRUE(bus.apply_all_configs().ok);
  for (const std::uint8_t id : {7, 8, 9}) {
    bus.sim().set_position_ticks(id, 4000);
  }
  lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());

  lekiwi::BasePose pose;
  const auto xs = drive(bus, base, pose, 0.2, 0.0, 0.0, 5.0, std::chrono::milliseconds(20));
  for (std::size_t i = 1; i < xs.size(); ++i) {
    ASSERT_GT(xs[i], xs[i - 1]) << "odometry went backwards at step " << i;
    ASSERT_LT(xs[i] - xs[i - 1], 0.01) << "odometry jumped at step " << i;
  }
  EXPECT_NEAR(pose.x, 1.0, 1e-2);
  EXPECT_NEAR(pose.y, 0.0, 1e-2);
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

TEST(OmniBase, ProtectionPreservesOutputsAndOdometryHistory)
{
  for (int bits : {1, 2, 4, 8, 16, 32, 34}) {
    SCOPED_TRACE(bits);
    feetech::Bus bus(wheel_bus_options());
    ASSERT_TRUE(bus.open());
    ASSERT_TRUE(bus.apply_all_configs().ok);
    lekiwi::OmniBase base(bus, {7, 8, 9}, test_geometry());
    bus.sim().set_converge_step_ticks(0);
    lekiwi::BasePose pose;
    ASSERT_TRUE(base.update_odometry(pose));
    const auto original_pose = pose;
    std::array<double, 3> positions{11.0, 12.0, 13.0};
    std::array<double, 3> velocities{21.0, 22.0, 23.0};
    const auto original_positions = positions;
    const auto original_velocities = velocities;
    bus.sim().set_position_ticks(7, 900);
    bus.sim().set_position_ticks(9, 3000);
    bus.sim().set_response_status(9, static_cast<std::uint8_t>(bits));
    EXPECT_FALSE(base.read_wheels(positions, velocities));
    EXPECT_EQ(positions, original_positions);
    EXPECT_EQ(velocities, original_velocities);
    EXPECT_FALSE(base.update_odometry(pose));
    EXPECT_DOUBLE_EQ(pose.x, original_pose.x);
    EXPECT_DOUBLE_EQ(pose.y, original_pose.y);
    EXPECT_DOUBLE_EQ(pose.theta, original_pose.theta);
    // Return to the seed pose: rejected data must not replace last_positions_.
    bus.sim().set_position_ticks(7, 2048);
    bus.sim().set_position_ticks(9, 2048);
    bus.sim().set_response_status(9, 0);
    ASSERT_TRUE(base.update_odometry(pose));
    EXPECT_NEAR(pose.x, original_pose.x, 1e-9);
    EXPECT_NEAR(pose.y, original_pose.y, 1e-9);
    EXPECT_NEAR(pose.theta, original_pose.theta, 1e-9);
  }
}

}  // namespace

TEST(OmniBaseKinematics, WheelPositionDeltasTakeTheShortPathAcrossTheWrap)
{
  const double turn = 2.0 * M_PI;
  const double near_top = turn - 0.05;
  // Forward across the wrap: top of the range -> just past zero.
  const auto forward = lekiwi::wheel_position_deltas({0.03, 1.0, 0.0}, {near_top, 0.9, 0.0});
  EXPECT_NEAR(forward[0], 0.08, 1e-9);
  EXPECT_NEAR(forward[1], 0.1, 1e-9);
  EXPECT_NEAR(forward[2], 0.0, 1e-9);
  // Backward across the wrap: just past zero -> top of the range.
  const auto backward = lekiwi::wheel_position_deltas({near_top, 0.9, 0.0}, {0.03, 1.0, 0.0});
  EXPECT_NEAR(backward[0], -0.08, 1e-9);
  EXPECT_NEAR(backward[1], -0.1, 1e-9);
}

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
