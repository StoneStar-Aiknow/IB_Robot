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

// Lifecycle tests for the LeKiwi thin adapter on the SDK simulated transport.
// The runtime TORQUE_OFF stop reaches the SDK through
// set_hardware_component_state(inactive) -> on_deactivate, and clearing the
// stop re-enters on_activate WITHOUT on_configure: the shared bus must survive.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "lekiwi_hardware/lekiwi_system_hardware.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace lekiwi_hardware
{
struct LeKiwiHardwareTestAccess
{
  static feetech::Bus & bus(LeKiwiSystemHardware & hw) {return *hw.bus_;}
  static const std::vector<std::uint8_t> & arm_ids(const LeKiwiSystemHardware & hw)
  {return hw.arm_ids_;}
  static const std::vector<std::uint8_t> & wheel_ids(const LeKiwiSystemHardware & hw)
  {return hw.wheel_ids_;}
  static const std::vector<double> & commands(const LeKiwiSystemHardware & hw)
  {return hw.hw_commands_;}
  static const std::vector<double> & positions(const LeKiwiSystemHardware & hw)
  {return hw.hw_positions_;}
};
}  // namespace lekiwi_hardware

namespace
{

class TempCalibFile
{
public:
  TempCalibFile()
  {
    path_ = "/tmp/lekiwi_hardware_test_" + std::to_string(getpid()) + ".json";
    std::ofstream file(path_);
    file <<
      R"({
      "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "3": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "6": {"homing_offset": 0, "range_min": 0, "range_max": 4095}
    })";
  }
  ~TempCalibFile() {std::remove(path_.c_str());}
  const std::string & path() const {return path_;}

private:
  std::string path_;
};

hardware_interface::HardwareInfo make_info(
  const std::unordered_map<std::string, std::string> & params, bool with_arm)
{
  hardware_interface::HardwareInfo info;
  info.name = "LeKiwiSystem";
  info.type = "system";
  for (const auto & [key, value] : params) {
    info.hardware_parameters[key] = value;
  }
  auto add_joint = [&info](const std::string & name, int id, const char * cmd_if) {
      hardware_interface::ComponentInfo joint;
      joint.name = name;
      joint.parameters["id"] = std::to_string(id);
      hardware_interface::InterfaceInfo cmd;
      cmd.name = cmd_if;
      joint.command_interfaces.push_back(cmd);
      hardware_interface::InterfaceInfo pos;
      pos.name = "position";
      joint.state_interfaces.push_back(pos);
      hardware_interface::InterfaceInfo vel;
      vel.name = "velocity";
      joint.state_interfaces.push_back(vel);
      info.joints.push_back(joint);
    };
  if (with_arm) {
    for (int id = 1; id <= 6; ++id) {
      add_joint(std::to_string(id), id, "position");
    }
  }
  add_joint("wheel_left", 7, "velocity");
  add_joint("wheel_back", 8, "velocity");
  add_joint("wheel_right", 9, "velocity");
  return info;
}

TEST(LeKiwiHardwareAdapter, SimulatedFullLifecycleDeactivateReactivates)
{
  TempCalibFile calib;
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}, {"calib_file", calib.path()}}, true)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);

  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);

  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.on_cleanup(unused), hardware_interface::CallbackReturn::SUCCESS);
}

// Fail-closed: the arm activates first, so a wheel failure leaves it holding
// torque. Activation as a whole has failed, and a half-energized robot is
// exactly what fail-closed exists to prevent.
TEST(LeKiwiHardwareAdapter, WheelActivationFailureAlsoReleasesArmTorque)
{
  TempCalibFile calib;
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}, {"calib_file", calib.path()}}, true)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);

  auto & bus = lekiwi_hardware::LeKiwiHardwareTestAccess::bus(hw);
  const auto arm_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::arm_ids(hw);
  const auto wheel_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::wheel_ids(hw);
  ASSERT_FALSE(arm_ids.empty());
  ASSERT_FALSE(wheel_ids.empty());

  // One wheel motor stops acknowledging configuration writes, so the wheel
  // group fails to configure after the arm has already been energized.
  bus.sim().set_write_ack(wheel_ids.front(), false);

  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::ERROR);

  for (const std::uint8_t id : arm_ids) {
    EXPECT_FALSE(bus.sim().torque_enabled(id))
      << "arm motor " << static_cast<int>(id)
      << " still holds torque after a failed activation";
  }
}

// Regression: activation must hand the SDK's hold targets to ros2_control.
// Arm::activate() leaves the arm holding its measured pose, but hw_commands_
// starts at zero, so an unsynchronized buffer makes the very first write()
// command zero -- driving a raised arm toward the zero position while the
// runtime is still idle.
TEST(LeKiwiHardwareAdapter, ActivationSeedsCommandBufferFromHoldTargets)
{
  TempCalibFile calib;
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}, {"calib_file", calib.path()}}, true)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);

  auto & bus = lekiwi_hardware::LeKiwiHardwareTestAccess::bus(hw);
  const auto arm_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::arm_ids(hw);
  ASSERT_FALSE(arm_ids.empty());
  // Park the arm well away from the zero tick so "held" and "zeroed" differ,
  // and let the simulation converge fully on each read so a command really
  // moves the motor.
  bus.sim().set_converge_step_ticks(8192);
  for (const std::uint8_t id : arm_ids) {
    bus.sim().set_position_ticks(id, 2900);
  }

  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);

  const auto & commands = lekiwi_hardware::LeKiwiHardwareTestAccess::commands(hw);
  const auto & positions = lekiwi_hardware::LeKiwiHardwareTestAccess::positions(hw);
  for (size_t i = 0; i < arm_ids.size(); ++i) {
    // Both assertions are needed. An unseeded buffer leaves commands AND
    // positions at zero, so comparing them to each other alone would pass
    // vacuously; the magnitude check is what pins the parked pose.
    EXPECT_GT(std::abs(commands[i]), 0.1)
      << "arm joint " << i << " command buffer is still at its zero default";
    EXPECT_NEAR(commands[i], positions[i], 1e-3)
      << "arm joint " << i << " command buffer was not seeded from the hold target";
  }
  for (size_t i = arm_ids.size(); i < commands.size(); ++i) {
    EXPECT_DOUBLE_EQ(commands[i], 0.0) << "wheel command buffer " << i << " was not zeroed";
  }

  // The first write() must therefore be a no-op hold, not a move to zero.
  EXPECT_EQ(hw.write(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
  EXPECT_EQ(hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
  for (const std::uint8_t id : arm_ids) {
    EXPECT_NEAR(bus.sim().position_ticks(id), 2900, 8)
      << "arm motor " << static_cast<int>(id) << " was commanded away from its hold pose";
  }
}

// Re-entry replays nothing: inactive -> active runs on_activate() again without
// on_configure(), so a command left in the buffer by the previous activation
// must be replaced by the current hold target rather than re-sent.
TEST(LeKiwiHardwareAdapter, ReactivationReseedsInsteadOfReplayingStaleCommands)
{
  TempCalibFile calib;
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}, {"calib_file", calib.path()}}, true)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);

  auto & bus = lekiwi_hardware::LeKiwiHardwareTestAccess::bus(hw);
  const auto arm_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::arm_ids(hw);
  bus.sim().set_converge_step_ticks(0);
  for (const std::uint8_t id : arm_ids) {
    bus.sim().set_position_ticks(id, 1200);
  }
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);

  // A controller writes somewhere else, then the runtime stops the arm.
  auto interfaces = hw.export_command_interfaces();
  for (auto & interface : interfaces) {
    if (interface.get_interface_name() == hardware_interface::HW_IF_POSITION) {
      interface.set_value(2.0);
    }
  }
  ASSERT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);

  // The arm is moved by hand while released, then reactivated.
  for (const std::uint8_t id : arm_ids) {
    bus.sim().set_position_ticks(id, 3300);
  }
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);

  const auto & commands = lekiwi_hardware::LeKiwiHardwareTestAccess::commands(hw);
  const auto & positions = lekiwi_hardware::LeKiwiHardwareTestAccess::positions(hw);
  for (size_t i = 0; i < arm_ids.size(); ++i) {
    EXPECT_NEAR(commands[i], positions[i], 1e-3)
      << "arm joint " << i << " replayed the pre-deactivation command";
  }
  EXPECT_EQ(hw.write(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
  for (const std::uint8_t id : arm_ids) {
    EXPECT_NEAR(bus.sim().position_ticks(id), 3300, 8)
      << "arm motor " << static_cast<int>(id) << " moved to a stale target after reactivation";
  }
}

// Regression: a wheel that refuses to release torque must fail the stop, in
// both forms. on_deactivate() is what /runtime/stop's TORQUE_OFF rides on, so
// returning SUCCESS here tells the caller the robot is stopped. Base-only is
// the dangerous case: with no arm, an ignored wheel-release result left nothing
// that could fail the call.
TEST(LeKiwiHardwareAdapter, BaseOnlyDeactivationFailsWhenWheelReleaseIsRefused)
{
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}}, false)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);

  auto & bus = lekiwi_hardware::LeKiwiHardwareTestAccess::bus(hw);
  const auto wheel_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::wheel_ids(hw);
  ASSERT_FALSE(wheel_ids.empty());
  bus.sim().set_write_ack(wheel_ids.front(), false);

  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::ERROR)
    << "a refused wheel torque release was reported as a successful stop";
}

TEST(LeKiwiHardwareAdapter, FullRobotDeactivationFailsWhenWheelReleaseIsRefused)
{
  TempCalibFile calib;
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}, {"calib_file", calib.path()}}, true)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);

  auto & bus = lekiwi_hardware::LeKiwiHardwareTestAccess::bus(hw);
  const auto wheel_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::wheel_ids(hw);
  const auto arm_ids = lekiwi_hardware::LeKiwiHardwareTestAccess::arm_ids(hw);
  bus.sim().set_write_ack(wheel_ids.front(), false);

  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::ERROR)
    << "a refused wheel torque release was reported as a successful stop";
  // Best effort, not fail fast: the arm can still be released, so it must be.
  for (const std::uint8_t id : arm_ids) {
    EXPECT_FALSE(bus.sim().torque_enabled(id))
      << "arm motor " << static_cast<int>(id)
      << " kept torque because the wheel release failed first";
  }
}

TEST(LeKiwiHardwareAdapter, SimulatedBaseOnlyLifecycle)
{
  lekiwi_hardware::LeKiwiSystemHardware hw;
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(
    hw.on_init(make_info({{"simulated", "true"}}, false)),
    hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.on_shutdown(unused), hardware_interface::CallbackReturn::SUCCESS);
}

}  // namespace
