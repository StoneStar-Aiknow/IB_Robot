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

#ifndef LEKIWI_HARDWARE__LEKIWI_SYSTEM_HARDWARE_HPP_
#define LEKIWI_HARDWARE__LEKIWI_SYSTEM_HARDWARE_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "feetech/bus.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "so101/arm.hpp"

namespace lekiwi_hardware
{

// Full LeKiwi robot dimensions
static constexpr size_t FULL_ARM_JOINTS = 6;
static constexpr size_t FULL_BASE_JOINTS = 3;
static constexpr size_t FULL_JOINTS = FULL_ARM_JOINTS + FULL_BASE_JOINTS;

// Thin ros2_control adapter over feetech_sdk/so101_sdk. One shared bus
// carries the arm (position) and wheel (velocity) motors; arm operations
// are scoped to the arm motor set, wheel operations to the wheel set, so
// one subsystem's failures do not affect the other.
//
// Relocated coverage (formerly tested by test_lekiwi_conversions.cpp —
// the pure-math conversion module remains in-tree for the ros2_control
// path duration; SDK conversions live in feetech_sdk test_conversion.cpp).
class LeKiwiSystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(LeKiwiSystemHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  friend struct LeKiwiHardwareTestAccess;
  std::string port_;
  std::string calib_file_;
  bool simulated_ = false;
  void teardown_bus();
  bool base_only_mode_ = false;
  size_t num_arm_joints_ = 0;
  size_t num_base_joints_ = 0;
  size_t num_joints_ = 0;

  // Shared bus + scoped subsystem handles.
  std::unique_ptr<feetech::Bus> bus_;
  std::unique_ptr<so101::Arm> arm_;  // null in base-only mode
  std::vector<std::uint8_t> arm_ids_;
  std::vector<std::uint8_t> wheel_ids_;

  // State buffers for all joints (ros2_control interface order).
  std::vector<double> hw_positions_;   // arm rad + wheel accumulated rad
  std::vector<double> hw_velocities_;  // arm rad/s + wheel rad/s
  // Command buffers: arm position (rad), wheel velocity (rad/s).
  std::vector<double> hw_commands_;
  std::map<std::string, double> arm_command_map_;
};

}  // namespace lekiwi_hardware

#endif  // LEKIWI_HARDWARE__LEKIWI_SYSTEM_HARDWARE_HPP_
