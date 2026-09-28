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

#include "lekiwi_hardware/lekiwi_system_hardware.hpp"

#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace lekiwi_hardware
{

hardware_interface::CallbackReturn
LeKiwiSystemHardware::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  port_ = info_.hardware_parameters.count("port") ?
    info_.hardware_parameters.at("port") : "/dev/ttyACM0";
  calib_file_ = info_.hardware_parameters.count("calib_file") ?
    info_.hardware_parameters.at("calib_file") : "";
  {
    const auto it = info_.hardware_parameters.find("simulated");
    simulated_ = it != info_.hardware_parameters.end() &&
      (it->second == "1" || it->second == "true" || it->second == "True" || it->second == "TRUE");
  }

  num_joints_ = info_.joints.size();
  base_only_mode_ = (num_joints_ == FULL_BASE_JOINTS);
  if (!base_only_mode_ && num_joints_ != FULL_JOINTS) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Expected %zu joints (full) or %zu joints (base-only), got %zu",
      FULL_JOINTS, FULL_BASE_JOINTS, num_joints_);
    return hardware_interface::CallbackReturn::ERROR;
  }
  num_arm_joints_ = base_only_mode_ ? 0 : FULL_ARM_JOINTS;
  num_base_joints_ = base_only_mode_ ? num_joints_ : FULL_BASE_JOINTS;

  hw_positions_.resize(num_joints_, 0.0);
  hw_velocities_.resize(num_joints_, 0.0);
  hw_commands_.resize(num_joints_, 0.0);

  for (size_t i = 0; i < num_joints_; i++) {
    if (i < num_arm_joints_) {
      arm_ids_.push_back(
        static_cast<std::uint8_t>(std::stoi(info_.joints[i].parameters.at("id"))));
    } else {
      wheel_ids_.push_back(
        static_cast<std::uint8_t>(std::stoi(info_.joints[i].parameters.at("id"))));
    }
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
LeKiwiSystemHardware::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Configuring...");

  // Load arm calibration (LeKiwi "id"-field format is accepted by the SDK's
  // Calibration::load alongside the SO-101 keyed format).
  so101::Calibration calibration;
  std::vector<feetech::MotorConfig> arm_motors;
  if (!base_only_mode_) {
    try {
      std::vector<std::string> joint_order;
      for (size_t i = 0; i < num_arm_joints_; i++) {
        joint_order.push_back(info_.joints[i].name);
      }
      calibration = so101::Calibration::load(calib_file_, joint_order);
    } catch (const so101::CalibError & e) {
      RCLCPP_ERROR(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        "Calibration error: %s (%s%s)", e.what(), e.path.c_str(),
        e.joint.empty() ? "" : (" joint " + e.joint).c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    for (size_t i = 0; i < num_arm_joints_; i++) {
      const std::string & name = info_.joints[i].name;
      feetech::MotorConfig motor;
      motor.id = arm_ids_[i];
      motor.name = name;
      const auto & jc = calibration.at(name);
      motor.homing_offset = jc.homing_offset;
      motor.range_min = jc.range_min;
      motor.range_max = jc.range_max;
      arm_motors.push_back(motor);
      arm_command_map_[name] = 0.0;
    }
  }

  // One shared bus: arm (position) + wheels (velocity).
  feetech::BusOptions options;
  options.port = port_;
  options.simulated = simulated_;
  options.motors = arm_motors;
  for (const std::uint8_t id : wheel_ids_) {
    feetech::MotorConfig motor;
    motor.id = id;
    motor.name = "wheel_" + std::to_string(id);
    motor.mode = feetech::Mode::Wheel;
    options.motors.push_back(motor);
  }

  bus_ = std::make_unique<feetech::Bus>(options);
  if (!bus_->open()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Failed to connect to motors on port %s", port_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (!base_only_mode_) {
    so101::ArmConfig arm_config;
    arm_config.port = port_;
    arm_config.simulated = simulated_;
    arm_config.calibration_file = calib_file_;
    arm_config.joint_order.clear();  // drop the SDK default "1".."6" before appending
    for (size_t i = 0; i < num_arm_joints_; i++) {
      arm_config.joint_order.push_back(info_.joints[i].name);
    }
    arm_ = std::make_unique<so101::Arm>(arm_config);
    if (!arm_->attach_shared_bus(*bus_, calibration)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        "Failed to attach arm to the shared bus");
      bus_.reset();
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Configured!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
LeKiwiSystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < num_joints_; i++) {
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]);
    state_interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
LeKiwiSystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < num_joints_; i++) {
    if (i < num_arm_joints_) {
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
    } else {
      command_interfaces.emplace_back(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_commands_[i]);
    }
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn
LeKiwiSystemHardware::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Activating...");

  // Motors are pinged as part of the scoped activation paths below: arm motors
  // by the arm's own activation, wheel motors by apply_configs. Nothing to ping
  // separately here (fail-closed behaviour comes from those two calls).

  // Arm: scoped configuration + initial sync (fail-closed with rollback).
  if (arm_ && !arm_->activate()) {
    const auto & health = arm_->health();
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Arm activation failed: %s (fault=%d)", health.detail.c_str(),
      static_cast<int>(health.fault));
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Wheels: scoped configuration (wheel mode).
  const auto wheel_result = bus_->apply_configs(wheel_ids_);
  if (!wheel_result.ok) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Wheel activation failed: %s", wheel_result.detail.c_str());
    bus_->emergency_release(wheel_ids_);
    // The arm activated successfully a few lines above, so it is holding
    // torque right now. Activation as a whole has failed, and leaving half a
    // robot energized is exactly the state fail-closed exists to prevent:
    // release the arm too before reporting ERROR.
    if (arm_ && !arm_->stop(so101::StopPolicy::TorqueOff)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        "Arm torque release during wheel-failure rollback also failed: %s",
        arm_->health().detail.c_str());
    }
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Seed wheel commands to zero.
  std::vector<feetech::MotorTarget> stop_targets;
  for (const std::uint8_t id : wheel_ids_) {
    feetech::MotorTarget t;
    t.id = id;
    t.velocity = 0.0;
    stop_targets.push_back(t);
  }
  bus_->sync_write_velocities(stop_targets);

  // Seed the ros2_control command buffer from the SDK's hold targets before
  // the first write(). Arm::activate() leaves the arm holding its measured
  // pose, but hw_commands_ still holds zeros from on_init (or the previous
  // activation's values on an inactive -> active re-entry). Without this the
  // very first write() commands those stale values, driving a non-zero arm
  // toward zero while the runtime is still idle. Wheels are seeded to zero to
  // match the stop written above.
  if (arm_) {
    so101::ArmState state;
    if (!arm_->read(state)) {
      RCLCPP_ERROR(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        "Arm read failed while seeding hold targets: %s", arm_->health().detail.c_str());
      bus_->emergency_release(wheel_ids_);
      arm_->stop(so101::StopPolicy::TorqueOff);
      return hardware_interface::CallbackReturn::ERROR;
    }
    const auto & targets = arm_->command_targets();
    for (size_t i = 0; i < num_arm_joints_; i++) {
      const std::string & name = info_.joints[i].name;
      const auto reading = state.joints.find(name);
      const auto target = targets.find(name);
      if (reading == state.joints.end() || target == targets.end()) {
        RCLCPP_ERROR(
          rclcpp::get_logger("LeKiwiSystemHardware"),
          "Arm joint %s missing while seeding hold targets", name.c_str());
        bus_->emergency_release(wheel_ids_);
        arm_->stop(so101::StopPolicy::TorqueOff);
        return hardware_interface::CallbackReturn::ERROR;
      }
      hw_positions_[i] = reading->second.position;
      hw_velocities_[i] = reading->second.velocity;
      hw_commands_[i] = target->second;
      arm_command_map_[name] = hw_commands_[i];
    }
  }
  for (size_t i = 0; i < wheel_ids_.size(); i++) {
    hw_commands_[num_arm_joints_ + i] = 0.0;
  }

  RCLCPP_INFO(
    rclcpp::get_logger("LeKiwiSystemHardware"),
    "Activated! %zu arm + %zu base motors running.",
    arm_ids_.size(), wheel_ids_.size());
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
LeKiwiSystemHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  // inactive -> active re-enters on_activate() without on_configure(), so the
  // shared bus must stay open: stop wheels and release torque only.
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Deactivating (torque off, bus kept)...");
  if (!bus_) {
    return hardware_interface::CallbackReturn::SUCCESS;
  }

  // Stop wheels first.
  std::vector<feetech::MotorTarget> stop_targets;
  for (const std::uint8_t id : wheel_ids_) {
    feetech::MotorTarget t;
    t.id = id;
    t.velocity = 0.0;
    stop_targets.push_back(t);
  }
  const auto stop_result = bus_->sync_write_velocities(stop_targets);
  if (!stop_result.ok) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Wheel stop command failed: %s", stop_result.detail.c_str());
  }

  // Both releases are attempted even when the first one fails -- a subsystem
  // that can still be released must be. The outcomes are aggregated: this
  // callback's return value is what /runtime/stop reports to its caller, so
  // swallowing a failed release here would confirm a stop while a motor is
  // still energized. In base-only mode there is no arm, which is precisely
  // when an ignored wheel-release result had nothing left to fail on.
  const auto wheel_release = bus_->emergency_release(wheel_ids_);
  if (!wheel_release.ok) {
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Wheel torque release failed: %s", wheel_release.detail.c_str());
  }
  bool arm_released = true;
  if (arm_ && !arm_->stop(so101::StopPolicy::TorqueOff)) {
    arm_released = false;
    RCLCPP_ERROR(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      "Arm torque release failed: %s", arm_->health().detail.c_str());
  }
  if (!wheel_release.ok || !arm_released) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
LeKiwiSystemHardware::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("LeKiwiSystemHardware"), "Cleaning up (closing bus)...");
  teardown_bus();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
LeKiwiSystemHardware::on_shutdown(const rclcpp_lifecycle::State &)
{
  teardown_bus();
  return hardware_interface::CallbackReturn::SUCCESS;
}

void LeKiwiSystemHardware::teardown_bus()
{
  if (!bus_) {
    return;
  }
  bus_->emergency_release(wheel_ids_);
  if (arm_) {
    arm_->deactivate();  // scoped release (shared bus: close is a no-op here)
  }
  bus_->close();
  bus_.reset();
  arm_.reset();
}

hardware_interface::return_type
LeKiwiSystemHardware::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);

  // Arm: scoped read (wheel failures do not affect the arm subsystem).
  if (arm_) {
    so101::ArmState state;
    if (!arm_->read(state)) {
      RCLCPP_ERROR_THROTTLE(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        steady_clock, 500, "Arm read failed");
      return hardware_interface::return_type::ERROR;
    }
    for (size_t i = 0; i < num_arm_joints_; i++) {
      const auto it = state.joints.find(info_.joints[i].name);
      if (it == state.joints.end()) {
        return hardware_interface::return_type::ERROR;
      }
      hw_positions_[i] = it->second.position;
      hw_velocities_[i] = it->second.velocity;
    }
  }

  // Wheels: scoped read.
  std::vector<feetech::MotorSample> wheel_samples;
  const auto wheel_result = bus_->sync_read(wheel_samples, wheel_ids_);
  if (!wheel_result.ok) {
    RCLCPP_ERROR_THROTTLE(
      rclcpp::get_logger("LeKiwiSystemHardware"),
      steady_clock, 500, "Wheel read failed: %s", wheel_result.detail.c_str());
    return hardware_interface::return_type::ERROR;
  }
  for (size_t i = 0; i < wheel_ids_.size(); i++) {
    const size_t idx = num_arm_joints_ + i;
    hw_positions_[idx] = wheel_samples[i].position;
    hw_velocities_[idx] = wheel_samples[i].velocity;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
LeKiwiSystemHardware::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);

  if (arm_) {
    for (size_t i = 0; i < num_arm_joints_; i++) {
      arm_command_map_[info_.joints[i].name] = hw_commands_[i];
    }
    if (!arm_->write_targets(arm_command_map_)) {
      RCLCPP_ERROR_THROTTLE(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        steady_clock, 500, "Arm write failed");
      return hardware_interface::return_type::ERROR;
    }
  }

  if (!wheel_ids_.empty()) {
    std::vector<feetech::MotorTarget> targets;
    for (size_t i = 0; i < wheel_ids_.size(); i++) {
      feetech::MotorTarget t;
      t.id = wheel_ids_[i];
      t.velocity = hw_commands_[num_arm_joints_ + i];
      targets.push_back(t);
    }
    if (!bus_->sync_write_velocities(targets).ok) {
      RCLCPP_ERROR_THROTTLE(
        rclcpp::get_logger("LeKiwiSystemHardware"),
        steady_clock, 500, "Wheel write failed");
      return hardware_interface::return_type::ERROR;
    }
  }

  return hardware_interface::return_type::OK;
}

}  // namespace lekiwi_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  lekiwi_hardware::LeKiwiSystemHardware,
  hardware_interface::SystemInterface)
