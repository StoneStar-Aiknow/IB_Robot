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

#include "lekiwi/mobile_manipulator.hpp"

#include <stdexcept>

#include "so101/calibration.hpp"

namespace lekiwi
{

namespace
{

feetech::BusOptions make_bus_options(
  const MobileManipulatorConfig & config, const std::vector<feetech::MotorConfig> & arm_motors)
{
  feetech::BusOptions options;
  options.port = config.port;
  options.baudrate = config.baudrate;
  options.simulated = config.simulated;
  options.motors = arm_motors;
  for (const std::uint8_t id : config.wheel_ids) {
    feetech::MotorConfig motor;
    motor.id = id;
    motor.name = "wheel_" + std::to_string(id);
    motor.mode = feetech::Mode::Wheel;
    options.motors.push_back(motor);
  }
  return options;
}

/// Build the arm's motor configs (with calibration) without opening a bus.
std::vector<feetech::MotorConfig> build_arm_motors(
  const so101::ArmConfig & arm, const so101::Calibration & calibration)
{
  std::vector<feetech::MotorConfig> motors;
  for (const std::string & joint : arm.joint_order) {
    feetech::MotorConfig motor;
    motor.id = static_cast<std::uint8_t>(std::stoi(joint));
    motor.name = joint;
    const so101::JointCalibration & joint_calib = calibration.at(joint);
    motor.homing_offset = joint_calib.homing_offset;
    motor.range_min = joint_calib.range_min;
    motor.range_max = joint_calib.range_max;
    motors.push_back(motor);
  }
  return motors;
}

}  // namespace

MobileManipulator::MobileManipulator(MobileManipulatorConfig config)
: config_(std::move(config))
{
  if (config_.wheel_ids.size() != 3) {
    throw std::invalid_argument("MobileManipulator requires exactly three wheel ids");
  }
}

bool MobileManipulator::connect()
{
  // Destroy in reverse dependency order. The arm and the base hold non-owning
  // references to the bus, and ~Arm() calls deactivate(), which touches it --
  // releasing the bus first would make that a use-after-free on a second
  // connect().
  base_.reset();
  arm_.reset();
  bus_.reset();

  std::vector<feetech::MotorConfig> arm_motors;
  so101::Calibration calibration;
  if (!config_.base_only) {
    try {
      calibration = so101::Calibration::load(
        config_.arm.calibration_file, config_.arm.joint_order);
    } catch (const so101::CalibError &) {
      return false;
    }
    arm_motors = build_arm_motors(config_.arm, calibration);
  }

  bus_ = std::make_unique<feetech::Bus>(make_bus_options(config_, arm_motors));
  if (!bus_->open()) {
    return false;
  }

  if (!config_.base_only) {
    // The arm shares this bus: construct it in place over the same bus with
    // the already-validated calibration.
    arm_ = std::make_unique<so101::Arm>(config_.arm);
    if (!arm_->attach_shared_bus(*bus_, calibration)) {
      arm_.reset();
      bus_.reset();
      return false;
    }
  }
  base_ = std::make_unique<OmniBase>(*bus_, config_.wheel_ids, config_.base);
  return true;
}

bool MobileManipulator::activate()
{
  if (!bus_ || !base_) {
    return false;
  }
  if (arm_ && !arm_->activate()) {
    return false;
  }
  if (!base_->activate()) {
    // Fail closed: the arm is energized at this point, and a half-energized
    // robot is the one state activation must never leave behind.
    if (arm_) {
      arm_->stop(so101::StopPolicy::TorqueOff);
    }
    return false;
  }
  return true;
}

so101::Arm & MobileManipulator::arm()
{
  if (!arm_) {
    throw std::logic_error("arm() called on a base-only MobileManipulator");
  }
  return *arm_;
}

OmniBase & MobileManipulator::base()
{
  if (!base_) {
    throw std::logic_error("base() called before connect()");
  }
  return *base_;
}

bool MobileManipulator::has_arm() const { return arm_ != nullptr; }

feetech::Bus::SimControl & MobileManipulator::sim()
{
  if (!bus_) {
    throw std::logic_error("sim() called before connect()");
  }
  return bus_->sim();
}

}  // namespace lekiwi
