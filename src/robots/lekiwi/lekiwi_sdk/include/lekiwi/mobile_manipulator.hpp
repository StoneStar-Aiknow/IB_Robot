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

#ifndef LEKIWI__MOBILE_MANIPULATOR_HPP_
#define LEKIWI__MOBILE_MANIPULATOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include "feetech/bus.hpp"
#include "lekiwi/omni_base.hpp"
#include "so101/arm.hpp"

namespace lekiwi
{

/// Mobile manipulator construction options.
struct MobileManipulatorConfig
{
  std::string port;
  std::uint32_t baudrate = 1'000'000;
  /// SO-101 arm options (joint_order, calibration, reset positions).
  so101::ArmConfig arm;
  /// Wheel motor ids in mount-angle order (left, back, right).
  std::vector<std::uint8_t> wheel_ids = {7, 8, 9};
  BaseGeometry base;
  /// Base-only configuration: no arm motors are registered and no arm
  /// calibration is required.
  bool base_only = false;
  bool simulated = false;
};

/// An SO-101 arm and an omni base sharing one Feetech bus. Arm and base
/// operations are independently addressable; one subsystem's lifecycle does
/// not affect the other's bus operations.
class MobileManipulator
{
public:
  explicit MobileManipulator(MobileManipulatorConfig config);

  MobileManipulator(const MobileManipulator &) = delete;
  MobileManipulator & operator=(const MobileManipulator &) = delete;

  /// Open the bus and, in full mode, load/validate the arm calibration.
  /// Base-only mode requires no calibration.
  bool connect();

  /// Configure all motors and bring both subsystems to ready state. In
  /// base-only mode only the wheels are configured.
  bool activate();

  so101::Arm & arm();
  OmniBase & base();

  /// Arm is present (non-base-only configuration).
  bool has_arm() const;

  /// Simulated-transport control handle (aborts when not simulated).
  feetech::Bus::SimControl & sim();

private:
  MobileManipulatorConfig config_;
  // Declaration order is destruction order reversed: the arm and the base hold
  // non-owning references to the bus, and ~Arm() touches it, so the bus must be
  // declared first and therefore destroyed last. Do not reorder.
  std::unique_ptr<feetech::Bus> bus_;
  std::unique_ptr<so101::Arm> arm_;
  std::unique_ptr<OmniBase> base_;
};

}  // namespace lekiwi

#endif  // LEKIWI__MOBILE_MANIPULATOR_HPP_
