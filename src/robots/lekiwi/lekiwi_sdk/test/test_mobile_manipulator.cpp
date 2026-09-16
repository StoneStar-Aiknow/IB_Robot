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

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include "lekiwi/mobile_manipulator.hpp"

namespace
{

class TempCalibFile
{
public:
  TempCalibFile()
  {
    path_ = "/tmp/lekiwi_sdk_calib_test_" + std::to_string(getpid()) + ".json";
    std::ofstream file(path_);
    file << R"({
      "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "3": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "6": {"homing_offset": 0, "range_min": 0, "range_max": 4095}
    })";
  }
  ~TempCalibFile() { std::remove(path_.c_str()); }
  const std::string & path() const { return path_; }

private:
  std::string path_;
};

lekiwi::MobileManipulatorConfig full_config(const std::string & calib_path)
{
  lekiwi::MobileManipulatorConfig config;
  config.simulated = true;
  config.arm.calibration_file = calib_path;
  return config;
}

TEST(MobileManipulator, ArmAndBaseOnOneSharedBus)
{
  TempCalibFile calib;
  lekiwi::MobileManipulator robot(full_config(calib.path()));
  ASSERT_TRUE(robot.connect());
  ASSERT_TRUE(robot.activate());
  ASSERT_TRUE(robot.has_arm());

  // Command the base while the arm holds: both command streams reach their
  // motors without interference, and each subsystem reads independently.
  ASSERT_TRUE(robot.base().set_body_velocity(0.1, 0.0, 0.0));
  so101::ArmState arm_state;
  ASSERT_TRUE(robot.arm().read(arm_state));
  ASSERT_EQ(arm_state.joints.size(), 6U);

  std::array<double, 3> positions{};
  std::array<double, 3> velocities{};
  ASSERT_TRUE(robot.base().read_wheels(positions, velocities));
  // Wheel 7 (left, first mount angle) follows its mount-angle projection.
  const auto & geo = robot.base().geometry();
  const double expected_wheel0 =
    0.1 * std::cos(geo.mount_angles[0]) / geo.wheel_radius;
  EXPECT_NEAR(velocities[0], expected_wheel0, 0.002);

  // Wheel-mode torque state is independent of arm activation.
  EXPECT_TRUE(robot.sim().torque_enabled(1));
  EXPECT_TRUE(robot.sim().torque_enabled(7));
}

// Regression: connect() used to release the bus before the arm, and ~Arm()
// deactivates through its non-owning bus pointer -- a use-after-free on the
// second connect(). An ordinary build may not fault on it; this pins the
// supported call sequence and is the case a sanitizer build would flag.
TEST(MobileManipulator, ReconnectDoesNotTouchAReleasedBus)
{
  TempCalibFile calib;
  lekiwi::MobileManipulator robot(full_config(calib.path()));
  ASSERT_TRUE(robot.connect());
  ASSERT_TRUE(robot.activate());

  ASSERT_TRUE(robot.connect());
  ASSERT_TRUE(robot.activate());
  EXPECT_TRUE(robot.has_arm());
  EXPECT_TRUE(robot.base().set_wheel_velocities({0.0, 0.0, 0.0}));
}

TEST(MobileManipulator, BaseOnlyRequiresNoArmCalibration)
{
  lekiwi::MobileManipulatorConfig config;
  config.simulated = true;
  config.base_only = true;
  // No calibration file configured: base-only activation must succeed.
  lekiwi::MobileManipulator robot(config);
  ASSERT_TRUE(robot.connect());
  ASSERT_TRUE(robot.activate());
  EXPECT_FALSE(robot.has_arm());
  ASSERT_TRUE(robot.base().set_wheel_velocities({1.0, 1.0, 1.0}));
  EXPECT_THROW((void)robot.arm(), std::logic_error);
}

TEST(MobileManipulator, BaseOnlyIgnoresArmConfigErrors)
{
  TempCalibFile calib;
  lekiwi::MobileManipulatorConfig config = full_config(calib.path());
  config.base_only = true;
  config.arm.calibration_file = "/tmp/lekiwi_missing.json";  // would fail
  lekiwi::MobileManipulator robot(config);
  ASSERT_TRUE(robot.connect());
  ASSERT_TRUE(robot.activate());
}

TEST(MobileManipulator, FullModeFailsOnMissingCalibration)
{
  lekiwi::MobileManipulatorConfig config = full_config("/tmp/lekiwi_missing.json");
  lekiwi::MobileManipulator robot(config);
  EXPECT_FALSE(robot.connect());
}

TEST(MobileManipulator, BadWheelCountRejected)
{
  lekiwi::MobileManipulatorConfig config = full_config("");
  config.wheel_ids = {7, 8};
  EXPECT_THROW(lekiwi::MobileManipulator robot(config), std::invalid_argument);
}

}  // namespace
