// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The projection rules, case for case with what aimdk_robot's test_projection.py
// asserted while they lived in Python.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "aimdk_dataplane/projection.hpp"

using aimdk_dataplane::CommandRejected;
using aimdk_dataplane::HandConfig;
using aimdk_dataplane::JointEntries;
using aimdk_dataplane::kVendorArmOrder;
using aimdk_dataplane::LocomotionLimits;
using Measured = std::array<std::map<std::string, double>, 2>;

namespace
{

std::string reason_of(const std::function<void()> & call)
{
  try {
    call();
  } catch (const CommandRejected & rejected) {
    return rejected.reason;
  }
  return "";
}

}  // namespace

TEST(ArmPayload, ReordersIntoTheVendorIndexOrder)
{
  std::vector<std::string> reversed(kVendorArmOrder.rbegin(), kVendorArmOrder.rend());
  std::vector<double> values(14);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<double>(index);
  }
  const auto payload = aimdk_dataplane::arm_payload(values, reversed);
  for (size_t index = 0; index < payload.size(); ++index) {
    EXPECT_EQ(payload[index], 13.0 - static_cast<double>(index));
  }
}

TEST(ArmPayload, RejectsAWrongLengthUnknownJointsAndNonFiniteValues)
{
  EXPECT_EQ(reason_of([] {aimdk_dataplane::arm_payload(std::vector<double>(13), kVendorArmOrder);}), "ARM_COMMAND_LENGTH");
  auto bad = kVendorArmOrder;
  bad[0] = "not_a_joint";
  EXPECT_EQ(reason_of([&] {aimdk_dataplane::arm_payload(std::vector<double>(14), bad);}), "ARM_COMMAND_JOINTS");
  for (double value : {std::nan(""), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
    std::vector<double> values(14);
    values[0] = value;
    EXPECT_EQ(reason_of([&] {aimdk_dataplane::arm_payload(values, kVendorArmOrder);}), "NON_FINITE_COMMAND");
  }
}

TEST(HeadPayload, UsesYawThenPitchAndRefusesTargetsOutsideTheTravel)
{
  EXPECT_EQ(
    aimdk_dataplane::head_payload({0.2, 0.1}, {"head_pitch_joint", "head_yaw_joint"}, {}),
    (std::vector<double>{0.1, 0.2}));
  const std::map<std::string, double> limits = {{"head_yaw_joint", 0.38}, {"head_pitch_joint", 0.35}};
  const std::vector<std::string> joints = {"head_yaw_joint", "head_pitch_joint"};
  EXPECT_EQ(aimdk_dataplane::head_payload({0.3, -0.3}, joints, limits), (std::vector<double>{0.3, -0.3}));
  EXPECT_EQ(reason_of([&] {aimdk_dataplane::head_payload({0.5, 0.0}, joints, limits);}), "HEAD_COMMAND_RANGE");
  EXPECT_EQ(reason_of([&] {aimdk_dataplane::head_payload({0.0, -0.4}, joints, limits);}), "HEAD_COMMAND_RANGE");
  EXPECT_EQ(aimdk_dataplane::head_payload({2.0, 2.0}, joints, {}), (std::vector<double>{2.0, 2.0}));
}

TEST(HandPayload, ClawRangeIsRefusedNotClamped)
{
  const auto [sub_mode, payload] = aimdk_dataplane::hand_payload({0.0, 1.0}, "claw", HandConfig{});
  EXPECT_EQ(sub_mode, aimdk_dataplane::kHandSubModeClaw);
  EXPECT_EQ(payload, (std::vector<double>{0.0, 1.0}));
  try {
    aimdk_dataplane::hand_payload({1.5, 0.5}, "claw", HandConfig{});
    FAIL();
  } catch (const CommandRejected & rejected) {
    EXPECT_EQ(rejected.reason, "GRIPPER_COMMAND_RANGE");
    EXPECT_NE(std::string(rejected.what()).find("left"), std::string::npos);
  }
}

TEST(HandPayload, DexterousJointsNeedBothHands)
{
  const auto [sub_mode, payload] = aimdk_dataplane::hand_payload(std::vector<double>(20), "dexterous_joint", {});
  EXPECT_EQ(sub_mode, aimdk_dataplane::kHandSubModeDexterousJoint);
  EXPECT_EQ(payload.size(), 20u);
  EXPECT_EQ(
    reason_of([] {aimdk_dataplane::hand_payload(std::vector<double>(10), "dexterous_joint", {});}),
    "HAND_COMMAND_LENGTH");
}

TEST(HandPayload, GesturesValidateIdsAndOpenness)
{
  HandConfig config;
  config.gesture_ids = std::vector<int>{0, 1, 2, 3};
  const auto [sub_mode, payload] = aimdk_dataplane::hand_payload({1.0, 0.5, 2.0, 1.0}, "dexterous_gesture", config);
  EXPECT_EQ(sub_mode, aimdk_dataplane::kHandSubModeDexterousGesture);
  EXPECT_EQ(payload, (std::vector<double>{1.0, 0.5, 2.0, 1.0}));
  config.gesture_ids = std::vector<int>{0, 1};
  EXPECT_EQ(
    reason_of([&] {aimdk_dataplane::hand_payload({9.0, 0.5, 1.0, 0.5}, "dexterous_gesture", config);}),
    "HAND_GESTURE_ID");
  EXPECT_EQ(
    reason_of([] {aimdk_dataplane::hand_payload({1.0, 1.5, 1.0, 0.5}, "dexterous_gesture", HandConfig{});}),
    "HAND_GESTURE_RANGE");
  EXPECT_THROW(aimdk_dataplane::hand_payload({0.0, 0.0}, "magnetic", {}), std::invalid_argument);
}

TEST(Twist, DeadbandMinimumAndMaximum)
{
  const LocomotionLimits limits;
  EXPECT_EQ(aimdk_dataplane::twist_to_locomotion(0.001, -0.002, 0.0, limits), (std::array<double, 3>{0.0, 0.0, 0.0}));
  try {
    aimdk_dataplane::twist_to_locomotion(0.1, 0.0, 0.0, limits);
    FAIL();
  } catch (const CommandRejected & rejected) {
    EXPECT_EQ(rejected.reason, "VELOCITY_BELOW_MIN");
    EXPECT_NE(std::string(rejected.what()).find("would not move the robot"), std::string::npos);
  }
  EXPECT_EQ(reason_of([&] {aimdk_dataplane::twist_to_locomotion(0.0, 0.0, 2.0, limits);}), "VELOCITY_ABOVE_MAX");
  EXPECT_EQ(aimdk_dataplane::twist_to_locomotion(0.5, -0.3, 0.4, limits), (std::array<double, 3>{0.5, -0.3, 0.4}));
}

TEST(Twist, ThePlatformEnvelopeOnlyTightensTheProfileCeiling)
{
  const LocomotionLimits limits;
  const auto tightened = limits.with_platform_envelope(std::make_pair(0.4, 0.3));
  EXPECT_EQ(tightened.max_linear_mps, 0.4);
  EXPECT_EQ(tightened.max_angular_radps, 0.3);
  EXPECT_EQ(tightened.min_linear_mps, limits.min_linear_mps);
  EXPECT_EQ(limits.with_platform_envelope(std::make_pair(5.0, 5.0)).max_linear_mps, limits.max_linear_mps);
  // All-zero bounds mean "not reported"; enforcing them would refuse everything.
  EXPECT_EQ(limits.with_platform_envelope(std::make_pair(0.0, 0.0)).max_linear_mps, limits.max_linear_mps);
  EXPECT_EQ(limits.with_platform_envelope(std::nullopt).max_angular_radps, limits.max_angular_radps);
}

TEST(DexterousHold, HoldsEachInstalledHandAtItsMeasuredPose)
{
  std::map<std::string, double> left;
  std::map<std::string, double> right;
  for (size_t index = 0; index < aimdk_dataplane::kDexterousJointOrder.size(); ++index) {
    left["L_" + aimdk_dataplane::kDexterousJointOrder[index] + "_joint"] = static_cast<double>(index);
    right["R_" + aimdk_dataplane::kDexterousJointOrder[index] + "_joint"] = 10.0 + static_cast<double>(index);
  }
  std::vector<double> left_only(20, 0.0);
  std::vector<double> both(20);
  for (size_t index = 0; index < 20; ++index) {
    both[index] = static_cast<double>(index);
    if (index < 10) {
      left_only[index] = static_cast<double>(index);
    }
  }
  EXPECT_EQ(aimdk_dataplane::dexterous_hold_payload({true, false}, Measured{left, {}}), left_only);
  EXPECT_EQ(aimdk_dataplane::dexterous_hold_payload({true, true}, Measured{left, right}), both);
  left.erase("L_pinky_pip_joint");
  EXPECT_FALSE(aimdk_dataplane::dexterous_hold_payload({true, false}, Measured{left, {}}).has_value());
  EXPECT_FALSE(aimdk_dataplane::dexterous_hold_payload({false, false}, Measured{}).has_value());
}

TEST(HandTypeFamily, CoversTheVendorEnumeration)
{
  EXPECT_EQ(aimdk_dataplane::hand_type_family(0x01), "dexterous");
  EXPECT_EQ(aimdk_dataplane::hand_type_family(0x02), "claw");
  EXPECT_EQ(aimdk_dataplane::hand_type_family(0x03), "dexterous");
  EXPECT_EQ(aimdk_dataplane::hand_type_family(0x04), "dexterous");
  EXPECT_FALSE(aimdk_dataplane::hand_type_family(0x00).has_value());
  EXPECT_FALSE(aimdk_dataplane::hand_type_family(0xFF).has_value());
}

TEST(Aggregate, FollowsThePublicOrderAndOmitsAbsentJoints)
{
  const JointEntries arm = {{"b", 2.0, 0.2, 0.02}, {"a", 1.0, 0.1, 0.01}};
  const JointEntries head = {{"c", 3.0, 0.3, 0.03}};
  const auto merged = aimdk_dataplane::aggregate_joint_state({&arm, &head}, {"a", "missing", "c", "b"});
  EXPECT_EQ(merged.names, (std::vector<std::string>{"a", "c", "b"}));
  EXPECT_EQ(merged.position, (std::vector<double>{1.0, 3.0, 2.0}));
  EXPECT_EQ(merged.effort, (std::vector<double>{0.01, 0.03, 0.02}));
}
