// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic fixtures from CBF/cbf/tests/{phase_c_tests,rigid_thumb_tests}.cpp.
// These inputs are in the frozen model's URDF coordinates, not SDK coordinates.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "cbf/command_adapter.hpp"
#include "cbf/safety_filter.hpp"

namespace
{

using x2::cbf::FilterStatus;
using x2::cbf::SafetyFilterConfig;
using x2::cbf::SafetyFilterInput;

SafetyFilterConfig test_config()
{
  SafetyFilterConfig config;
  // Test the result contract independently of the machine's performance.
  // The deadline case below supplies its own deliberately exhausted budget.
  config.deadline_us = 100000.0;
  return config;
}

SafetyFilterInput baseline_input()
{
  SafetyFilterInput input;
  input.q_current = x2::cbf::generated::kNaturalStanding;
  input.q_nominal = input.q_current;
  return input;
}

std::array<std::array<double, 2>, 4> historical_native_thumb_bounds()
{
  // hand_overlay.py root-axis ranges mapped by the historical, UNVERIFIED
  // scale [-1, 1, 1, -1], offset [0, -1.6424, -0.576, -1.6424].
  return {{{-0.03, 1.117}, {-1.6924, -0.0024}, {-0.606, 0.294}, {-1.6924, -0.0024}}};
}

// Copied fixture from rigid_thumb_tests.cpp::moving_input.
SafetyFilterInput moving_input()
{
  auto input = baseline_input();
  input.thumb_current_valid = true;
  input.q_nominal[4] += 0.004;
  input.q_nominal[11] -= 0.004;
  input.q_thumb_urdf[0] += 0.005;
  input.q_thumb_urdf[1] -= 0.004;
  input.q_thumb_urdf[3] -= 0.005;
  input.q_thumb_urdf[4] += 0.004;
  return input;
}

x2::cbf::PrimitiveDistanceReport geometry(
  const SafetyFilterInput & input, bool current_thumb)
{
  x2::cbf::KinematicsInput pose;
  pose.q_arm = input.q_current;
  pose.q_thumb_urdf = current_thumb ? input.q_thumb_current_urdf : input.q_thumb_urdf;
  pose.thumb_pose_valid = true;
  pose.hands_installed = input.hands_installed;
  return x2::cbf::evaluate_proxy_distances(pose);
}

x2::cbf::KinematicsInput proxy_input()
{
  x2::cbf::KinematicsInput input;
  input.q_arm = x2::cbf::generated::kNaturalStanding;
  input.q_thumb_urdf = x2::cbf::kRigidThumbBaselineUrdf;
  input.thumb_pose_valid = true;
  return input;
}

std::size_t find_pair(x2::cbf::PrimitiveId first, x2::cbf::PrimitiveId second)
{
  for (std::size_t index = 0; index < x2::cbf::kPrimitivePairs.size(); ++index) {
    const auto pair = x2::cbf::kPrimitivePairs[index];
    if (pair.first == first && pair.second == second) return index;
  }
  return x2::cbf::kPrimitivePairCount;
}

void expect_finite_limited_arm(const x2::cbf::SafetyFilterResult & result)
{
  for (std::size_t joint = 0; joint < result.q_safe.size(); ++joint) {
    SCOPED_TRACE(joint);
    EXPECT_TRUE(std::isfinite(result.q_safe[joint]));
    EXPECT_GE(result.q_safe[joint], x2::cbf::generated::kArmLimits[joint][0]);
    EXPECT_LE(result.q_safe[joint], x2::cbf::generated::kArmLimits[joint][1]);
  }
}

}  // namespace

namespace
{
SafetyFilterInput airwall_incident_input()
{
  SafetyFilterInput input;
  // Passive frozen status at 09:47:08, c8703860. Do not combine it with
  // the later hand-state capture, which already contains a different gesture.
  input.q_current = {
    -0.6970882415771484, 0.5183825492858887, -1.0642800331115723,
    -1.79597806930542, -0.16691350936889648, -0.16613292694091797,
    0.23441696166992188, 0.1872386932373047, 0.06969928741455078,
    -0.3392925262451172, -2.0849380493164063, 0.1093902587890625,
    0.08754825592041016, -0.5922403335571289};
  input.q_nominal = input.q_current;
  input.check_current_arm_goal_range = false;
  input.hands_installed = {true, false};
  input.q_thumb_current_urdf = {
    0.03490234375000001, -1.0238843750000002, 0.0, 0.0872665, -0.7094284610304067, 0.0};
  input.q_thumb_urdf = input.q_thumb_current_urdf;
  input.thumb_current_valid = true;
  input.thumb_joint_limits_urdf[0] = {-0.03, 1.117};
  input.thumb_joint_limits_urdf[1] = {-1.6924, -0.0024};
  // Original live command cap, not a wider limit invented for recovery.
  input.command_joint_limits[8][1] = 0.0349067074800567;
  input.q_nominal[8] = input.command_joint_limits[8][1];
  return input;
}

void expect_nonworsening_recovery_path(const SafetyFilterInput & input, const SafetyFilterConfig & config)
{
  std::array<double, x2::cbf::kPrimitivePairCount> previous{};
  for (int step = 0; step <= 128; ++step) {
    const double t = static_cast<double>(step) / 128;
    x2::cbf::KinematicsInput pose;
    pose.thumb_pose_valid = true;
    pose.hands_installed = input.hands_installed;
    for (std::size_t j = 0; j < 14; ++j) {
      pose.q_arm[j] = input.q_current[j] + t * (input.q_nominal[j] - input.q_current[j]);
    }
    for (std::size_t j = 0; j < 6; ++j) {
      pose.q_thumb_urdf[j] = input.q_thumb_current_urdf[j] + t *
        (input.q_thumb_urdf[j] - input.q_thumb_current_urdf[j]);
    }
    const auto kin = x2::cbf::forward_kinematics(pose);
    const auto distances = x2::cbf::evaluate_proxy_distances(kin);
    ASSERT_TRUE(kin.valid);
    for (std::size_t p = 0; p < x2::cbf::kPrimitivePairCount; ++p) {
      if (!x2::cbf::proxy_pair_enabled(kin, p)) {continue;}
      const auto pair = x2::cbf::kPrimitivePairs[p];
      const bool shoulder_torso = pair.second == x2::cbf::PrimitiveId::Torso &&
        (pair.first == x2::cbf::PrimitiveId::LeftShoulder || pair.first == x2::cbf::PrimitiveId::RightShoulder);
      const double required = shoulder_torso ? config.shoulder_torso_clearance_m :
        config.extra_clearance_m + std::max(kin.primitives[static_cast<std::size_t>(pair.first)].margin,
        kin.primitives[static_cast<std::size_t>(pair.second)].margin);
      ASSERT_TRUE(distances.distances[p].valid);
      const double margin = distances.distances[p].signed_distance - required;
      if (step > 0) {EXPECT_GE(margin, std::min(0.0, previous[p]) - 1e-9);}
      previous[p] = margin;
    }
  }
}
}  // namespace

TEST(CbfKernel, RecordedAirwallIncidentAllowsWithdrawalWithoutRemovingClearance)
{
  auto input = airwall_incident_input();
  input.q_nominal[3] += 0.04;
  auto config = test_config();
  EXPECT_EQ(x2::cbf::verify_arm_target(input, config).status, FilterStatus::DegradedHold);
  config.allow_airwall_recovery = true;
  const auto verified = x2::cbf::verify_arm_target(input, config);
  ASSERT_TRUE(verified.thumb_target_accepted);
  EXPECT_TRUE(verified.nonlinear_verified);
  EXPECT_TRUE(verified.recovery_verified);
  EXPECT_EQ(verified.status, FilterStatus::AirwallRecovery);
  EXPECT_NEAR(verified.input_min_safety_margin_m, -0.00950410669244261, 1e-10);
  EXPECT_GT(verified.output_min_safety_margin_m, 0.0);
  EXPECT_EQ(verified.q_safe, input.q_nominal);
  expect_nonworsening_recovery_path(input, config);
}

TEST(CbfKernel, RateLimitedPartialWithdrawalCanContinueForMultipleFrames)
{
  auto input = airwall_incident_input();
  auto config = test_config();
  config.allow_airwall_recovery = true;
  int recovery_frames = 0;
  for (int frame = 0; frame < 12; ++frame) {
    input.q_nominal = input.q_current;
    input.q_nominal[8] = std::min(input.q_nominal[8], input.command_joint_limits[8][1]);
    input.q_nominal[3] += 0.005;
    const auto result = x2::cbf::verify_arm_target(input, config);
    ASSERT_TRUE(result.thumb_target_accepted) << x2::cbf::filter_status_name(result.status);
    ASSERT_TRUE(result.nonlinear_verified);
    if (frame == 0) {
      EXPECT_LT(result.output_min_safety_margin_m, 0.0);  // Not forced to clear 10 mm in one frame.
      EXPECT_GT(result.output_min_safety_margin_m, result.input_min_safety_margin_m);
    }
    recovery_frames += result.recovery_verified;
    expect_nonworsening_recovery_path(input, config);
    input.q_current = result.q_safe;
  }
  EXPECT_GT(recovery_frames, 1);
  EXPECT_LT(recovery_frames, 12);
}

TEST(CbfKernel, ProjectionFindsAnAirwallEscapeInsideThePublishedCommandStepBox)
{
  auto input = airwall_incident_input();
  const auto reference = input.q_nominal;
  for (std::size_t j = 0; j < 14; ++j) {
    input.command_joint_limits[j] = {
      std::max(x2::cbf::generated::kArmLimits[j][0], reference[j] - 0.04),
      std::min(input.command_joint_limits[j][1], reference[j] + 0.04)};
  }
  input.q_thumb_urdf[0] = 0.06153846153846155;
  input.q_thumb_urdf[1] = -1.519323076923077;
  auto config = test_config();
  config.allow_airwall_recovery = true;
  config.max_step_rad = 6.5;  // The input command box owns the actual published step.
  const auto result = x2::cbf::filter_arm_target(input, config);
  ASSERT_TRUE(result.thumb_target_accepted) << x2::cbf::filter_status_name(result.status)
    << " pair=" << result.path_violation_pair << " t=" << result.path_violation_fraction
    << " deficit=" << result.path_violation_deficit_m << " iterations=" << result.projection_iterations;
  ASSERT_TRUE(result.recovery_verified);
  EXPECT_GT(result.active_pair_count, 0u);
  double moved = 0.0;
  for (std::size_t j = 0; j < 14; ++j) {
    EXPECT_GE(result.q_safe[j], input.command_joint_limits[j][0]);
    EXPECT_LE(result.q_safe[j], input.command_joint_limits[j][1]);
    moved = std::max(moved, std::abs(result.q_safe[j] - reference[j]));
  }
  EXPECT_GT(moved, 1e-6);
  input.q_nominal = result.q_safe;
  expect_nonworsening_recovery_path(input, config);
}

TEST(CbfKernel, RecoveryRepairDeadlineAlwaysRejectsTheTargetWithDeadlineStatus)
{
  auto input = airwall_incident_input();
  const auto reference = input.q_nominal;
  for (std::size_t j = 0; j < 14; ++j) {
    input.command_joint_limits[j] = {
      std::max(x2::cbf::generated::kArmLimits[j][0], reference[j] - 0.04),
      std::min(input.command_joint_limits[j][1], reference[j] + 0.04)};
  }
  // The frozen new-thumb sweep needs a coordinated arm escape and path repair.
  input.q_thumb_urdf[0] = 0.06153846153846155;
  input.q_thumb_urdf[1] = -1.519323076923077;
  auto config = test_config();
  config.allow_airwall_recovery = true;
  config.max_step_rad = 6.5;  // The caller box owns the published command step.
  config.deadline_guard_us = 0.0;
  for (double budget_us : {1.0, 50.0, 250.0, 1000.0, 3000.0}) {
    SCOPED_TRACE(budget_us);
    config.deadline_us = budget_us;
    const auto result = x2::cbf::filter_arm_target(input, config);
    expect_finite_limited_arm(result);
    for (std::size_t j = 0; j < 14; ++j) {
      EXPECT_GE(result.q_safe[j], input.command_joint_limits[j][0]);
      EXPECT_LE(result.q_safe[j], input.command_joint_limits[j][1]);
    }
    // Timing decides which branch runs, never whether the test passes.
    if (result.deadline_missed) {
      EXPECT_EQ(result.status, FilterStatus::DeadlineFallback);
      EXPECT_FALSE(result.thumb_target_accepted);
    } else {
      EXPECT_NE(result.status, FilterStatus::DeadlineFallback);
      if (result.thumb_target_accepted) {
        ASSERT_TRUE(result.nonlinear_verified);
        ASSERT_TRUE(result.recovery_verified);
        EXPECT_EQ(result.status, FilterStatus::AirwallRecovery);
        auto accepted = input;
        accepted.q_nominal = result.q_safe;
        expect_nonworsening_recovery_path(accepted, config);
      }
    }
  }
}

TEST(CbfKernel, RecoveryRejectsInwardMotionAndAnUnsafeThumbSweep)
{
  auto input = airwall_incident_input();
  auto config = test_config();
  config.allow_airwall_recovery = true;
  input.q_nominal[3] -= 0.005;
  const auto inward = x2::cbf::verify_arm_target(input, config);
  EXPECT_FALSE(inward.thumb_target_accepted);
  EXPECT_FALSE(inward.recovery_verified);
  input = airwall_incident_input();
  input.q_thumb_urdf[0] = 0.06153846153846155;
  input.q_thumb_urdf[1] = -1.519323076923077;
  const auto hand_alone = x2::cbf::verify_arm_target(input, config);
  EXPECT_FALSE(hand_alone.thumb_target_accepted);  // Endpoint improvement cannot authorize a deeper path.
  EXPECT_FALSE(hand_alone.recovery_verified);
  input.q_nominal[3] += 0.04;
  const auto coordinated = x2::cbf::verify_arm_target(input, config);
  ASSERT_TRUE(coordinated.thumb_target_accepted);
  EXPECT_TRUE(coordinated.recovery_verified);
  expect_nonworsening_recovery_path(input, config);
}

TEST(CbfKernel, ExistingProxyOverlapCanRetreatButCannotDeepen)
{
  auto input = airwall_incident_input();
  input.q_current[3] -= 0.01;
  input.q_nominal = input.q_current;
  input.q_nominal[8] = input.command_joint_limits[8][1];
  input.q_nominal[3] += 0.005;
  auto config = test_config();
  config.allow_airwall_recovery = true;
  const auto outward = x2::cbf::verify_arm_target(input, config);
  ASSERT_LT(outward.input_min_distance_m, 0.0);
  ASSERT_TRUE(outward.thumb_target_accepted);
  EXPECT_EQ(outward.status, FilterStatus::PenetrationRecovery);
  EXPECT_GT(outward.output_min_distance_m, outward.input_min_distance_m);
  expect_nonworsening_recovery_path(input, config);
  input.q_nominal[3] = input.q_current[3] - 0.005;
  EXPECT_FALSE(x2::cbf::verify_arm_target(input, config).thumb_target_accepted);
}

TEST(CbfKernel, ANewThumbGestureMustNotAssumeTheArmAlreadyRetreated)
{
  auto input = airwall_incident_input();
  input.q_thumb_urdf[0] = 0.06153846153846155;
  input.q_thumb_urdf[1] = -1.519323076923077;
  auto config = test_config();
  config.allow_airwall_recovery = true;
  // The right shoulder is an observation outside the goal box. This query
  // must retain it as geometry rather than rejecting/clipping that coordinate.
  ASSERT_GT(input.q_current[8], input.command_joint_limits[8][1]);
  const auto premature = x2::cbf::verify_thumb_target_with_arm_held(input, config);
  EXPECT_FALSE(premature.thumb_target_accepted);
  input.q_current[3] += 0.06;
  const auto after_retreat = x2::cbf::verify_thumb_target_with_arm_held(input, config);
  ASSERT_TRUE(after_retreat.thumb_target_accepted);
  EXPECT_TRUE(after_retreat.nonlinear_verified);
  EXPECT_EQ(after_retreat.q_safe, input.q_current);  // A geometry query, never an arm command.
}

TEST(CbfKernel, ProjectionAndExactVerificationHonorTheCallerCommandBox)
{
  auto input = baseline_input();
  for (std::size_t j = 0; j < 14; ++j) {
    input.command_joint_limits[j] = {input.q_current[j] - 0.002, input.q_current[j] + 0.002};
    input.q_nominal[j] += j % 2 ? -0.04 : 0.04;
  }
  const auto projected = x2::cbf::filter_arm_target(input, test_config());
  ASSERT_TRUE(projected.thumb_target_accepted);
  for (std::size_t j = 0; j < 14; ++j) {
    EXPECT_GE(projected.q_safe[j], input.command_joint_limits[j][0]);
    EXPECT_LE(projected.q_safe[j], input.command_joint_limits[j][1]);
  }
  const auto exact = x2::cbf::verify_arm_target(input, test_config());
  EXPECT_EQ(exact.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(exact.invalid_input_reason, "nominal_arm_out_of_range");
}

TEST(CbfKernel, AcceptsTheFrozenNaturalStandingPose)
{
  const auto input = baseline_input();
  ASSERT_GT(geometry(input, false).minimum_distance, 0.0);

  const auto result = x2::cbf::filter_arm_target(input, test_config());
  EXPECT_EQ(result.status, FilterStatus::Normal);
  EXPECT_EQ(result.q_safe, input.q_nominal);
  EXPECT_TRUE(result.nonlinear_verified);
  EXPECT_TRUE(result.thumb_target_accepted);
  EXPECT_FALSE(result.deadline_missed);
  EXPECT_GE(result.output_min_safety_margin_m, 0.0);
  expect_finite_limited_arm(result);
  EXPECT_STREQ(result.invalid_input_reason, "");
  EXPECT_EQ(result.invalid_input_index, -1);

  const auto verified = x2::cbf::verify_arm_target(input, test_config());
  EXPECT_EQ(verified.status, FilterStatus::Normal);
  EXPECT_EQ(verified.q_safe, input.q_nominal);
  EXPECT_TRUE(verified.nonlinear_verified);
  EXPECT_TRUE(verified.thumb_target_accepted);
  EXPECT_EQ(verified.projection_iterations, 0);
  EXPECT_STREQ(verified.invalid_input_reason, "");
  EXPECT_EQ(verified.invalid_input_index, -1);
}

TEST(CbfKernel, CurrentArmRangeDiagnosticPreservesTheRawExcessAndOriginalFallback)
{
  constexpr std::size_t joint = 3;  // left_elbow_joint in the frozen arm14 order.
  const auto bounds = x2::cbf::generated::kArmLimits[joint];
  auto input = baseline_input();
  input.q_current[joint] = bounds[1] + 1e-6;
  auto expected_hold = input.q_current;
  expected_hold[joint] = bounds[1];  // Existing invalid-current clamp, not new recovery.
  for (bool verify_only : {false, true}) {
    SCOPED_TRACE(verify_only);
    const auto result = verify_only ? x2::cbf::verify_arm_target(input, test_config()) :
      x2::cbf::filter_arm_target(input, test_config());
    EXPECT_EQ(result.status, FilterStatus::InvalidInputFallback);
    EXPECT_FALSE(result.thumb_target_accepted);
    EXPECT_FALSE(result.nonlinear_verified);
    EXPECT_EQ(result.q_safe, expected_hold);
    EXPECT_FALSE(result.deadline_missed);
    EXPECT_STREQ(result.invalid_input_reason, "current_arm_out_of_range");
    ASSERT_EQ(result.invalid_input_index, static_cast<int>(joint));
    EXPECT_STREQ(x2::cbf::generated::kArmJointNames[result.invalid_input_index], "left_elbow_joint");
    EXPECT_DOUBLE_EQ(result.invalid_input_value, input.q_current[joint]);
    EXPECT_DOUBLE_EQ(result.invalid_input_lower, bounds[0]);
    EXPECT_DOUBLE_EQ(result.invalid_input_upper, bounds[1]);
    EXPECT_GT(result.invalid_input_value, result.invalid_input_upper);
    expect_finite_limited_arm(result);
  }
}

TEST(CbfKernel, RecordedShoulderFeedbackUsesNativeDomainWithoutWideningCommands)
{
  // Exact first failed frame from run-1108-035505, event 1, call 200.
  // The native profile permits +0.061 rad; +0.05236 is the operator envelope.
  auto input = baseline_input();
  input.q_current = {
    0.2698812484741211, -0.027132034301757813, 0.041512489318847656, -0.9933342933654785,
    -0.0018215179443359375, 0.07877445220947266, 0.061608314514160156,
    0.2936573028564453, 0.052634239196777344, -0.28905534744262695, -1.0819206237792969,
    0.009299278259277344, 0.09212589263916016, -0.12264442443847656};
  input.q_nominal = {
    0.23605849349154875, 0.020327284868150518, 0.06284284189253467, -1.0137120862253512,
    -0.013004390778715588, 0.08979877085391544, 0.09377184852928588,
    0.28129737927273946, 0.03445081610300152, -0.35012300453062606, -1.1090341908946615,
    0.014577933023973443, 0.10682537988496096, -0.14390409509995367};
  input.q_thumb_current_urdf = {
    0.06105468749999998, -0.996696875, 0.0, 0.0872665, -0.7094284610304067, 0.0};
  input.q_thumb_urdf = {
    0.016761575722854145, -0.8937160885221986, 0.0, 0.0872665, -0.7094284610304067, 0.0};
  input.thumb_current_valid = true;
  input.hands_installed = {true, false};
  input.thumb_joint_limits_urdf = historical_native_thumb_bounds();
  input.thumb_joint_limits_urdf[2] = x2::cbf::kRigidThumbLimits[2];
  input.thumb_joint_limits_urdf[3] = x2::cbf::kRigidThumbLimits[3];
  auto config = test_config();
  config.max_step_rad = 6.5;  // Adapter separately bounds steps about the published target.
  config.path_check_substeps = 64;
  const auto strict = x2::cbf::verify_arm_target(input, config);
  ASSERT_EQ(strict.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(strict.invalid_input_reason, "current_arm_out_of_range");
  EXPECT_EQ(strict.invalid_input_index, 8);

  // From x2_ultra_omnihand.yaml vendor.arm.limits, not an invented tolerance.
  input.current_arm_joint_limits = {{
    {-3.08, 2.04}, {-0.061, 3.0456}, {-2.556, 2.556}, {-2.3556, 0.0},
    {-2.556, 2.556}, {-0.5236, 0.5236}, {-1.5097, 0.724},
    {-3.08, 2.04}, {-3.0456, 0.061}, {-2.556, 2.556}, {-2.3556, 0.0},
    {-2.556, 2.556}, {-0.5236, 0.5236}, {-0.724, 1.5097}}};
  const auto measured = input.q_current;
  const auto raw_distances = geometry(input, true);
  ASSERT_GT(raw_distances.minimum_distance, 0.0);
  const auto verified = x2::cbf::verify_arm_target(input, config);
  EXPECT_EQ(verified.status, FilterStatus::Normal);
  EXPECT_TRUE(verified.thumb_target_accepted);
  EXPECT_TRUE(verified.nonlinear_verified);
  EXPECT_EQ(verified.q_safe, input.q_nominal);
  EXPECT_NEAR(verified.input_min_distance_m, raw_distances.minimum_distance, 1e-12);
  EXPECT_EQ(input.q_current, measured);  // No silent clamping of physical observations.
  expect_finite_limited_arm(verified);

  auto illegal_target = input;
  illegal_target.q_nominal[8] = 0.055;  // Native-valid feedback, but illegal command.
  const auto rejected = x2::cbf::verify_arm_target(illegal_target, config);
  EXPECT_EQ(rejected.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(rejected.invalid_input_reason, "nominal_arm_out_of_range");
  EXPECT_FALSE(rejected.thumb_target_accepted);
  expect_finite_limited_arm(rejected);
}

TEST(CbfKernel, InvalidMeasuredDomainsRemainRejectedAndDiagnosed)
{
  for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto input = baseline_input();
    if (invalid_case == 0) {input.current_arm_joint_limits[8][0] = std::numeric_limits<double>::quiet_NaN();}
    else if (invalid_case == 1) {input.current_arm_joint_limits[8] = {1.0, -1.0};}
    else {input.current_arm_joint_limits[8][1] = std::numeric_limits<double>::infinity();}
    const auto rejected = x2::cbf::verify_arm_target(input, test_config());
    EXPECT_EQ(rejected.status, FilterStatus::InvalidInputFallback);
    EXPECT_STREQ(rejected.invalid_input_reason, "current_arm_limits_invalid");
    EXPECT_EQ(rejected.invalid_input_index, 8);
    EXPECT_FALSE(rejected.thumb_target_accepted);
    expect_finite_limited_arm(rejected);
  }
}

TEST(CbfKernel, RecordedNativeBoundaryOvershootKeepsRawGeometryAndStrictGoals)
{
  // Exact third fault from run-1108-043239. Native goal upper=0.061;
  // a raw encoder sample at 0.061646 must not masquerade as a target violation.
  auto input = baseline_input();
  input.q_current = {
    0.22271156311035156, -0.025406360626220703, 0.09500980377197266,
    -0.8127107620239258, -0.0033555030822753906, 0.061226844787597656,
    0.01964569091796875, 0.2980670928955078, 0.0616459846496582,
    -0.2424616813659668, -0.9563279151916504, -0.025789737701416016,
    0.06275272369384766, -0.050927162170410156};
  input.q_nominal = {
    0.19858145522292833, 0.004963689468073048, 0.1351866457028887,
    -0.79112446712693, -0.005856080141253305, 0.06581175530302923,
    0.011338952617661243, 0.3087563219920679, 0.03490658503988659,
    -0.2897452618821158, -0.9499978730755176, -0.03419837501844017,
    0.0696628720216186, -0.048841797172931564};
  input.q_thumb_current_urdf = {
    0.03490234375000001, -1.0332300781250001, 0.0, 0.0872665, -0.7094284610304067, 0.0};
  input.q_thumb_urdf = {
    0.013187928368633778, -1.0424000000000002, 0.0, 0.0872665, -0.7094284610304067, 0.0};
  input.thumb_current_valid = true;
  input.hands_installed = {true, false};
  input.thumb_joint_limits_urdf = historical_native_thumb_bounds();
  input.thumb_joint_limits_urdf[2] = x2::cbf::kRigidThumbLimits[2];
  input.thumb_joint_limits_urdf[3] = x2::cbf::kRigidThumbLimits[3];
  input.current_arm_joint_limits[8][1] = 0.061;
  auto config = test_config();
  config.max_step_rad = 6.5;
  config.path_check_substeps = 64;
  const auto strict = x2::cbf::verify_arm_target(input, config);
  ASSERT_EQ(strict.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(strict.invalid_input_reason, "current_arm_out_of_range");

  constexpr double allowance = 0.017453292519943295;
  input.current_arm_joint_limits[8][1] += allowance;
  for (double observed : {0.061070919036865234, 0.0616459846496582, 0.061 + allowance}) {
    SCOPED_TRACE(observed);
    input.q_current[8] = observed;
    const auto raw_distance = geometry(input, true).minimum_distance;
    const auto result = x2::cbf::verify_arm_target(input, config);
    ASSERT_TRUE(result.thumb_target_accepted);
    EXPECT_TRUE(result.nonlinear_verified);
    EXPECT_EQ(result.q_safe, input.q_nominal);
    EXPECT_NEAR(result.input_min_distance_m, raw_distance, 1e-12);
    expect_finite_limited_arm(result);
  }
  input.q_current[8] = 0.0616459846496582;
  input.q_nominal[8] = 0.055;
  const auto illegal = x2::cbf::verify_arm_target(input, config);
  EXPECT_EQ(illegal.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(illegal.invalid_input_reason, "nominal_arm_out_of_range");
}

TEST(CbfKernel, NativeObservationDomainDoesNotHideGeometryByClampingFeedback)
{
  // Select a deterministic boundary pose whose raw geometry differs from a
  // clipped observation. This checks the geometry source, even for an unsafe pose.
  bool checked = false;
  for (std::size_t joint = 0; joint < 14 && !checked; ++joint) {
    for (std::size_t side = 0; side < 2 && !checked; ++side) {
      auto input = baseline_input();
      const double bound = x2::cbf::generated::kArmLimits[joint][side];
      input.q_current[joint] = bound + (side == 0 ? -0.0005 : 0.0005);
      input.current_arm_joint_limits[joint][side] = input.q_current[joint];
      input.q_nominal[joint] = bound;
      auto clipped = input;
      clipped.q_current[joint] = bound;
      const double actual_distance = geometry(input, true).minimum_distance;
      const double clipped_distance = geometry(clipped, true).minimum_distance;
      if (!std::isfinite(actual_distance) || !std::isfinite(clipped_distance) ||
        std::abs(actual_distance - clipped_distance) <= 1e-8) {continue;}
      for (bool verify_only : {false, true}) {
        SCOPED_TRACE(verify_only);
        const auto result = verify_only ? x2::cbf::verify_arm_target(input, test_config()) :
          x2::cbf::filter_arm_target(input, test_config());
        EXPECT_NE(result.status, FilterStatus::InvalidInputFallback);
        EXPECT_NEAR(result.input_min_distance_m, actual_distance, 1e-12);
        EXPECT_GT(std::abs(result.input_min_distance_m - clipped_distance), 1e-8);
        expect_finite_limited_arm(result);
      }
      checked = true;
    }
  }
  EXPECT_TRUE(checked);
}

TEST(CbfKernel, EncoderObservationRangeCanBeIndependentOfGoalAdmissionWithoutClipping)
{
  auto input = baseline_input();
  input.hands_installed = {true, false};
  input.check_current_arm_goal_range = false;
  auto config = test_config();
  config.path_check_substeps = 64;
  for (std::size_t joint = 0; joint < input.q_current.size(); ++joint) {
    for (std::size_t side = 0; side < 2; ++side) {
      for (double excess : {0.0001, 0.02, 0.2}) {
        SCOPED_TRACE(joint);
        SCOPED_TRACE(side);
        SCOPED_TRACE(excess);
        input.q_current = x2::cbf::generated::kNaturalStanding;
        input.q_current[joint] = input.current_arm_joint_limits[joint][side] +
          (side == 0 ? -excess : excess);
        const auto raw = geometry(input, true);
        const auto result = x2::cbf::verify_arm_target(input, config);
        // Geometry may reject a real collision; travel bands cannot reject an
        // encoder observation before its actual geometry has been evaluated.
        EXPECT_NE(result.status, FilterStatus::InvalidInputFallback);
        EXPECT_NEAR(result.input_min_distance_m, raw.minimum_distance, 1e-12);
        expect_finite_limited_arm(result);
      }
    }
  }
}

TEST(CbfKernel, ObservationRangeBypassKeepsFiniteConfigurationAndTargetChecks)
{
  auto input = baseline_input();
  input.check_current_arm_goal_range = false;
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    input.q_current[8] = invalid;
    const auto result = x2::cbf::verify_arm_target(input, test_config());
    EXPECT_EQ(result.status, FilterStatus::InvalidInputFallback);
    EXPECT_STREQ(result.invalid_input_reason, "current_arm_nonfinite");
    EXPECT_FALSE(result.thumb_target_accepted);
  }
  input.q_current = x2::cbf::generated::kNaturalStanding;
  input.q_nominal[8] = x2::cbf::generated::kArmLimits[8][1] + 0.0001;
  const auto target = x2::cbf::verify_arm_target(input, test_config());
  EXPECT_EQ(target.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(target.invalid_input_reason, "nominal_arm_out_of_range");
  EXPECT_FALSE(target.thumb_target_accepted);
  input.q_nominal = input.q_current;
  input.current_arm_joint_limits[8] = {1.0, -1.0};
  const auto bad_config = x2::cbf::verify_arm_target(input, test_config());
  EXPECT_EQ(bad_config.status, FilterStatus::InvalidInputFallback);
  EXPECT_STREQ(bad_config.invalid_input_reason, "current_arm_limits_invalid");
}

TEST(CbfKernel, ThumbRangeDiagnosticsDistinguishCurrentAndTargetUsingPose6Indices)
{
  for (bool invalid_current : {true, false}) {
    SCOPED_TRACE(invalid_current);
    auto input = baseline_input();
    input.thumb_current_valid = true;
    input.thumb_joint_limits_urdf = historical_native_thumb_bounds();
    // Use the existing mapped-bounds fixture: diagnostics must report the
    // actual override, and pose6 slots 3/4 rather than compact axis indices 2/3.
    const std::size_t axis = invalid_current ? 2 : 3;
    const std::size_t slot = x2::cbf::kRigidThumbIndices[axis];
    const auto bounds = input.thumb_joint_limits_urdf[axis];
    const double value = invalid_current ? bounds[1] + 1e-6 : bounds[0] - 1e-6;
    if (invalid_current) {input.q_thumb_current_urdf[slot] = value;}
    else {input.q_thumb_urdf[slot] = value;}
    for (bool verify_only : {false, true}) {
      SCOPED_TRACE(verify_only);
      const auto result = verify_only ? x2::cbf::verify_arm_target(input, test_config()) :
        x2::cbf::filter_arm_target(input, test_config());
      EXPECT_EQ(result.status, FilterStatus::InvalidInputFallback);
      EXPECT_FALSE(result.thumb_target_accepted);
      EXPECT_EQ(result.q_safe, input.q_current);
      EXPECT_EQ(result.nonlinear_verified, !verify_only && !invalid_current);
      EXPECT_FALSE(result.deadline_missed);
      EXPECT_STREQ(result.invalid_input_reason,
        invalid_current ? "current_thumb_out_of_range" : "target_thumb_out_of_range");
      EXPECT_EQ(result.invalid_input_index, static_cast<int>(slot));
      EXPECT_DOUBLE_EQ(result.invalid_input_value, value);
      EXPECT_DOUBLE_EQ(result.invalid_input_lower, bounds[0]);
      EXPECT_DOUBLE_EQ(result.invalid_input_upper, bounds[1]);
      expect_finite_limited_arm(result);
    }
  }
}

TEST(CbfKernel, NonfiniteDiagnosticsIdentifyTheInputWithoutChangingRejectionBehavior)
{
  constexpr std::array<const char *, 4> reasons{
    "current_arm_nonfinite", "nominal_arm_nonfinite", "current_thumb_nonfinite", "target_thumb_nonfinite"};
  constexpr std::array<int, 4> indices{5, 3, 3, 4};
  for (std::size_t invalid_case = 0; invalid_case < reasons.size(); ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto input = baseline_input();
    input.thumb_current_valid = true;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (invalid_case == 0) {input.q_current[5] = nan;}
    else if (invalid_case == 1) {input.q_nominal[3] = nan;}
    else if (invalid_case == 2) {input.q_thumb_current_urdf[3] = nan;}
    else {input.q_thumb_urdf[4] = nan;}
    for (bool verify_only : {false, true}) {
      SCOPED_TRACE(verify_only);
      const auto result = verify_only ? x2::cbf::verify_arm_target(input, test_config()) :
        x2::cbf::filter_arm_target(input, test_config());
      EXPECT_EQ(result.status, FilterStatus::InvalidInputFallback);
      EXPECT_FALSE(result.thumb_target_accepted);
      EXPECT_FALSE(result.deadline_missed);
      EXPECT_EQ(result.q_safe, x2::cbf::generated::kNaturalStanding);
      // A filter may certify only its unchanged, valid actual hold. The exact
      // verifier never authorizes a request rejected by input validation.
      EXPECT_EQ(result.nonlinear_verified, !verify_only && (invalid_case == 1 || invalid_case == 3));
      EXPECT_STREQ(result.invalid_input_reason, reasons[invalid_case]);
      EXPECT_EQ(result.invalid_input_index, indices[invalid_case]);
      EXPECT_TRUE(std::isnan(result.invalid_input_value));
      expect_finite_limited_arm(result);
    }
  }
}

TEST(CbfKernel, ProjectsTheKnownArmRequest)
{
  // phase_c_tests.cpp::hand_passthrough_and_invalid_fallback.
  auto input = baseline_input();
  input.q_nominal[0] += 0.04;
  const auto config = test_config();
  const auto result = x2::cbf::filter_arm_target(input, config);

  EXPECT_TRUE(
    result.status == FilterStatus::Projected ||
    result.status == FilterStatus::IterationLimitSafe);
  EXPECT_NE(result.q_safe, input.q_nominal);
  EXPECT_NE(result.q_safe, input.q_current);
  EXPECT_LT(result.q_safe[0], input.q_nominal[0]);
  EXPECT_TRUE(result.nonlinear_verified);
  EXPECT_TRUE(result.thumb_target_accepted);
  EXPECT_FALSE(result.deadline_missed);
  EXPECT_GE(result.output_min_distance_m, 0.0);
  EXPECT_GE(result.output_min_safety_margin_m, 0.0);
  for (std::size_t joint = 0; joint < result.q_safe.size(); ++joint) {
    EXPECT_LE(std::abs(result.q_safe[joint] - input.q_current[joint]), config.max_step_rad + 1e-12);
  }
  expect_finite_limited_arm(result);
}

TEST(CbfKernel, InvalidInputsHoldCurrentAndRefuseTheHandTarget)
{
  // rigid_thumb_tests.cpp::invalid_active_angles_are_rejected_without_clamping.
  auto invalid_nominal = moving_input();
  invalid_nominal.q_nominal[3] = std::numeric_limits<double>::quiet_NaN();
  const auto nominal = x2::cbf::filter_arm_target(invalid_nominal, test_config());
  EXPECT_EQ(nominal.status, FilterStatus::InvalidInputFallback);
  EXPECT_EQ(nominal.q_safe, invalid_nominal.q_current);
  EXPECT_FALSE(nominal.thumb_target_accepted);
  EXPECT_TRUE(nominal.nonlinear_verified);  // Only the actual held pose is verified.
  expect_finite_limited_arm(nominal);

  auto invalid_mapping = moving_input();
  invalid_mapping.thumb_pose_valid = false;
  const auto unmapped = x2::cbf::filter_arm_target(invalid_mapping, test_config());
  EXPECT_EQ(unmapped.status, FilterStatus::InvalidInputFallback);
  EXPECT_EQ(unmapped.q_safe, invalid_mapping.q_current);
  EXPECT_FALSE(unmapped.thumb_target_accepted);
  EXPECT_FALSE(unmapped.nonlinear_verified);
  expect_finite_limited_arm(unmapped);
}

TEST(CbfKernel, DeadlineFallbackHoldsTheActualCurrentPose)
{
  // rigid_thumb_tests.cpp::dynamic_deadline_holds_actual_current_thumb.
  const auto input = moving_input();
  auto config = test_config();
  config.deadline_us = 1e-9;
  config.deadline_guard_us = 0.0;
  const auto result = x2::cbf::filter_arm_target(input, config);

  EXPECT_EQ(result.status, FilterStatus::DeadlineFallback);
  EXPECT_TRUE(result.deadline_missed);
  EXPECT_EQ(result.q_safe, input.q_current);
  EXPECT_FALSE(result.thumb_target_accepted);
  EXPECT_TRUE(result.nonlinear_verified);
  EXPECT_NEAR(result.output_min_distance_m, geometry(input, true).minimum_distance, 1e-12);
  expect_finite_limited_arm(result);
}

TEST(CbfKernel, RejectsTheKnownPenetratingThumbTarget)
{
  // Exact kPenetratingThumbTarget and zero-arm-step case from rigid_thumb_tests.cpp.
  auto input = baseline_input();
  input.thumb_current_valid = true;
  input.q_thumb_urdf = {
    1.1287931351854765, -1.3127632185785869, 0.0,
    -0.50685251960405675, -0.018401995180100439, 0.0,
  };
  auto config = test_config();
  config.max_step_rad = 0.0;
  ASSERT_TRUE(x2::cbf::valid_rigid_thumb_pose(input.q_thumb_urdf));
  const auto current = geometry(input, true);
  ASSERT_GT(current.minimum_distance, 0.0);
  ASSERT_LT(geometry(input, false).minimum_distance, -0.003);

  const auto result = x2::cbf::filter_arm_target(input, config);
  EXPECT_EQ(result.status, FilterStatus::ThumbMotionRejected);
  EXPECT_EQ(result.q_safe, input.q_current);
  EXPECT_FALSE(result.thumb_target_accepted);
  EXPECT_FALSE(result.deadline_missed);
  EXPECT_TRUE(result.nonlinear_verified);  // A verified hold does not authorize the target hand pose.
  EXPECT_GT(result.output_min_distance_m, 0.0);
  EXPECT_NEAR(result.output_min_distance_m, current.minimum_distance, 1e-12);
  expect_finite_limited_arm(result);
}

TEST(CbfKernel, HandPassthroughDoesNotValidateTheSdkMapping)
{
  // rigid_thumb_tests.cpp::twenty_hand_commands_remain_exact_passthrough.
  auto input = baseline_input();
  input.q_nominal[0] += 0.04;
  const auto filtered = x2::cbf::filter_arm_target(input, test_config());
  ASSERT_TRUE(filtered.thumb_target_accepted);
  ASSERT_TRUE(filtered.nonlinear_verified);

  x2::cbf::UpperBodySafetyCommand command;
  command.arm_pos = input.q_nominal;
  for (std::size_t hand = 0; hand < command.hand_pos.size(); ++hand) {
    const double magnitude = 100.25 + 3.5 * static_cast<double>(hand);
    command.hand_pos[hand] = hand % 2 == 0 ? magnitude : -magnitude;
  }
  const auto original = command;
  const auto passthrough = x2::cbf::pass_through_hand_and_arm(command, filtered.q_safe);
  EXPECT_EQ(passthrough.safe_arm_pos, filtered.q_safe);
  EXPECT_EQ(passthrough.hand_pos, original.hand_pos);
  EXPECT_EQ(command.arm_pos, original.arm_pos);
  EXPECT_EQ(command.hand_pos, original.hand_pos);
  EXPECT_FALSE(x2::cbf::extract_thumb_commands(command.hand_pos).mapping_valid);
}

TEST(CbfKernel, DefaultThumbBoundsStayStrictAndMappedNativeOpenNeedsExplicitBounds)
{
  EXPECT_EQ(baseline_input().thumb_joint_limits_urdf, x2::cbf::kRigidThumbLimits);
  for (std::size_t axis = 0; axis < x2::cbf::kRigidThumbIndices.size(); ++axis) {
    SCOPED_TRACE(axis);
    const auto slot = x2::cbf::kRigidThumbIndices[axis];
    for (std::size_t end = 0; end < 2; ++end) {
      auto pose = x2::cbf::kRigidThumbBaselineUrdf;
      pose[slot] = x2::cbf::kRigidThumbLimits[axis][end];
      EXPECT_TRUE(x2::cbf::valid_rigid_thumb_pose(pose));
      pose[slot] += end == 0 ? -1e-6 : 1e-6;
      EXPECT_FALSE(x2::cbf::valid_rigid_thumb_pose(pose));
    }
  }
  // Native open == 0 mapped by an unverified fixture, not SDK calibration.
  const std::array<double, 6> mapped_open{0.0, -1.6424, 0.0, -0.576, -1.6424, 0.0};
  EXPECT_FALSE(x2::cbf::valid_rigid_thumb_pose(mapped_open));
  EXPECT_TRUE(x2::cbf::valid_rigid_thumb_pose(mapped_open, historical_native_thumb_bounds()));
}

TEST(CbfKernel, BothThumbInputsUseExplicitBoundsAndRejectNonfiniteOrReversedBounds)
{
  const std::array<double, 6> mapped_open{0.0, -1.6424, 0.0, -0.576, -1.6424, 0.0};
  for (bool change_current : {false, true}) {
    SCOPED_TRACE(change_current);
    auto input = baseline_input();
    input.thumb_current_valid = true;
    (change_current ? input.q_thumb_current_urdf : input.q_thumb_urdf) = mapped_open;
    EXPECT_EQ(x2::cbf::filter_arm_target(input, test_config()).status, FilterStatus::InvalidInputFallback);
    input.thumb_joint_limits_urdf = historical_native_thumb_bounds();
    // Accepting an input domain does not certify this pose's collision safety.
    EXPECT_NE(x2::cbf::filter_arm_target(input, test_config()).status, FilterStatus::InvalidInputFallback);
  }
  for (int invalid_case = 0; invalid_case < 4; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto input = baseline_input();
    auto & bounds = input.thumb_joint_limits_urdf;
    if (invalid_case == 0) {
      bounds[0][0] = std::numeric_limits<double>::quiet_NaN();
    } else if (invalid_case == 1) {
      bounds[0][1] = std::numeric_limits<double>::quiet_NaN();
    } else if (invalid_case == 2) {
      bounds[0][1] = std::numeric_limits<double>::infinity();
    } else {
      bounds[0] = {1.0, 0.0};
    }
    EXPECT_FALSE(x2::cbf::valid_rigid_thumb_pose(input.q_thumb_urdf, bounds));
    const auto rejected = x2::cbf::filter_arm_target(input, test_config());
    EXPECT_EQ(rejected.status, FilterStatus::InvalidInputFallback);
    EXPECT_FALSE(rejected.thumb_target_accepted);
    EXPECT_EQ(rejected.q_safe, input.q_current);
  }
}

TEST(CbfKernel, ExactVerifierPreservesTheStandingTargetDespiteShoulderTrackingLag)
{
  auto input = baseline_input();
  input.thumb_current_valid = true;
  // Same frozen-URDF lag fixture as the unchanged CbfSafety TrackingLag case.
  // The measured shoulder is low; the already published target has not moved.
  input.q_current[0] -= 0.06;
  const auto config = test_config();
  ASSERT_GT(std::abs(input.q_nominal[0] - input.q_current[0]), config.max_step_rad);

  const auto verified = x2::cbf::verify_arm_target(input, config);
  ASSERT_EQ(verified.status, FilterStatus::Normal);
  EXPECT_EQ(verified.q_safe, input.q_nominal);
  EXPECT_NE(verified.q_safe, input.q_current);
  EXPECT_TRUE(verified.nonlinear_verified);
  EXPECT_TRUE(verified.thumb_target_accepted);
  EXPECT_FALSE(verified.deadline_missed);
  EXPECT_EQ(verified.projection_iterations, 0);
  EXPECT_GE(verified.input_min_safety_margin_m, 0.0);
  EXPECT_GE(verified.output_min_safety_margin_m, 0.0);
  expect_finite_limited_arm(verified);
}

TEST(CbfKernel, ExactVerifierRejectsTheKnownPenetratingThumbAndInvalidInputs)
{
  auto input = baseline_input();
  input.thumb_current_valid = true;
  input.q_thumb_urdf = {
    1.1287931351854765, -1.3127632185785869, 0.0,
    -0.50685251960405675, -0.018401995180100439, 0.0,
  };
  ASSERT_TRUE(x2::cbf::valid_rigid_thumb_pose(input.q_thumb_urdf));
  ASSERT_GT(geometry(input, true).minimum_distance, 0.0);
  ASSERT_LT(geometry(input, false).minimum_distance, -0.003);
  const auto unsafe = x2::cbf::verify_arm_target(input, test_config());
  EXPECT_EQ(unsafe.status, FilterStatus::ThumbMotionRejected);
  EXPECT_EQ(unsafe.q_safe, input.q_current);
  EXPECT_FALSE(unsafe.nonlinear_verified);
  EXPECT_FALSE(unsafe.thumb_target_accepted);
  EXPECT_FALSE(unsafe.deadline_missed);
  EXPECT_EQ(unsafe.projection_iterations, 0);

  for (int invalid_case = 0; invalid_case < 4; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto invalid = baseline_input();
    invalid.thumb_current_valid = true;
    if (invalid_case == 0) {
      invalid.q_nominal[3] = std::numeric_limits<double>::quiet_NaN();
    } else if (invalid_case == 1) {
      invalid.q_current[0] = std::numeric_limits<double>::infinity();
    } else if (invalid_case == 2) {
      invalid.q_thumb_current_urdf[0] = std::numeric_limits<double>::quiet_NaN();
    } else {
      invalid.thumb_joint_limits_urdf[0] = {1.0, 0.0};
    }
    const auto rejected = x2::cbf::verify_arm_target(invalid, test_config());
    EXPECT_EQ(rejected.status, FilterStatus::InvalidInputFallback);
    EXPECT_FALSE(rejected.nonlinear_verified);
    EXPECT_FALSE(rejected.thumb_target_accepted);
    EXPECT_FALSE(rejected.deadline_missed);
    EXPECT_EQ(rejected.projection_iterations, 0);
    expect_finite_limited_arm(rejected);
  }
}

TEST(CbfKernel, ExactVerifierDeadlineDoesNotAuthorizeTheRequestedTarget)
{
  const auto input = moving_input();
  auto config = test_config();
  config.deadline_us = 1e-9;
  config.deadline_guard_us = 0.0;
  const auto late = x2::cbf::verify_arm_target(input, config);

  EXPECT_EQ(late.status, FilterStatus::DeadlineFallback);
  EXPECT_TRUE(late.deadline_missed);
  EXPECT_FALSE(late.nonlinear_verified);
  EXPECT_FALSE(late.thumb_target_accepted);
  EXPECT_EQ(late.q_safe, input.q_current);
  EXPECT_NE(late.q_safe, input.q_nominal);
  EXPECT_EQ(late.projection_iterations, 0);
  expect_finite_limited_arm(late);
}

TEST(CbfKernel, AbsentRightHandCollisionsDoNotContributeToDistancesOrGradients)
{
  using x2::cbf::PrimitiveId;
  auto input = proxy_input();
  auto both = x2::cbf::forward_kinematics(input);
  const auto both_gradients = x2::cbf::evaluate_distance_gradients(input);
  input.hands_installed = {true, false};
  auto left_only = x2::cbf::forward_kinematics(input);
  const auto left_gradients = x2::cbf::evaluate_distance_gradients(input);
  ASSERT_TRUE(left_gradients.valid);

  // Inject overlapping poses in this test only; every frozen size stays intact.
  const auto head = static_cast<std::size_t>(PrimitiveId::Head);
  for (const auto primitive : {PrimitiveId::RightPalm, PrimitiveId::RightThumb}) {
    const auto index = static_cast<std::size_t>(primitive);
    both.primitives[index].pose.translation = both.primitives[head].pose.translation;
    left_only.primitives[index].pose.translation = left_only.primitives[head].pose.translation;
  }
  const auto full_distances = x2::cbf::evaluate_proxy_distances(both);
  const auto masked_distances = x2::cbf::evaluate_proxy_distances(left_only);
  ASSERT_LT(full_distances.minimum_distance, 0.0);
  EXPECT_GT(masked_distances.minimum_distance, 0.0);
  EXPECT_TRUE(x2::cbf::proxy_pair_enabled(left_only, masked_distances.minimum_pair_index));
  std::size_t disabled = 0;
  for (std::size_t index = 0; index < x2::cbf::kPrimitivePairCount; ++index) {
    SCOPED_TRACE(index);
    if (!x2::cbf::proxy_pair_enabled(left_only, index)) {
      ++disabled;
      EXPECT_DOUBLE_EQ(masked_distances.distances[index].signed_distance, 1e9);
      EXPECT_TRUE(masked_distances.distances[index].valid);
      EXPECT_TRUE(left_gradients.pairs[index].valid);
      EXPECT_EQ(left_gradients.pairs[index].gradient, x2::cbf::ArmGradient{});
      EXPECT_DOUBLE_EQ(left_gradients.pairs[index].gradient_squared_norm, 0.0);
    } else {
      EXPECT_DOUBLE_EQ(masked_distances.distances[index].signed_distance,
        full_distances.distances[index].signed_distance);
      EXPECT_EQ(left_gradients.pairs[index].gradient, both_gradients.pairs[index].gradient);
      EXPECT_DOUBLE_EQ(left_gradients.pairs[index].gradient_squared_norm,
        both_gradients.pairs[index].gradient_squared_norm);
    }
  }
  EXPECT_EQ(disabled, 16u);
}

TEST(CbfKernel, AbsentRightHandStillChecksLeftHandAndBothArms)
{
  using x2::cbf::PrimitiveId;
  auto input = proxy_input();
  input.hands_installed = {true, false};
  const std::array<x2::cbf::PrimitivePair, 6> retained{{
    {PrimitiveId::LeftPalm, PrimitiveId::Head},
    {PrimitiveId::LeftThumb, PrimitiveId::Head},
    {PrimitiveId::LeftShoulder, PrimitiveId::Torso},
    {PrimitiveId::RightShoulder, PrimitiveId::Torso},
    {PrimitiveId::RightForearm, PrimitiveId::Torso},
    {PrimitiveId::LeftForearm, PrimitiveId::RightForearm}}};
  for (const auto pair : retained) {
    const auto index = find_pair(pair.first, pair.second);
    SCOPED_TRACE(index);
    ASSERT_LT(index, x2::cbf::kPrimitivePairCount);
    auto kinematics = x2::cbf::forward_kinematics(input);
    const auto first = static_cast<std::size_t>(pair.first);
    const auto second = static_cast<std::size_t>(pair.second);
    kinematics.primitives[first].pose.translation = kinematics.primitives[second].pose.translation;
    const auto distances = x2::cbf::evaluate_proxy_distances(kinematics);
    EXPECT_TRUE(x2::cbf::proxy_pair_enabled(kinematics, index));
    EXPECT_TRUE(distances.distances[index].valid);
    EXPECT_LT(distances.distances[index].signed_distance, 0.0);
    EXPECT_LT(distances.minimum_distance, 0.0);
  }
}

TEST(CbfKernel, DefaultBothHandsMatchExplicitBothHandsForFilteringAndVerification)
{
  for (bool projected_request : {false, true}) {
    SCOPED_TRACE(projected_request);
    auto implicit = baseline_input();
    if (projected_request) implicit.q_nominal[0] += 0.04;
    auto explicit_both = implicit;
    explicit_both.hands_installed = {true, true};
    const auto config = test_config();
    const auto first = x2::cbf::filter_arm_target(implicit, config);
    const auto second = x2::cbf::filter_arm_target(explicit_both, config);
    ASSERT_FALSE(first.deadline_missed);
    ASSERT_FALSE(second.deadline_missed);
    EXPECT_TRUE(first.thumb_target_accepted);
    EXPECT_EQ(first.status, second.status);
    EXPECT_EQ(first.q_safe, second.q_safe);
    EXPECT_EQ(first.nonlinear_verified, second.nonlinear_verified);
    EXPECT_EQ(first.thumb_target_accepted, second.thumb_target_accepted);
    EXPECT_DOUBLE_EQ(first.output_min_distance_m, second.output_min_distance_m);
    EXPECT_EQ(first.active_pair_count, second.active_pair_count);
    const auto checked_first = x2::cbf::verify_arm_target(implicit, config);
    const auto checked_second = x2::cbf::verify_arm_target(explicit_both, config);
    ASSERT_FALSE(checked_first.deadline_missed);
    ASSERT_FALSE(checked_second.deadline_missed);
    EXPECT_EQ(checked_first.status, checked_second.status);
    EXPECT_EQ(checked_first.q_safe, checked_second.q_safe);
    EXPECT_EQ(checked_first.nonlinear_verified, checked_second.nonlinear_verified);
    EXPECT_EQ(checked_first.thumb_target_accepted, checked_second.thumb_target_accepted);
    EXPECT_DOUBLE_EQ(checked_first.output_min_distance_m, checked_second.output_min_distance_m);
  }
}

TEST(CbfKernel, AbsentRightThumbModelPoseDoesNotChangeProjectedOrVerifiedOutput)
{
  auto input = baseline_input();
  input.hands_installed = {true, false};
  input.thumb_current_valid = true;
  input.q_nominal[0] += 0.04;
  const auto config = test_config();
  const auto reference = x2::cbf::filter_arm_target(input, config);
  ASSERT_TRUE(reference.thumb_target_accepted);
  auto other = input;
  // Legal model-only placeholders, never fabricated right-hand measurements.
  other.q_thumb_urdf[3] = x2::cbf::kRigidThumbLimits[2][1];
  other.q_thumb_urdf[4] = x2::cbf::kRigidThumbLimits[3][1];
  other.q_thumb_current_urdf = other.q_thumb_urdf;
  const auto filtered = x2::cbf::filter_arm_target(other, config);
  EXPECT_TRUE(filtered.thumb_target_accepted);
  EXPECT_EQ(filtered.status, reference.status);
  EXPECT_EQ(filtered.q_safe, reference.q_safe);
  EXPECT_DOUBLE_EQ(filtered.output_min_distance_m, reference.output_min_distance_m);

  for (bool culling : {false, true}) {
    SCOPED_TRACE(culling);
    auto path = other;
    path.q_nominal = x2::cbf::generated::kNaturalStanding;
    path.q_current[0] -= 0.06;
    auto path_config = config;
    path_config.use_path_separation_culling = culling;
    const auto verified = x2::cbf::verify_arm_target(path, path_config);
    EXPECT_EQ(verified.status, FilterStatus::Normal);
    EXPECT_EQ(verified.q_safe, path.q_nominal);
    EXPECT_TRUE(verified.nonlinear_verified);
    EXPECT_TRUE(verified.thumb_target_accepted);
    EXPECT_FALSE(verified.deadline_missed);
  }
}
