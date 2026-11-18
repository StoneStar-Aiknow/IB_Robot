// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The identity mapping below is ONLY a frozen-URDF test fixture. It does not
// establish, calibrate, or verify any real SDK-to-URDF mapping.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "cbf/generated_constants.hpp"
#include "cbf/kinematics.hpp"
#include "cbf/rigid_thumb.hpp"
#include "cbf/safety_filter.hpp"
#include "nlohmann/json.hpp"

namespace
{

using aimdk_dataplane::CbfDecision;
using aimdk_dataplane::CbfDiagnostics;
using aimdk_dataplane::CbfSafety;
using aimdk_dataplane::CbfSafetyConfig;
using aimdk_dataplane::CbfSnapshot;
using aimdk_dataplane::Measured;
using aimdk_dataplane::Output;
using nlohmann::json;

json urdf_fixture_json()
{
  return {
    {"enabled", true},
    {"mode", "enforce"},
    {"protected_mirror_mode", "policy_stream"},
    {"deadline_us", 100000.0},  // Correctness cases are not timing benchmarks.
    {"feedback_max_age_s", 0.1},
    {"thumb_mapping", {
      {"verified", true},  // Explicitly authorized only within this URDF fixture.
      {"command_scale", {1.0, 1.0, 1.0, 1.0}},
      {"command_offset", {0.0, 0.0, 0.0, 0.0}},
      {"feedback_scale", {1.0, 1.0, 1.0, 1.0}},
      {"feedback_offset", {0.0, 0.0, 0.0, 0.0}},
    }},
  };
}

CbfSafetyConfig fixture_config(const json & value = urdf_fixture_json())
{
  return CbfSafetyConfig::from_json(value.dump(), 50.0);
}

constexpr std::array<double, 4> kHistoricalThumbScale{-1.0, 1.0, 1.0, -1.0};
constexpr std::array<double, 4> kHistoricalThumbOffset{0.0, -1.6424, -0.576, -1.6424};

json historical_native_fixture_json()
{
  auto value = urdf_fixture_json();
  value["mode"] = "shadow";
  value["thumb_mapping"] = {
    // Historical conversion only: neither these values nor this fixture verify SDK calibration.
    {"verified", false},
    {"command_scale", kHistoricalThumbScale},
    {"command_offset", kHistoricalThumbOffset},
    {"feedback_scale", kHistoricalThumbScale},
    {"feedback_offset", kHistoricalThumbOffset},
    // hand_overlay.py HAND_LOW/HAND_HIGH at slots [0, 1, 10, 11].
    {"native_lower", {-1.117, -0.05, -0.03, -1.64}},
    {"native_upper", {0.03, 1.64, 0.87, 0.05}},
  };
  return value;
}

CbfSnapshot urdf_snapshot()
{
  CbfSnapshot snapshot;
  snapshot.epoch = 1;
  snapshot.requested.kind = Output::Kind::kUpperBody;
  snapshot.requested.sequence = 7;
  snapshot.requested.hand_sub_mode = aimdk_dataplane::kHandSubModeDexterousJoint;
  snapshot.requested.head = {0.0, 0.0};
  const auto & standing = x2::cbf::generated::kNaturalStanding;
  snapshot.requested.arm.assign(standing.begin(), standing.end());
  snapshot.requested.hand.assign(20, 0.0);
  constexpr std::array<std::size_t, 4> hand_slots{0, 1, 10, 11};
  Measured thumb;
  for (std::size_t axis = 0; axis < hand_slots.size(); ++axis) {
    const double angle = x2::cbf::kRigidThumbBaselineUrdf[x2::cbf::kRigidThumbIndices[axis]];
    snapshot.requested.hand[hand_slots[axis]] = angle;
    thumb.position.push_back(angle);
  }
  thumb.age_s = 0.0;
  snapshot.thumb = thumb;
  snapshot.arm = Measured{snapshot.requested.arm, 0.0};
  return snapshot;
}

json historical_left_candidate_fixture_json()
{
  auto value = historical_native_fixture_json();
  value["mode"] = "enforce";
  value["hands_installed"] = {true, false};
  return value;
}

CbfSnapshot historical_left_native_snapshot()
{
  auto snapshot = urdf_snapshot();
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  // Algebraic inverse of the historical candidate for an offline URDF fixture.
  // This is not a measured SDK pose or evidence of calibration; only the left
  // two roots are supplied as model feedback, with right-hand slots opaque.
  for (std::size_t axis = 0; axis < slots.size(); ++axis) {
    const double native =
      (snapshot.requested.hand[slots[axis]] - kHistoricalThumbOffset[axis]) / kHistoricalThumbScale[axis];
    snapshot.requested.hand[slots[axis]] = native;
    if (axis < 2) {snapshot.thumb->position[axis] = native;}
  }
  snapshot.thumb->position.resize(2);
  return snapshot;
}

void expect_held_pair(const CbfDecision & decision, const Output & published)
{
  EXPECT_FALSE(decision.accept_new);
  ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(decision.output.arm, published.arm);
  EXPECT_EQ(decision.output.hand, published.hand);
  EXPECT_EQ(decision.output.hand_sub_mode, published.hand_sub_mode);
}

void expect_original_output(const Output & actual, const Output & requested)
{
  EXPECT_EQ(actual.kind, requested.kind);
  EXPECT_EQ(actual.sequence, requested.sequence);
  EXPECT_EQ(actual.hand_sub_mode, requested.hand_sub_mode);
  EXPECT_EQ(actual.arm, requested.arm);
  EXPECT_EQ(actual.hand, requested.hand);
  EXPECT_EQ(actual.head, requested.head);
  EXPECT_EQ(actual.twist, requested.twist);
}

}  // namespace

TEST(CbfSafety, FirstAcceptPreservesAllTwentyHandValuesAndCommitsOnlyAfterPublication)
{
  const auto config = fixture_config();
  ASSERT_TRUE(config.enabled);
  ASSERT_TRUE(config.enforce);
  ASSERT_TRUE(config.mapping_verified);  // Frozen-URDF fixture only.
  EXPECT_EQ(config.mirror_mode, "policy_stream");
  CbfSafety safety(config);
  const auto snapshot = urdf_snapshot();
  const auto decision = safety.filter(snapshot, 1.0);

  ASSERT_TRUE(decision.accept_new);
  ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
  EXPECT_FALSE(decision.modified);
  EXPECT_EQ(decision.output.arm, snapshot.requested.arm);
  ASSERT_EQ(decision.output.hand.size(), 20u);
  EXPECT_EQ(decision.output.hand, snapshot.requested.hand);
  EXPECT_EQ(safety.diagnostics().accepted, 0u);
  EXPECT_EQ(safety.diagnostics().published, 0u);

  safety.committed(decision, 1.0);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().published, 1u);
  EXPECT_EQ(safety.diagnostics().held, 0u);
}

TEST(CbfSafety, TrackingLagDoesNotPullThePublishedShoulderTargetDown)
{
  CbfSafety safety(fixture_config());
  const auto initial = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(initial.accept_new);
  safety.committed(initial, 1.0);

  auto snapshot = urdf_snapshot();
  snapshot.requested = initial.output;
  ++snapshot.requested.sequence;
  ASSERT_TRUE(snapshot.arm.has_value());
  // User-provided frozen-URDF regression: left shoulder pitch tracks 0.06 rad
  // below its safe standing target under load. The nominal command is unchanged.
  constexpr std::size_t shoulder_pitch = 0;
  snapshot.arm->position[shoulder_pitch] = initial.output.arm[shoulder_pitch] - 0.06;

  const auto decision = safety.filter(snapshot, 1.02);
  ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
  ASSERT_EQ(decision.output.arm.size(), initial.output.arm.size());
  EXPECT_EQ(decision.output.arm, initial.output.arm);
  EXPECT_EQ(decision.output.hand, initial.output.hand);
  EXPECT_FALSE(decision.modified);
  EXPECT_GT(decision.output.arm[shoulder_pitch], snapshot.arm->position[shoulder_pitch]);
}

TEST(CbfSafety, AnUncommittedDecisionCannotEstablishAFallback)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  const auto unpublished = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(unpublished.accept_new);
  // No committed() call: this frame has not been published.
  snapshot.requested.hand[0] = std::numeric_limits<double>::quiet_NaN();
  const auto rejected = safety.filter(snapshot, 1.02);

  EXPECT_FALSE(rejected.accept_new);
  EXPECT_EQ(rejected.output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.diagnostics().accepted, 0u);
  EXPECT_EQ(safety.diagnostics().published, 0u);
}

TEST(CbfSafety, NanHandAndNanNominalHoldTheCommittedArmAndHandTogether)
{
  CbfSafety safety(fixture_config());
  const auto initial = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(initial.accept_new);
  safety.committed(initial, 1.0);

  for (int invalid_case = 0; invalid_case < 2; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto snapshot = urdf_snapshot();
    ++snapshot.requested.sequence;
    if (invalid_case == 0) {
      // Existing phase-C shoulder request plus a nonfinite thumb command. An
      // out-of-range one is clamped instead (UnreachableThumbTargetsAreClamped...).
      snapshot.requested.arm[0] += 0.04;
      snapshot.requested.hand[0] = std::numeric_limits<double>::quiet_NaN();
    } else {
      snapshot.requested.arm[3] = std::numeric_limits<double>::quiet_NaN();
    }
    const double now = 1.02 + 0.02 * invalid_case;
    const auto held = safety.filter(snapshot, now);
    expect_held_pair(held, initial.output);
    ASSERT_EQ(held.output.kind, Output::Kind::kUpperBody);
    safety.committed(held, now);
  }
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().held, 2u);
  EXPECT_EQ(safety.diagnostics().published, 3u);
}

TEST(CbfSafety, EnforceWithoutAnExplicitMappingWithholdsTheCommand)
{
  auto value = urdf_fixture_json();
  value.erase("thumb_mapping");
  CbfSafety safety(fixture_config(value));
  ASSERT_TRUE(safety.config().enforce);
  ASSERT_FALSE(safety.config().mapping_verified);

  const auto decision = safety.filter(urdf_snapshot(), 1.0);
  EXPECT_FALSE(decision.accept_new);
  EXPECT_EQ(decision.output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.diagnostics().reason, "mapping_required");
}

TEST(CbfSafety, ShadowPreservesBothProjectedAndInvalidRequestsExactly)
{
  auto value = urdf_fixture_json();
  value["mode"] = "shadow";
  CbfSafety safety(fixture_config(value));
  ASSERT_FALSE(safety.config().enforce);

  for (int request_case = 0; request_case < 2; ++request_case) {
    SCOPED_TRACE(request_case);
    auto snapshot = urdf_snapshot();
    snapshot.requested.arm[0] += 0.04;  // Known projected request from phase_c_tests.cpp.
    if (request_case == 1) {
      snapshot.requested.hand[0] = x2::cbf::kRigidThumbLimits[0][1] + 0.01;
    }
    const double now = 1.0 + 0.02 * request_case;
    const auto decision = safety.filter(snapshot, now);
    EXPECT_FALSE(decision.accept_new);
    expect_original_output(decision.output, snapshot.requested);
    safety.committed(decision, now);
  }
  EXPECT_EQ(safety.diagnostics().accepted, 0u);
  EXPECT_EQ(safety.diagnostics().published, 0u);
}

TEST(CbfSafety, StaleArmFeedbackHoldsTheCommittedPairButStaleThumbFeedbackDoesNot)
{
  CbfSafety safety(fixture_config());
  const auto initial = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(initial.accept_new);
  safety.committed(initial, 1.0);

  auto snapshot = urdf_snapshot();
  snapshot.arm->age_s = safety.config().feedback_max_age_s + 0.001;
  const auto held = safety.filter(snapshot, 1.02);
  expect_held_pair(held, initial.output);
  EXPECT_FALSE(held.collision_refusal);
  EXPECT_EQ(safety.diagnostics().reason, "feedback_unavailable");

  snapshot = urdf_snapshot();
  snapshot.thumb->age_s = safety.config().feedback_max_age_s + 0.001;
  const auto moved_on = safety.filter(snapshot, 1.02);
  ASSERT_TRUE(moved_on.accept_new) << safety.diagnostics().reason;
  EXPECT_EQ(moved_on.output.hand, snapshot.requested.hand);
  EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, "stale");
  EXPECT_EQ(safety.diagnostics().published, 1u);
}

TEST(CbfSafety, ArmThumbSkewWithinTheFeedbackAgeDoesNotRejectAFrame)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  snapshot.arm->age_s = 0.001;
  snapshot.thumb->age_s = 0.06;
  EXPECT_DOUBLE_EQ(safety.config().state_skew_max_s, 0.04);
  ASSERT_LT(snapshot.arm->age_s, safety.config().feedback_max_age_s);
  ASSERT_LT(snapshot.thumb->age_s, safety.config().feedback_max_age_s);
  ASSERT_GT(std::abs(snapshot.arm->age_s - snapshot.thumb->age_s), safety.config().state_skew_max_s);

  // Thumb samples are advisory; a fresh-enough one is used as is.
  const auto accepted = safety.filter(snapshot, 1.0);
  EXPECT_TRUE(accepted.accept_new) << safety.diagnostics().reason;
  EXPECT_EQ(safety.diagnostics().reason, "verified_nominal_path");
  EXPECT_TRUE(safety.diagnostics().thumb_feedback_issue.empty());
}

TEST(CbfSafety, ANewEpochCannotReuseThePreviousCommittedPair)
{
  CbfSafety safety(fixture_config());
  const auto initial = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(initial.accept_new);
  safety.committed(initial, 1.0);

  auto snapshot = urdf_snapshot();
  snapshot.epoch = 2;
  snapshot.requested.hand[0] = std::numeric_limits<double>::quiet_NaN();
  const auto rejected = safety.filter(snapshot, 1.02);
  EXPECT_FALSE(rejected.accept_new);
  EXPECT_EQ(rejected.output.kind, Output::Kind::kNone);

  snapshot = urdf_snapshot();
  snapshot.epoch = 2;
  const auto fresh = safety.filter(snapshot, 1.04);
  EXPECT_TRUE(fresh.accept_new);
  EXPECT_EQ(fresh.output.arm, snapshot.requested.arm);
  EXPECT_EQ(fresh.output.hand, snapshot.requested.hand);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);  // The new epoch is still uncommitted.
}

TEST(CbfSafety, KernelProjectionCannotBypassTheOriginalStepLimit)
{
  auto value = urdf_fixture_json();
  std::array<double, 14> velocity;
  velocity.fill(1e-6);
  value["arm_velocity_limits"] = velocity;
  CbfSafety safety(fixture_config(value));
  auto snapshot = urdf_snapshot();
  // The nominal request is much larger than the configured original allowance
  // of 1e-6 rad/s * 0.02 s = 2e-8 rad. Check final commands, not a solver status.
  snapshot.requested.arm[0] += 0.04;
  const auto first = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(first.accept_new);
  for (size_t j = 0; j < 14; ++j) {
    EXPECT_LE(std::abs(first.output.arm[j] - snapshot.arm->position[j]), 2e-8 + 1e-9);
  }
  safety.committed(first, 1.0);
  const auto next = safety.filter(snapshot, 1.02);
  ASSERT_TRUE(next.accept_new);
  for (size_t j = 0; j < 14; ++j) {
    EXPECT_LE(std::abs(next.output.arm[j] - first.output.arm[j]), 2e-8 + 1e-9);
  }
}

TEST(CbfSafety, ALateNodeResultCanOnlyUseThePreparedPairedHold)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  const auto first = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(first.accept_new);
  const auto late_first = safety.deadline_fallback(first);
  EXPECT_EQ(late_first.output.kind, Output::Kind::kNone);
  safety.committed(first, 1.0);
  snapshot.requested.arm[0] += 0.004;
  const auto next = safety.filter(snapshot, 1.02);
  ASSERT_TRUE(next.accept_new);
  const auto late = safety.deadline_fallback(next);
  expect_held_pair(late, first.output);
  safety.committed(late, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().held, 1u);
}

TEST(CbfSafety, OldCachesCannotBypassTheFilterAfterAnEpochChange)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  const auto first = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(first.accept_new);
  safety.committed(first, 1.0);
  snapshot.source_ready = false;
  snapshot.requested.arm[0] += 0.004;
  snapshot.requested.hand[2] = 0.2;
  const auto stale = safety.filter(snapshot, 1.02);
  expect_held_pair(stale, first.output);
  EXPECT_EQ(safety.diagnostics().reason, "mirror_input_stale");
  safety.committed(stale, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);

  snapshot.source_ready = true;
  const auto candidate = safety.filter(snapshot, 1.04);
  ASSERT_TRUE(candidate.accept_new);
  // A source expiring during the solve also retransmits history, never this
  // newly verified but unpublished candidate or its changed finger gesture.
  const auto late_stale = safety.feedback_fallback(candidate, "mirror_input_stale");
  expect_held_pair(late_stale, first.output);
  EXPECT_FALSE(late_stale.recovery_verified);
  EXPECT_EQ(safety.diagnostics().reason, "publication_mirror_input_stale");
  safety.committed(late_stale, 1.04);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);

  const auto resumed = safety.filter(snapshot, 1.06);
  ASSERT_TRUE(resumed.accept_new);
  EXPECT_EQ(resumed.output.hand, snapshot.requested.hand);
  EXPECT_GT(resumed.output.arm[0], first.output.arm[0]);
  safety.committed(resumed, 1.06);
  EXPECT_EQ(safety.diagnostics().accepted, 2u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
  ++snapshot.epoch;
  snapshot.source_ready = false;
  const auto pending = safety.filter(snapshot, 1.08);
  EXPECT_EQ(pending.output.kind, Output::Kind::kNone);
  EXPECT_FALSE(pending.accept_new);
  EXPECT_EQ(safety.diagnostics().reason, "awaiting_mirror_input");
}

TEST(CbfSafety, RejectsShortAndLongMappingNativeAndArmLimitArrays)
{
  for (const auto * field : {"command_scale", "command_offset", "feedback_scale", "feedback_offset",
      "native_lower", "native_upper"})
  {
    SCOPED_TRACE(field);
    for (std::size_t count : {3u, 5u}) {
      SCOPED_TRACE(count);
      auto value = historical_native_fixture_json();
      value["thumb_mapping"][field] = json::array();
      for (std::size_t j = 0; j < count; ++j) {value["thumb_mapping"][field].push_back(1.0);}
      EXPECT_THROW(fixture_config(value), std::invalid_argument);
    }
  }
  auto base = historical_native_fixture_json();
  for (std::size_t j = 0; j < 14; ++j) {
    base["arm_lower_limits"].push_back(x2::cbf::generated::kArmLimits[j][0]);
    base["arm_upper_limits"].push_back(x2::cbf::generated::kArmLimits[j][1]);
    base["arm_velocity_limits"].push_back(1.0);
  }
  for (const auto * field : {"arm_lower_limits", "arm_upper_limits", "arm_velocity_limits"}) {
    SCOPED_TRACE(field);
    for (std::size_t count : {13u, 15u}) {
      SCOPED_TRACE(count);
      auto value = base;
      value[field] = json::array();
      for (std::size_t j = 0; j < count; ++j) {value[field].push_back(1.0);}
      EXPECT_THROW(fixture_config(value), std::invalid_argument);
    }
  }
}

TEST(CbfSafety, UnverifiedNativeMappingReportsShadowAnglesAndEnforcesGeometry)
{
  auto value = historical_native_fixture_json();
  CbfSafety shadow(fixture_config(value));
  ASSERT_TRUE(shadow.config().mapping_available);
  ASSERT_TRUE(shadow.config().native_limits_available);
  ASSERT_FALSE(shadow.config().mapping_verified);
  const std::array<std::array<double, 2>, 4> expected_bounds{{
    {-0.03, 1.117}, {-1.6924, -0.0024}, {-0.606, 0.294}, {-1.6924, -0.0024}}};
  for (std::size_t axis = 0; axis < expected_bounds.size(); ++axis) {
    for (std::size_t end = 0; end < 2; ++end) {
      EXPECT_NEAR(shadow.config().thumb_model_limits[axis][end], expected_bounds[axis][end], 1e-12);
    }
  }
  value["mode"] = "enforce";
  CbfSafety enforce(fixture_config(value));
  ASSERT_FALSE(enforce.config().mapping_verified);
  constexpr std::array<std::size_t, 4> hand_slots{0, 1, 10, 11};
  for (bool open_target : {false, true}) {
    SCOPED_TRACE(open_target);
    auto snapshot = urdf_snapshot();
    ASSERT_TRUE(snapshot.thumb.has_value());
    for (std::size_t axis = 0; axis < hand_slots.size(); ++axis) {
      const double urdf = x2::cbf::kRigidThumbBaselineUrdf[x2::cbf::kRigidThumbIndices[axis]];
      const double native = (urdf - kHistoricalThumbOffset[axis]) / kHistoricalThumbScale[axis];
      ASSERT_GE(native, shadow.config().native_lower[axis]);
      ASSERT_LE(native, shadow.config().native_upper[axis]);
      snapshot.thumb->position[axis] = native;
      snapshot.requested.hand[hand_slots[axis]] = open_target ? 0.0 : native;
    }
    const auto decision = shadow.filter(snapshot, 1.0);
    expect_original_output(decision.output, snapshot.requested);
    EXPECT_FALSE(decision.accept_new);
    const auto & diagnostics = shadow.diagnostics();
    ASSERT_TRUE(diagnostics.thumb_angles_valid);
    EXPECT_EQ(diagnostics.reason.find("unverified_mapping/"), 0u);
    for (std::size_t axis = 0; axis < hand_slots.size(); ++axis) {
      const double baseline = x2::cbf::kRigidThumbBaselineUrdf[x2::cbf::kRigidThumbIndices[axis]];
      EXPECT_NEAR(diagnostics.thumb_current_urdf[axis], baseline, 1e-12);
      EXPECT_NEAR(diagnostics.thumb_target_urdf[axis], open_target ? kHistoricalThumbOffset[axis] : baseline, 1e-12);
    }
    const auto enforced = enforce.filter(snapshot, 1.0);
    const auto & reason = enforce.diagnostics().reason;
    EXPECT_EQ(reason.find("unverified_mapping/"), 0u);
    EXPECT_EQ(reason.find("mapping_unverified"), std::string::npos);
    EXPECT_EQ(reason.find("mapping_required"), std::string::npos);
    EXPECT_FALSE(enforce.config().mapping_verified);
    if (!open_target) {
      ASSERT_TRUE(enforced.accept_new);
      ASSERT_EQ(enforced.output.kind, Output::Kind::kUpperBody);
      EXPECT_EQ(enforced.output.arm, snapshot.requested.arm);
      EXPECT_EQ(enforced.output.hand, snapshot.requested.hand);
    }
    // An open target may still fail geometry; candidate availability is not a safety pass.
  }
}

TEST(CbfSafety, LeftCandidateAcceptsSafeCompactFeedbackAndClampsAnUnreachableThumbTarget)
{
  CbfSafety safety(fixture_config(historical_left_candidate_fixture_json()));
  ASSERT_TRUE(safety.config().enforce);
  ASSERT_TRUE(safety.config().mapping_available);
  ASSERT_FALSE(safety.config().mapping_verified);
  EXPECT_TRUE(safety.config().hands_installed[0]);
  EXPECT_FALSE(safety.config().hands_installed[1]);
  EXPECT_DOUBLE_EQ(safety.config().deadline_us, 100000.0);
  const auto snapshot = historical_left_native_snapshot();
  ASSERT_EQ(snapshot.thumb->position.size(), 2u);

  const auto decision = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(decision.accept_new);
  ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(decision.output.arm, snapshot.requested.arm);
  ASSERT_EQ(decision.output.hand.size(), 20u);
  EXPECT_EQ(decision.output.hand, snapshot.requested.hand);
  EXPECT_TRUE(safety.diagnostics().distance_valid);
  EXPECT_EQ(safety.diagnostics().reason.find("unverified_mapping/"), 0u);
  safety.committed(decision, 1.0);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().published, 1u);
  EXPECT_FALSE(safety.config().mapping_verified);

  // The hand cannot reach past its native range; the command is clamped there.
  auto unreachable = snapshot;
  unreachable.requested.hand[0] = safety.config().native_upper[0] + 0.01;
  const auto clamped = safety.filter(unreachable, 1.02);
  ASSERT_TRUE(clamped.accept_new) << safety.diagnostics().reason;
  EXPECT_TRUE(clamped.modified);
  EXPECT_DOUBLE_EQ(clamped.output.hand[0], safety.config().native_upper[0]);
  EXPECT_EQ(clamped.output.arm, unreachable.requested.arm);
  for (std::size_t slot = 1; slot < 20; ++slot) {
    EXPECT_EQ(clamped.output.hand[slot], unreachable.requested.hand[slot]) << slot;
  }
  safety.committed(clamped, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 2u);
  EXPECT_EQ(safety.diagnostics().held, 0u);
  EXPECT_EQ(safety.diagnostics().published, 2u);
  EXPECT_FALSE(safety.config().mapping_verified);
}

TEST(CbfSafety, LeftCandidateFollowsThroughOutOfRangeNanAndStaleThumbFeedback)
{
  // 11-17 field freeze: one thumb root read past the native range, the held
  // frame froze the thumb, and the reading never came back. Thumb feedback
  // now only feeds FK, clamped or replaced by the last published command.
  CbfSafety safety(fixture_config(historical_left_candidate_fixture_json()));
  const auto first = safety.filter(historical_left_native_snapshot(), 1.0);
  ASSERT_TRUE(first.accept_new);
  safety.committed(first, 1.0);
  const double raw = safety.config().native_lower[0] - 0.35;  // About 20 degrees out.
  for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto snapshot = historical_left_native_snapshot();
    const char * issue;
    if (invalid_case == 0) {
      snapshot.thumb->position[0] = raw;
      issue = "out_of_range";
    } else if (invalid_case == 1) {
      snapshot.thumb->position[0] = std::numeric_limits<double>::quiet_NaN();
      issue = "unavailable";
    } else {
      snapshot.thumb->age_s = safety.config().feedback_max_age_s + 0.001;
      issue = "stale";
    }
    const double now = 1.02 + 0.02 * invalid_case;
    const auto decision = safety.filter(snapshot, now);
    ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
    EXPECT_EQ(decision.output.arm, snapshot.requested.arm);
    EXPECT_EQ(decision.output.hand, snapshot.requested.hand);
    EXPECT_EQ(safety.diagnostics().reason.find("unverified_mapping/"), 0u);
    EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, issue);
    ASSERT_TRUE(safety.diagnostics().last_thumb_feedback_event.has_value());
    const auto & event = *safety.diagnostics().last_thumb_feedback_event;
    EXPECT_EQ(event.issue, issue);
    EXPECT_EQ(event.event, static_cast<uint64_t>(invalid_case + 1));
    EXPECT_DOUBLE_EQ(event.at_s, now);
    if (invalid_case == 0) {
      EXPECT_EQ(event.axis, 0);
      EXPECT_DOUBLE_EQ(event.value, raw);
      EXPECT_NEAR(event.lower, safety.config().native_lower[0], 1e-12);
      EXPECT_NEAR(event.upper, safety.config().native_upper[0], 1e-12);
    }
    safety.committed(decision, now);
  }
  EXPECT_EQ(safety.diagnostics().thumb_feedback_frames, 3u);
  EXPECT_EQ(safety.diagnostics().accepted, 4u);
  EXPECT_EQ(safety.diagnostics().published, 4u);
  EXPECT_EQ(safety.diagnostics().held, 0u);

  // A valid reading clears the issue; the event stays for the log.
  const auto recovered = safety.filter(historical_left_native_snapshot(), 1.1);
  ASSERT_TRUE(recovered.accept_new);
  EXPECT_TRUE(safety.diagnostics().thumb_feedback_issue.empty());
  EXPECT_EQ(safety.diagnostics().last_thumb_feedback_event->event, 3u);
}

TEST(CbfSafety, ExplicitLeftHandOnlyUsesTwoMeasuredRootsAndPreservesHand20)
{
  auto value = urdf_fixture_json();
  value["hands_installed"] = {true, false};
  const auto config = fixture_config(value);
  EXPECT_TRUE(config.matches_hand_types({1, 0}));
  // Left installed but reported as none: an empty flange, checked as a phantom hand.
  EXPECT_TRUE(config.matches_hand_types({0, 0}));
  EXPECT_FALSE(config.matches_hand_types({1, 1}));
  EXPECT_FALSE(config.matches_hand_types({1, 2}));
  EXPECT_FALSE(config.matches_hand_types({1}));
  CbfSafety safety(config);
  auto snapshot = urdf_snapshot();
  snapshot.thumb->position.resize(2);
  // Native command slots still exist for an absent right hand. They must be
  // preserved as opaque message values and cannot drive a phantom thumb.
  snapshot.requested.hand[10] = 12.0;
  snapshot.requested.hand[11] = -12.0;
  const auto decision = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(decision.accept_new);
  EXPECT_EQ(decision.output.arm, snapshot.requested.arm);
  EXPECT_EQ(decision.output.hand, snapshot.requested.hand);
  safety.committed(decision, 1.0);
  snapshot.thumb->position.pop_back();
  const auto lost_left = safety.filter(snapshot, 1.02);
  ASSERT_TRUE(lost_left.accept_new) << safety.diagnostics().reason;
  EXPECT_EQ(lost_left.output.hand, snapshot.requested.hand);
  EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, "unavailable");
}

TEST(CbfSafety, AnInstalledHandCannotSilentlyDisappearOrChangeHardware)
{
  // Missing thumb roots no longer refuse a frame; the hardware check below does.
  CbfSafety both(fixture_config());
  auto snapshot = urdf_snapshot();
  snapshot.thumb->position.resize(2);
  const auto missing = both.filter(snapshot, 1.0);
  ASSERT_TRUE(missing.accept_new) << both.diagnostics().reason;
  EXPECT_EQ(both.diagnostics().thumb_feedback_issue, "unavailable");

  auto value = urdf_fixture_json();
  value["hands_installed"] = {true, false};
  CbfSafety left_only(fixture_config(value));
  const auto accepted = left_only.filter(snapshot, 1.0);
  ASSERT_TRUE(accepted.accept_new);
  left_only.committed(accepted, 1.0);
  snapshot.hand_configuration_matches = false;
  const auto mismatch = left_only.filter(snapshot, 1.02);
  EXPECT_EQ(mismatch.output.kind, Output::Kind::kNone);
  EXPECT_FALSE(mismatch.accept_new);
  EXPECT_EQ(left_only.diagnostics().reason, "hand_configuration_mismatch");
}

TEST(CbfSafety, InstalledHandConfigurationRequiresExplicitBooleans)
{
  for (const auto & invalid : {json{true}, json{true, false, true}, json{1, 0}, json{false, false}}) {
    auto value = urdf_fixture_json();
    value["hands_installed"] = invalid;
    EXPECT_THROW(fixture_config(value), std::invalid_argument);
  }
  // Disabled ancillary fields remain inert, including invalid new configuration.
  EXPECT_FALSE(CbfSafetyConfig::from_json(
      R"({"enabled":false,"hands_installed":"unused"})", 50.0).enabled);
}

TEST(CbfSafety, PublicationFeedbackGateChecksTheArmOnly)
{
  auto value = urdf_fixture_json();
  value["hands_installed"] = {true, false};
  const auto config = fixture_config(value);
  auto latest = urdf_snapshot();
  latest.thumb->position.resize(2);
  ASSERT_TRUE(config.feedback_error(latest).empty());
  latest.arm->position[0] += 0.001;
  EXPECT_TRUE(config.feedback_error(latest).empty());
  latest.thumb->position[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(config.feedback_error(latest).empty());
  latest.thumb.reset();
  EXPECT_TRUE(config.feedback_error(latest).empty());
  latest.arm->position[3] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(config.feedback_error(latest), "feedback_unavailable");
  latest.arm->position[3] = 0.0;
  latest.arm->age_s = config.feedback_max_age_s + 0.001;
  EXPECT_EQ(config.feedback_error(latest), "feedback_unavailable");
  latest.arm.reset();
  EXPECT_EQ(config.feedback_error(latest), "feedback_unavailable");
}

TEST(CbfSafety, InputFaultKeepsTheFirstFailingFrameAndDoesNotChangeRejectionBehavior)
{
  CbfSafety safety(fixture_config());
  const auto valid = urdf_snapshot();
  const auto seeded = safety.filter(valid, 1.0);
  ASSERT_TRUE(seeded.accept_new);
  safety.committed(seeded, 1.0);
  EXPECT_EQ(safety.diagnostics().invalid_input_frames, 0u);
  EXPECT_FALSE(safety.diagnostics().last_input_fault.has_value());

  auto bad = valid;
  constexpr std::size_t joint = 5;
  const double upper = x2::cbf::generated::kArmLimits[joint][1];
  bad.arm->position[joint] = upper + 0.0001;
  bad.requested.sequence = 23;
  const auto rejected = safety.filter(bad, 1.02);
  expect_held_pair(rejected, seeded.output);
  ASSERT_TRUE(safety.diagnostics().last_input_fault.has_value());
  const auto first = *safety.diagnostics().last_input_fault;
  EXPECT_EQ(first.event, 1u);
  EXPECT_EQ(first.call, 2u);
  EXPECT_EQ(first.command_sequence, 23u);
  EXPECT_EQ(first.stage, "nominal");  // Old-target verification is now lazy, after projection only.
  EXPECT_EQ(first.issue, "current_arm_out_of_range");
  EXPECT_EQ(first.index, static_cast<int>(joint));
  EXPECT_DOUBLE_EQ(first.value, bad.arm->position[joint]);
  EXPECT_DOUBLE_EQ(first.lower, x2::cbf::generated::kArmLimits[joint][0]);
  EXPECT_DOUBLE_EQ(first.upper, upper);
  EXPECT_DOUBLE_EQ(first.arm_current[joint], bad.arm->position[joint]);
  EXPECT_EQ(first.arm_requested, bad.requested.arm);

  bad.arm->position[joint] = upper + 0.0002;
  const auto repeated = safety.filter(bad, 1.04);
  expect_held_pair(repeated, seeded.output);
  EXPECT_EQ(safety.diagnostics().invalid_input_frames, 2u);
  ASSERT_TRUE(safety.diagnostics().last_input_fault.has_value());
  EXPECT_EQ(safety.diagnostics().last_input_fault->event, first.event);
  EXPECT_DOUBLE_EQ(safety.diagnostics().last_input_fault->value, first.value);

  const auto recovered = safety.filter(valid, 1.06);
  ASSERT_TRUE(recovered.accept_new);
  EXPECT_EQ(recovered.output.arm, valid.requested.arm);
  EXPECT_EQ(recovered.output.hand, valid.requested.hand);
  safety.committed(recovered, 1.06);
  EXPECT_EQ(safety.diagnostics().last_input_fault->event, first.event);
  safety.filter(bad, 1.08);
  ASSERT_TRUE(safety.diagnostics().last_input_fault.has_value());
  EXPECT_EQ(safety.diagnostics().last_input_fault->event, first.event + 1);
  EXPECT_DOUBLE_EQ(safety.diagnostics().last_input_fault->value, bad.arm->position[joint]);
}

TEST(CbfSafety, RecordedShoulderFeedbackUsesVendorBoundsWithoutRelaxingPublishedCommands)
{
  // Real CBF_INPUT_FAULT at 1794081437.808596544, stage=hold, from
  // artifacts/mirror_integration_20261108/live-inspect-20261108-035745/cbf-input-fault.log.
  // The capture contains only four hand roots; the other 16 command slots retain
  // the existing zero fixture, not invented full-hand measurements.
  const auto fault = json::parse(R"json({
    "arm_current14": [
      0.2698812484741211, -0.027132034301757813, 0.041512489318847656,
      -0.9933342933654785, -0.0018215179443359375, 0.07877445220947266,
      0.061608314514160156, 0.2936573028564453, 0.052634239196777344,
      -0.28905534744262695, -1.0819206237792969, 0.009299278259277344,
      0.09212589263916016, -0.12264442443847656
    ],
    "arm_requested14": [
      0.23047414038024538, 0.02165195969586532, 0.06343637224227436,
      -1.0078913844086494, -0.01558182072825891, 0.09223377305599367,
      0.09912697780938515, 0.2779952384733474, 0.03490658503988659,
      -0.36483514450089555, -1.1070037886454627, 0.014695470314202338,
      0.11045987550165319, -0.15082766020717253
    ],
    "arm_target14": [
      0.23605849349154875, 0.020327284868150518, 0.06284284189253467,
      -1.0137120862253512, -0.013004390778715588, 0.08979877085391544,
      0.09377184852928588, 0.28129737927273946, 0.03445081610300152,
      -0.35012300453062606, -1.1090341908946615, 0.014577933023973443,
      0.10682537988496096, -0.14390409509995367
    ],
    "thumb_native_feedback_compact": [-0.06105468749999998, 0.645703125],
    "thumb_native_target_slots_0_1_10_11": [
      -0.016761575722854145, 0.7486839114778014, 0.28476566561321354, -0.5941380272128859
    ],
    "arm_age_s": 0.002676814,
    "thumb_age_s": 0.003344156
  })json");
  auto seed = historical_left_native_snapshot();
  seed.epoch = 7;
  seed.requested.sequence = 204;
  seed.requested.arm = fault.at("arm_target14").get<std::vector<double>>();
  seed.arm->position = seed.requested.arm;
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  const auto roots = fault.at("thumb_native_target_slots_0_1_10_11").get<std::array<double, 4>>();
  for (std::size_t axis = 0; axis < slots.size(); ++axis) {seed.requested.hand[slots[axis]] = roots[axis];}
  seed.thumb->position = {roots[0], roots[1]};
  seed.arm->age_s = seed.thumb->age_s = 0.0;

  auto observed = seed;
  observed.requested.arm = fault.at("arm_requested14").get<std::vector<double>>();
  observed.arm->position = fault.at("arm_current14").get<std::vector<double>>();
  observed.thumb->position = fault.at("thumb_native_feedback_compact").get<std::vector<double>>();
  observed.arm->age_s = fault.at("arm_age_s").get<double>();
  observed.thumb->age_s = fault.at("thumb_age_s").get<double>();
  // Keep exactly the seeded hand target, isolating the native feedback domain.
  ASSERT_EQ(observed.thumb->position.size(), 2u);
  constexpr std::size_t shoulder_roll = 8;
  const double upper = x2::cbf::generated::kArmLimits[shoulder_roll][1];
  ASSERT_GT(observed.arm->position[shoulder_roll], upper);

  auto value = historical_left_candidate_fixture_json();
  // Existing mirror command bands from aimdk_x2_magicmirror.yaml
  // motion_imitation_stream.mock_playback.arm_limits, in arm14 profile order.
  constexpr std::array<double, 14> command_lower{
    -3.08, -0.061, -2.556, -2.3556, -2.556, -0.558, -1.571,
    -3.08, -2.993, -2.556, -2.3556, -2.556, -0.558, -0.724};
  constexpr std::array<double, 14> command_upper{
    2.04, 2.993, 2.556, 0.0, 2.556, 0.558, 0.724,
    2.04, 0.061, 2.556, 0.0, 2.556, 0.558, 1.571};
  value["arm_lower_limits"] = command_lower;
  value["arm_upper_limits"] = command_upper;
  CbfSafety strict(fixture_config(value));
  const auto strict_seed = strict.filter(seed, 1.0);
  ASSERT_TRUE(strict_seed.accept_new) << strict.diagnostics().reason;
  strict.committed(strict_seed, 1.0);
  const auto rejected = strict.filter(observed, 1.02);
  expect_held_pair(rejected, strict_seed.output);
  EXPECT_EQ(strict.diagnostics().reason, "unverified_mapping/INVALID_INPUT_FALLBACK");
  ASSERT_TRUE(strict.diagnostics().last_input_fault.has_value());
  const auto & strict_fault = *strict.diagnostics().last_input_fault;
  EXPECT_EQ(strict_fault.stage, "nominal");
  EXPECT_EQ(strict_fault.issue, "current_arm_out_of_range");
  EXPECT_EQ(strict_fault.index, static_cast<int>(shoulder_roll));
  EXPECT_DOUBLE_EQ(strict_fault.value, observed.arm->position[shoulder_roll]);
  EXPECT_DOUBLE_EQ(strict_fault.upper, upper);

  // x2_ultra_omnihand.yaml vendor.arm.limits in that profile's arm_joints order.
  // These are native observation bounds, not an expansion of command bounds.
  constexpr std::array<double, 14> feedback_lower{
    -3.08, -0.061, -2.556, -2.3556, -2.556, -0.5236, -1.5097,
    -3.08, -3.0456, -2.556, -2.3556, -2.556, -0.5236, -0.724};
  constexpr std::array<double, 14> feedback_upper{
    2.04, 3.0456, 2.556, 0.0, 2.556, 0.5236, 0.724,
    2.04, 0.061, 2.556, 0.0, 2.556, 0.5236, 1.5097};
  for (std::size_t joint = 0; joint < feedback_lower.size(); ++joint) {
    ASSERT_GE(observed.arm->position[joint], feedback_lower[joint]);
    ASSERT_LE(observed.arm->position[joint], feedback_upper[joint]);
  }
  value["arm_feedback_lower_limits"] = feedback_lower;
  value["arm_feedback_upper_limits"] = feedback_upper;
  CbfSafety native_feedback(fixture_config(value));
  EXPECT_DOUBLE_EQ(native_feedback.config().deadline_us, 100000.0);
  EXPECT_FALSE(native_feedback.config().mapping_verified);
  EXPECT_EQ(native_feedback.config().lower, strict.config().lower);
  EXPECT_EQ(native_feedback.config().upper, strict.config().upper);
  EXPECT_EQ(native_feedback.config().velocity, strict.config().velocity);
  const auto published_seed = native_feedback.filter(seed, 1.0);
  ASSERT_TRUE(published_seed.accept_new) << native_feedback.diagnostics().reason;
  native_feedback.committed(published_seed, 1.0);
  const auto followed = native_feedback.filter(observed, 1.02);
  EXPECT_EQ(native_feedback.diagnostics().reason.find("INVALID_INPUT_FALLBACK"), std::string::npos);
  EXPECT_EQ(native_feedback.diagnostics().invalid_input_frames, 0u);
  ASSERT_EQ(followed.output.kind, Output::Kind::kUpperBody) << native_feedback.diagnostics().reason;
  ASSERT_TRUE(followed.accept_new) << native_feedback.diagnostics().reason;
  ASSERT_EQ(followed.output.arm.size(), 14u);
  EXPECT_EQ(followed.output.hand, seed.requested.hand);
  double maximum_step = 0.0;
  for (std::size_t joint = 0; joint < followed.output.arm.size(); ++joint) {
    SCOPED_TRACE(joint);
    const double position = followed.output.arm[joint];
    const double step = std::abs(position - published_seed.output.arm[joint]);
    maximum_step = std::max(maximum_step, step);
    EXPECT_TRUE(std::isfinite(position));
    EXPECT_GE(position, native_feedback.config().lower[joint] - 1e-10);
    EXPECT_LE(position, native_feedback.config().upper[joint] + 1e-10);
    EXPECT_LE(step, std::min(0.04, native_feedback.config().velocity[joint] * native_feedback.config().period_s) + 1e-9);
  }
  EXPECT_GT(maximum_step, 1e-6) << native_feedback.diagnostics().reason;
  EXPECT_DOUBLE_EQ(observed.arm->position[shoulder_roll], 0.052634239196777344);
}

TEST(CbfSafety, FeedbackTolerancePreservesAllCommandBoundsAndVelocity)
{
  auto value = historical_left_candidate_fixture_json();
  const auto strict = fixture_config(value);
  EXPECT_DOUBLE_EQ(strict.feedback_limit_tolerance_rad, 0.0);
  constexpr double one_degree = 0.017453292519943295;
  value["feedback_limit_tolerance_rad"] = one_degree;
  const auto observation_allowance = fixture_config(value);
  EXPECT_DOUBLE_EQ(observation_allowance.feedback_limit_tolerance_rad, one_degree);
  for (std::size_t joint = 0; joint < strict.lower.size(); ++joint) {
    SCOPED_TRACE(joint);
    EXPECT_DOUBLE_EQ(observation_allowance.lower[joint], strict.lower[joint]);
    EXPECT_DOUBLE_EQ(observation_allowance.upper[joint], strict.upper[joint]);
    EXPECT_DOUBLE_EQ(observation_allowance.velocity[joint], strict.velocity[joint]);
  }
  EXPECT_EQ(observation_allowance.arm_feedback_limits, strict.arm_feedback_limits);
  EXPECT_EQ(observation_allowance.native_lower, strict.native_lower);
  EXPECT_EQ(observation_allowance.native_upper, strict.native_upper);
  EXPECT_EQ(observation_allowance.thumb_model_limits, strict.thumb_model_limits);
  EXPECT_FALSE(observation_allowance.mapping_verified);
}

TEST(CbfSafety, ThumbFeedbackBeyondTheModelRangeIsClampedForGeometryWithoutRelaxingTargets)
{
  auto value = historical_left_candidate_fixture_json();
  CbfSafety safety(fixture_config(value));
  const auto & config = safety.config();
  const double native_upper = config.native_upper[0];
  const auto & band = config.thumb_model_limits[0];
  // native_upper maps to the model lower bound (scale -1). Readings a hair and
  // about 20 degrees past it both drive FK from the bound itself.
  for (double excess : {0.0006, 0.35}) {
    SCOPED_TRACE(excess);
    auto beyond = historical_left_native_snapshot();
    ASSERT_EQ(beyond.thumb->position.size(), 2u);
    beyond.thumb->position[0] = native_upper + excess;
    EXPECT_TRUE(config.feedback_error(beyond).empty());
    const auto decision = safety.filter(beyond, 1.0);
    EXPECT_EQ(safety.diagnostics().invalid_input_frames, 0u);
    EXPECT_EQ(safety.diagnostics().reason.find("INVALID_INPUT_FALLBACK"), std::string::npos);
    EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, "out_of_range");
    ASSERT_TRUE(safety.diagnostics().thumb_angles_valid);
    const double raw_model_current = config.feedback_scale[0] * beyond.thumb->position[0] + config.feedback_offset[0];
    ASSERT_LT(raw_model_current, band[0]);
    EXPECT_DOUBLE_EQ(safety.diagnostics().thumb_current_urdf[0], band[0]);
    EXPECT_TRUE(decision.accept_new) << safety.diagnostics().reason;
  }

  // Target bounds stay strict: a command past native_upper is clamped onto it
  // (into the model range), never handed to the kernel beyond it.
  auto unreachable = historical_left_native_snapshot();
  unreachable.requested.hand[0] = native_upper + 0.35;
  const auto clamped = safety.filter(unreachable, 1.04);
  ASSERT_TRUE(clamped.accept_new) << safety.diagnostics().reason;
  EXPECT_DOUBLE_EQ(clamped.output.hand[0], native_upper);
  EXPECT_GE(safety.diagnostics().thumb_target_urdf[0], band[0] - 1e-12);
  EXPECT_LE(safety.diagnostics().thumb_target_urdf[0], band[1] + 1e-12);
  EXPECT_FALSE(config.mapping_verified);
}

namespace
{

json freeze091650_config()
{
  auto value = historical_left_candidate_fixture_json();
  value["check_arm_feedback_goal_range"] = false;
  value["feedback_limit_tolerance_rad"] = 0.017453292519943295;  // Captured configuration; thumb remains unchanged.
  value["arm_lower_limits"] = {
    -3.08, -0.061, -2.556, -2.3556, -2.556, -0.558, -1.571,
    -3.08, -2.993, -2.556, -2.3556, -2.556, -0.558, -0.724};
  value["arm_upper_limits"] = {
    2.04, 2.993, 2.556, 0.0, 2.556, 0.558, 0.724,
    2.04, 0.061, 2.556, 0.0, 2.556, 0.558, 1.571};
  value["arm_feedback_lower_limits"] = {
    -3.08, -0.061, -2.556, -2.3556, -2.556, -0.5236, -1.5097,
    -3.08, -3.0456, -2.556, -2.3556, -2.556, -0.5236, -0.724};
  value["arm_feedback_upper_limits"] = {
    2.04, 3.0456, 2.556, 0.0, 2.556, 0.5236, 0.724,
    2.04, 0.061, 2.556, 0.0, 2.556, 0.5236, 1.5097};
  return value;
}

std::array<CbfSnapshot, 2> freeze091650_snapshots()
{
  // Exact fault at 1794100419.631301570 from live-inspect-20261108-091650/cbf-input-fault.log.
  // Only four hand roots were captured; other command slots keep the existing zero fixture.
  const auto recorded = json::parse(R"json({
    "arm_current14": [
      -0.28867197036743164, 0.05896139144897461, 0.7967953681945801,
      -2.002871036529541, -0.04994964599609375, -0.057412147521972656,
      0.5601968765258789, -0.05685234069824219, 0.07851982116699219,
      -0.9423298835754395, -2.168729782104492, 0.09941959381103516,
      -0.02918243408203125, -0.5750741958618164
    ],
    "arm_requested14": [
      -0.3568861829656448, 0.12048106447920368, 0.7168539760469147,
      -2.1038023323131334, -0.08758554631111054, -0.11925771910318672,
      0.5636436553304075, -0.08219935181512805, 0.03490658503988659,
      -0.879255917212644, -2.265567746522001, 0.1257394162114828,
      -0.0766667316915563, -0.5861225637455074
    ],
    "arm_target14": [
      -0.3552903042402299, 0.1268837210026983, 0.740317543715007,
      -2.0910783981303167, -0.0742829679272163, -0.09652702761162991,
      0.5665138272996783, -0.08346404967531795, 0.03490658503988659,
      -0.9005852873250887, -2.252534808740709, 0.11314159080837327,
      -0.05935505643040753, -0.583169423221417
    ],
    "thumb_native_feedback_compact": [-0.009003906249999916, 0.48427734375],
    "thumb_native_target_slots_0_1_10_11": [
      -0.011974565134550776, 0.5610939091618077, 0.010129200875939227, -0.11865635311814521
    ],
    "arm_age_s": 0.0032358200000000004,
    "thumb_age_s": 0.003603466
  })json");
  auto observed = historical_left_native_snapshot();
  observed.epoch = 7;
  observed.requested.sequence = 529;
  observed.requested.arm = recorded.at("arm_requested14").get<std::vector<double>>();
  observed.arm->position = recorded.at("arm_current14").get<std::vector<double>>();
  observed.thumb->position = recorded.at("thumb_native_feedback_compact").get<std::vector<double>>();
  observed.arm->age_s = recorded.at("arm_age_s").get<double>();
  observed.thumb->age_s = recorded.at("thumb_age_s").get<double>();
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  const auto roots = recorded.at("thumb_native_target_slots_0_1_10_11").get<std::array<double, 4>>();
  for (std::size_t axis = 0; axis < slots.size(); ++axis) {observed.requested.hand[slots[axis]] = roots[axis];}
  auto seed = observed;
  seed.requested.arm = recorded.at("arm_target14").get<std::vector<double>>();
  seed.arm->position = seed.requested.arm;
  seed.thumb->position = {roots[0], roots[1]};
  seed.arm->age_s = seed.thumb->age_s = 0.0;
  return {seed, observed};
}

double expect_freeze091650_command(const Output & output, const Output & previous, const CbfSafetyConfig & config)
{
  EXPECT_EQ(output.hand, previous.hand);
  double maximum_step = 0.0;
  for (std::size_t joint = 0; joint < 14; ++joint) {
    SCOPED_TRACE(joint);
    const double step = std::abs(output.arm[joint] - previous.arm[joint]);
    maximum_step = std::max(maximum_step, step);
    EXPECT_TRUE(std::isfinite(output.arm[joint]));
    EXPECT_GE(output.arm[joint], config.lower[joint] - 1e-10);
    EXPECT_LE(output.arm[joint], config.upper[joint] + 1e-10);
    EXPECT_LE(step, std::min(0.04, config.velocity[joint] * config.period_s) + 1e-9);
  }
  return maximum_step;
}

x2::cbf::SafetyFilterInput freeze091650_input(const CbfSnapshot & snapshot, const CbfSafetyConfig & config)
{
  x2::cbf::SafetyFilterInput input;
  std::copy_n(snapshot.arm->position.begin(), 14, input.q_current.begin());
  std::copy_n(snapshot.requested.arm.begin(), 14, input.q_nominal.begin());
  input.hands_installed = config.hands_installed;
  input.check_current_arm_goal_range = false;
  input.current_arm_joint_limits = config.arm_feedback_limits;
  for (std::size_t joint = 0; joint < 14; ++joint) {
    input.command_joint_limits[joint] = {config.lower[joint], config.upper[joint]};
  }
  for (auto & band : input.current_arm_joint_limits) {
    band[0] -= config.feedback_limit_tolerance_rad;
    band[1] += config.feedback_limit_tolerance_rad;
  }
  input.thumb_current_valid = true;
  input.thumb_joint_limits_urdf = config.thumb_model_limits;
  input.thumb_joint_limits_urdf[2] = x2::cbf::kRigidThumbLimits[2];
  input.thumb_joint_limits_urdf[3] = x2::cbf::kRigidThumbLimits[3];
  // Production clamps the reading into the model range; no thumb allowance.
  input.current_thumb_joint_limits_urdf = input.thumb_joint_limits_urdf;
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const auto slot = x2::cbf::kRigidThumbIndices[axis];
    input.q_thumb_current_urdf[slot] = std::clamp(
      config.feedback_scale[axis] * snapshot.thumb->position[axis] + config.feedback_offset[axis],
      config.thumb_model_limits[axis][0], config.thumb_model_limits[axis][1]);
    input.q_thumb_urdf[slot] = config.command_scale[axis] * snapshot.requested.hand[axis] + config.command_offset[axis];
  }
  return input;
}

x2::cbf::SafetyFilterResult freeze091650_path(
  const CbfSnapshot & snapshot, const CbfSafetyConfig & config, bool allow_recovery = false)
{
  const auto input = freeze091650_input(snapshot, config);
  auto numerical = x2::cbf::SafetyFilterConfig{};
  numerical.deadline_us = 100000.0;
  numerical.path_check_substeps = 64;
  numerical.allow_airwall_recovery = allow_recovery;
  if (allow_recovery) {numerical.max_step_rad = 6.5;}
  return x2::cbf::verify_arm_target(input, numerical);
}

}  // namespace

TEST(CbfSafety, Recorded091650FrameAcceptsRawFeedbackBeyondGoalRangeAndFollows)
{
  const auto config = fixture_config(freeze091650_config());
  ASSERT_FALSE(config.check_arm_feedback_goal_range);
  const auto frames = freeze091650_snapshots();
  CbfSafety safety(config);
  const auto seeded = safety.filter(frames[0], 1.0);
  ASSERT_TRUE(seeded.accept_new) << safety.diagnostics().reason;
  safety.committed(seeded, 1.0);
  const auto followed = safety.filter(frames[1], 1.02);
  ASSERT_EQ(followed.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
  ASSERT_TRUE(followed.accept_new) << safety.diagnostics().reason;
  ASSERT_EQ(followed.output.arm.size(), 14u);
  EXPECT_EQ(safety.diagnostics().invalid_input_frames, 0u);
  EXPECT_GT(expect_freeze091650_command(followed.output, seeded.output, config), 1e-6);
  EXPECT_DOUBLE_EQ(frames[1].arm->position[8], 0.07851982116699219);
}

TEST(CbfSafety, ShoulderFeedbackCrossesFormerBoundariesWithoutInvalidOrLatchedRefusal)
{
  const auto config = fixture_config(freeze091650_config());
  const auto frames = freeze091650_snapshots();
  CbfSafety safety(config);
  const auto seeded = safety.filter(frames[0], 1.0);
  ASSERT_TRUE(seeded.accept_new) << safety.diagnostics().reason;
  safety.committed(seeded, 1.0);
  constexpr std::array<double, 7> samples{0.061, 0.078453292519943295, 0.15, 0.25, 0.15, 0.078453292519943295, 0.061};
  for (std::size_t index = 0; index < samples.size(); ++index) {
    SCOPED_TRACE(samples[index]);
    auto observed = frames[1];
    observed.arm->position[8] = samples[index];
    observed.requested = seeded.output;  // An unchanged, legal target isolates the observation domain.
    const auto geometry = freeze091650_path(observed, config);
    ASSERT_NE(geometry.status, x2::cbf::FilterStatus::InvalidInputFallback);
    ASSERT_FALSE(geometry.deadline_missed);
    const double now = 1.02 + config.period_s * index;
    const auto decision = safety.filter(observed, now);
    ASSERT_EQ(safety.diagnostics().invalid_input_frames, 0u) << safety.diagnostics().reason;
    EXPECT_EQ(safety.diagnostics().reason.find("INVALID_INPUT_FALLBACK"), std::string::npos);
    if (geometry.nonlinear_verified && geometry.thumb_target_accepted) {
      ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
      ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
      ASSERT_EQ(decision.output.arm.size(), 14u);
      EXPECT_EQ(decision.output.arm, seeded.output.arm);
      expect_freeze091650_command(decision.output, seeded.output, config);
      safety.committed(decision, now);
    }
    // A geometry/path-unsafe trial is not published in this input-domain test.
    // The last published target therefore remains the seeded complete pair.
  }
  // Return to the known-safe observed/target pair in the SAME epoch. Unsafe
  // geometry trials above must not create an input-rejection latch.
  const auto recovered = safety.filter(frames[0], 1.02 + config.period_s * samples.size());
  ASSERT_TRUE(recovered.accept_new) << safety.diagnostics().reason;
  ASSERT_EQ(recovered.output.kind, Output::Kind::kUpperBody);
  ASSERT_EQ(recovered.output.arm.size(), 14u);
  EXPECT_EQ(recovered.output.arm, seeded.output.arm);
  expect_freeze091650_command(recovered.output, seeded.output, config);
  EXPECT_EQ(safety.diagnostics().invalid_input_frames, 0u);
}

namespace
{

CbfSnapshot freeze094711_snapshot(const CbfSafetyConfig & config)
{
  // Same passive arm/thumb snapshot as incident-094711.json. These are raw
  // observations, not a reconstructed historical published command.
  auto snapshot = historical_left_native_snapshot();
  snapshot.arm->position = {
    -0.6970882415771484, 0.5183825492858887, -1.0642800331115723,
    -1.79597806930542, -0.16691350936889648, -0.16613292694091797,
    0.23441696166992188, 0.1872386932373047, 0.06969928741455078,
    -0.3392925262451172, -2.0849380493164063, 0.1093902587890625,
    0.08754825592041016, -0.5922403335571289};
  constexpr std::array<double, 2> model_current{0.03490234375000001, -1.0238843750000002};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const double native = (model_current[axis] - config.feedback_offset[axis]) / config.feedback_scale[axis];
    snapshot.thumb->position[axis] = native;
    snapshot.requested.hand[axis] = native;
  }
  snapshot.requested.arm = snapshot.arm->position;
  for (std::size_t joint = 0; joint < 14; ++joint) {
    snapshot.requested.arm[joint] = std::clamp(snapshot.requested.arm[joint], config.lower[joint], config.upper[joint]);
  }
  return snapshot;
}

CbfSnapshot airwall_tracking_lag_seed(const CbfSnapshot & frozen)
{
  auto seed = frozen;
  // Synthetic safe command: +0.04 rad left elbow was independently checked
  // in incident-094711.json. It is NOT the log's missing historical target.
  seed.requested.arm[3] += 0.04;
  seed.arm->position = seed.requested.arm;
  return seed;
}

}  // namespace

TEST(CbfSafety, RecordedAirwallTrackingLagCanMakePartialWithdrawalsAndLeaveTheWall)
{
  const auto config = fixture_config(freeze091650_config());
  auto observed = freeze094711_snapshot(config);
  const double frozen_elbow = observed.arm->position[3];
  CbfSafety safety(config);
  const auto seed = safety.filter(airwall_tracking_lag_seed(observed), 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  ASSERT_FALSE(seed.recovery_verified);  // The synthetic seed itself is outside the airwall.
  safety.committed(seed, 1.0);
  auto previous = seed.output;
  int recovery_frames = 0;
  double final_margin = -1.0;
  for (int frame = 0; frame < 12; ++frame) {
    SCOPED_TRACE(frame);
    observed.requested = previous;
    observed.requested.arm[3] = frozen_elbow + 0.005 * (frame + 1);
    const double now = 1.02 + config.period_s * frame;
    const auto decision = safety.filter(observed, now);
    ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
    ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
    EXPECT_GT(decision.output.arm[3] - observed.arm->position[3], 1e-6);
    EXPECT_GT(expect_freeze091650_command(decision.output, previous, config), 1e-6);
    auto path = observed;
    path.requested = decision.output;
    const auto verified = freeze091650_path(path, config, true);
    ASSERT_TRUE(verified.thumb_target_accepted);
    ASSERT_TRUE(verified.nonlinear_verified);
    EXPECT_EQ(decision.recovery_verified, verified.recovery_verified);
    if (frame == 0) {
      EXPECT_TRUE(decision.recovery_verified);
      EXPECT_LT(verified.output_min_safety_margin_m, 0.0);
      EXPECT_GT(verified.output_min_safety_margin_m, verified.input_min_safety_margin_m);
    }
    recovery_frames += decision.recovery_verified;
    final_margin = verified.output_min_safety_margin_m;
    safety.committed(decision, now);
    previous = decision.output;
    observed.arm->position = previous.arm;  // Offline model follows only the published pair.
    observed.thumb->position = {previous.hand[0], previous.hand[1]};
  }
  EXPECT_GT(recovery_frames, 1);
  EXPECT_LT(recovery_frames, 12);
  EXPECT_GT(final_margin, 0.0);
  EXPECT_GT(previous.arm[3] - frozen_elbow, 0.05);
  EXPECT_EQ(safety.diagnostics().published, 13u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
}

TEST(CbfSafety, AirwallFeedbackFailuresAndNodeDeadlinesKeepPublishingThenResumeInTheSameEpoch)
{
  const auto config = fixture_config(freeze091650_config());
  auto frozen = freeze094711_snapshot(config);
  frozen.requested.head = {0.12, -0.08};
  CbfSafety safety(config);
  const auto seed = safety.filter(airwall_tracking_lag_seed(frozen), 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  safety.committed(seed, 1.0);
  CbfDecision last_hold;
  for (int frame = 0; frame < 8; ++frame) {
    SCOPED_TRACE(frame);
    auto bad = frozen;
    bad.requested.arm[3] += 0.01;
    bad.requested.hand[2] += 0.25;  // Never mix a new finger gesture into a stale hold.
    if (frame % 4 == 0) {
      bad.arm->age_s = config.feedback_max_age_s + 0.01;
    } else if (frame % 4 == 1) {
      bad.thumb.reset();
      bad.requested.kind = Output::Kind::kNone;  // Core may also lack a feedback-based hand hold.
      bad.requested.head.clear();
    } else if (frame % 4 == 2) {
      bad.arm->position[0] = std::numeric_limits<double>::quiet_NaN();
    } else {
      bad.arm.reset();  // Thumb feedback no longer gates a frame.
    }
    const double now = 1.02 + config.period_s * frame;
    const auto decision = safety.filter(bad, now);
    expect_held_pair(decision, seed.output);
    EXPECT_EQ(decision.output.head, seed.output.head);
    EXPECT_EQ(decision.prepared_hold.arm, seed.output.arm);
    EXPECT_EQ(decision.prepared_hold.hand, seed.output.hand);
    EXPECT_EQ(decision.prepared_hold.head, seed.output.head);
    last_hold = frame % 2 ? safety.deadline_fallback(decision) :
      safety.feedback_fallback(decision, "feedback_unavailable");
    expect_held_pair(last_hold, seed.output);
    EXPECT_EQ(last_hold.output.head, seed.output.head);
    EXPECT_FALSE(last_hold.recovery_verified);
    EXPECT_FALSE(safety.diagnostics().distance_valid);
    if (frame % 2 == 0) {
      EXPECT_NE(safety.diagnostics().reason.find("publication_feedback_unavailable"), std::string::npos);
      EXPECT_EQ(safety.diagnostics().reason.find("node_deadline"), std::string::npos);
    }
    safety.committed(last_hold, now);
  }
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().held, 8u);
  EXPECT_EQ(safety.diagnostics().published, 9u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
  auto fresh = frozen;
  fresh.requested.arm[3] += 0.005;
  const auto resumed = safety.filter(fresh, 1.20);
  ASSERT_TRUE(resumed.accept_new) << safety.diagnostics().reason;
  EXPECT_TRUE(resumed.recovery_verified);
  EXPECT_GT(resumed.output.arm[3] - fresh.arm->position[3], 1e-6);
  expect_freeze091650_command(resumed.output, seed.output, config);
  safety.committed(resumed, 1.20);

  safety.inactive(fresh.epoch);  // An explicit ownership exit invalidates even a prepared hold.
  EXPECT_EQ(safety.deadline_fallback(last_hold).output.kind, Output::Kind::kNone);
  ++fresh.epoch;
  fresh.arm.reset();
  const auto new_epoch = safety.filter(fresh, 1.22);
  EXPECT_EQ(new_epoch.output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.feedback_fallback(last_hold, "feedback_unavailable").output.kind, Output::Kind::kNone);
}

TEST(CbfSafety, InwardAirwallRequestsNeverStopTheStreamOrAuthorizeAWorseningPath)
{
  const auto config = fixture_config(freeze091650_config());
  auto observed = freeze094711_snapshot(config);
  CbfSafety safety(config);
  const auto seed = safety.filter(airwall_tracking_lag_seed(observed), 1.0);
  ASSERT_TRUE(seed.accept_new);
  safety.committed(seed, 1.0);
  auto previous = seed.output;
  for (int frame = 0; frame < 3; ++frame) {
    observed.requested.arm[3] = observed.arm->position[3] - 0.02;
    const double now = 1.02 + config.period_s * frame;
    const auto decision = safety.filter(observed, now);
    ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
    if (decision.accept_new) {
      auto path = observed;
      path.requested = decision.output;
      const auto verified = freeze091650_path(path, config, true);
      EXPECT_TRUE(verified.thumb_target_accepted);
      EXPECT_TRUE(verified.nonlinear_verified);
      expect_freeze091650_command(decision.output, previous, config);
    } else {
      expect_held_pair(decision, previous);
    }
    safety.committed(decision, now);
    previous = decision.output;
  }
  EXPECT_EQ(safety.diagnostics().published, 4u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
}

TEST(CbfSafety, UnsafeNewThumbSweepCanDeferTheWholeGestureWithoutBlockingVerifiedArmRecovery)
{
  const auto config = fixture_config(freeze091650_config());
  auto frozen = freeze094711_snapshot(config);
  CbfSafety safety(config);
  // A stationary recovery fixture establishes verified history at h<0. At the
  // same publication instant its arm command box has zero width, so no arm
  // optimization can disguise the recorded unsafe thumb-only sweep.
  const auto seed = safety.filter(frozen, 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  ASSERT_TRUE(seed.recovery_verified);
  safety.committed(seed, 1.0);
  constexpr std::array<double, 2> model_target{0.06153846153846155, -1.519323076923077};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    frozen.requested.hand[axis] = (model_target[axis] - config.command_offset[axis]) / config.command_scale[axis];
  }
  frozen.requested.hand[2] = 0.3;
  const auto deferred = safety.filter(frozen, 1.0);
  ASSERT_TRUE(deferred.accept_new) << safety.diagnostics().reason;
  EXPECT_TRUE(deferred.hand_held);
  EXPECT_TRUE(deferred.recovery_verified);
  EXPECT_TRUE(deferred.geometry_modified);
  EXPECT_EQ(deferred.output.arm, seed.output.arm);
  EXPECT_EQ(deferred.output.hand, seed.output.hand);
  EXPECT_NE(safety.diagnostics().reason.find("arm_recovery_hand_held/"), std::string::npos);
  safety.committed(deferred, 1.0);

  frozen.requested.arm[3] += 0.01;
  const auto retreat = safety.filter(frozen, 1.02);
  ASSERT_TRUE(retreat.accept_new) << safety.diagnostics().reason;
  ASSERT_EQ(retreat.output.arm.size(), 14u);
  EXPECT_TRUE(retreat.recovery_verified);
  auto path = frozen;
  path.requested = retreat.output;
  const auto verified = freeze091650_path(path, config, true);
  EXPECT_TRUE(verified.thumb_target_accepted);
  EXPECT_TRUE(verified.nonlinear_verified);
  double command_movement = 0.0;
  for (std::size_t joint = 0; joint < 14; ++joint) {
    EXPECT_GE(retreat.output.arm[joint], config.lower[joint] - 1e-10);
    EXPECT_LE(retreat.output.arm[joint], config.upper[joint] + 1e-10);
    EXPECT_LE(std::abs(retreat.output.arm[joint] - deferred.output.arm[joint]), config.velocity[joint] * config.period_s + 1e-9);
    command_movement = std::max(command_movement, std::abs(retreat.output.arm[joint] - deferred.output.arm[joint]));
  }
  EXPECT_GT(command_movement, 1e-6);
  EXPECT_TRUE(retreat.output.hand == frozen.requested.hand || retreat.output.hand == seed.output.hand);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
}

TEST(CbfSafety, FasterHandFeedbackWaitsForLaggingArmWithdrawalThenReleasesTheWholeGesture)
{
  auto value = freeze091650_config();
  value["deadline_us"] = 10000.0;
  value["deadline_guard_us"] = 500.0;
  const auto config = fixture_config(value);
  ASSERT_DOUBLE_EQ(config.deadline_us, 10000.0);
  const auto frozen = freeze094711_snapshot(config);
  const auto seed_snapshot = airwall_tracking_lag_seed(frozen);
  CbfSafety safety(config);
  const auto seed = safety.filter(seed_snapshot, 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  safety.committed(seed, 1.0);

  auto goal = seed.output;
  constexpr std::array<double, 2> captured_model_target{0.06153846153846155, -1.519323076923077};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    goal.hand[axis] = (captured_model_target[axis] - config.command_offset[axis]) / config.command_scale[axis];
  }
  goal.hand[2] = 0.3;  // A deferred thumb defers this complete gesture, without rewriting any finger.
  auto coordinated = frozen;
  coordinated.requested = goal;
  const auto combined = freeze091650_path(coordinated, config, true);
  ASSERT_TRUE(combined.thumb_target_accepted);  // The known +0.04 elbow coordinated path is safe.
  ASSERT_TRUE(combined.nonlinear_verified);

  auto numerical = x2::cbf::SafetyFilterConfig{};
  // Independent geometry oracles have a correctness budget; only the runtime
  // adapter below is timed against the real 10 ms / 500 us guard budget.
  numerical.deadline_us = 100000.0;
  numerical.max_step_rad = 6.5;
  numerical.path_check_substeps = 64;
  numerical.allow_airwall_recovery = true;
  const auto hand_first = x2::cbf::verify_thumb_target_with_arm_held(
    freeze091650_input(coordinated, config), numerical);
  ASSERT_FALSE(hand_first.thumb_target_accepted);  // Hand-first ordering of that same request is unsafe.
  ASSERT_FALSE(hand_first.deadline_missed);

  // Synthetic tracking model, not SDK behavior or captured historical targets:
  // the arm follows a 2-3-frame-old published command with a fixed elbow bias;
  // hand feedback follows the newest published gesture, so it can move first.
  constexpr double elbow_bias = -0.04;
  const double right_roll_bias = frozen.arm->position[8] - seed.output.arm[8];
  std::vector<Output> published(3, seed.output);
  int deferred_frames = 0;
  int advancing_frames = 0;
  bool released = false;
  double final_actual_margin = -1.0;
  constexpr int frames = 24;
  for (int frame = 0; frame < frames; ++frame) {
    SCOPED_TRACE(frame);
    const auto previous = published.back();
    auto observed = frozen;
    observed.arm->position = published[published.size() - 3].arm;
    observed.arm->position[3] += elbow_bias;
    observed.arm->position[8] += right_roll_bias;
    observed.thumb->position = {previous.hand[0], previous.hand[1]};
    observed.requested = goal;
    observed.requested.arm[3] = seed.output.arm[3] + 0.05 + 0.002 * frame;
    observed.requested.sequence += frame + 1;
    if (frame == 0) {
      for (std::size_t joint = 0; joint < 14; ++joint) {
        EXPECT_NEAR(observed.arm->position[joint], frozen.arm->position[joint], 1e-12);
      }
    }

    const auto raw_input = freeze091650_input(observed, config);
    auto stationary = raw_input;
    stationary.q_thumb_urdf = stationary.q_thumb_current_urdf;
    const auto actual = x2::cbf::verify_thumb_target_with_arm_held(stationary, numerical);
    ASSERT_TRUE(actual.thumb_target_accepted);
    ASSERT_FALSE(actual.deadline_missed);
    final_actual_margin = actual.input_min_safety_margin_m;
    if (frame == 0) {EXPECT_LT(final_actual_margin, 0.0);}

    const double now = 1.02 + config.period_s * frame;
    auto supplied = observed;
    if (frame == 1) {supplied.arm->age_s = config.feedback_max_age_s + 0.001;}
    auto decision = safety.filter(supplied, now);
    if (frame == 2) {decision = safety.deadline_fallback(decision);}
    ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
    ASSERT_EQ(decision.output.arm.size(), 14u);
    EXPECT_TRUE(decision.output.hand == seed.output.hand || decision.output.hand == goal.hand);
    for (std::size_t joint = 0; joint < 14; ++joint) {
      EXPECT_TRUE(std::isfinite(decision.output.arm[joint]));
      EXPECT_GE(decision.output.arm[joint], config.lower[joint] - 1e-10);
      EXPECT_LE(decision.output.arm[joint], config.upper[joint] + 1e-10);
      EXPECT_LE(std::abs(decision.output.arm[joint] - previous.arm[joint]),
        config.velocity[joint] * config.period_s + 1e-9);
    }
    if (frame == 1 || frame == 2) {
      expect_held_pair(decision, previous);
      EXPECT_FALSE(decision.recovery_verified);  // Cache retransmission is not a new proof.
    } else if (decision.accept_new) {
      EXPECT_GT(decision.output.arm[3] - previous.arm[3], 1e-6);
      ++advancing_frames;
      if (decision.hand_held) {
        ++deferred_frames;
        EXPECT_EQ(decision.output.hand, seed.output.hand);
      }
      if (!released && decision.output.hand == goal.hand) {
        const auto ordering = x2::cbf::verify_thumb_target_with_arm_held(raw_input, numerical);
        EXPECT_TRUE(ordering.thumb_target_accepted);
        EXPECT_TRUE(ordering.nonlinear_verified);
        EXPECT_FALSE(decision.hand_held);
        released = true;
      }
    } else {
      // Ordinary budget exhaustion is allowed to repeat a committed pair,
      // but cannot count as motion, gesture release or a new path proof.
      expect_held_pair(decision, previous);
      EXPECT_FALSE(decision.recovery_verified);
      EXPECT_NE(safety.diagnostics().reason.find("deadline"), std::string::npos)
        << safety.diagnostics().reason;
    }
    if (frame == 0) {
      if (decision.accept_new) {EXPECT_TRUE(decision.hand_held);}
      EXPECT_EQ(decision.output.hand, seed.output.hand);
    }
    auto path = observed;  // Independent check uses fresh numerical actuals, even during the two holds.
    path.requested = decision.output;
    const auto verified = freeze091650_path(path, config, true);
    EXPECT_TRUE(verified.thumb_target_accepted);
    EXPECT_TRUE(verified.nonlinear_verified);
    safety.committed(decision, now);
    published.push_back(decision.output);  // Never advance the model to an unpublished candidate.
  }
  EXPECT_GT(deferred_frames, 0);
  EXPECT_GT(advancing_frames, 1);
  EXPECT_GE(final_actual_margin, 0.0);
  EXPECT_TRUE(released);
  EXPECT_EQ(published.back().hand, goal.hand);
  EXPECT_EQ(safety.diagnostics().published, static_cast<uint64_t>(frames + 1));
  EXPECT_GE(safety.diagnostics().held, 2u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
  const auto & diagnostics = safety.diagnostics();
  std::printf("CBF delayed-feedback: max_elapsed_us=%.0f published=%llu held=%llu withheld=%llu\n",
    diagnostics.max_elapsed_us, static_cast<unsigned long long>(diagnostics.published),
    static_cast<unsigned long long>(diagnostics.held), static_cast<unsigned long long>(diagnostics.withheld));
}

TEST(CbfSafety, ACommittedHoldPreservesTheLatestPublishedHeadWhenTheNextCandidateHasNone)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  snapshot.requested.head = {0.1, -0.05};
  const auto seed = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(seed.accept_new);
  safety.committed(seed, 1.0);
  snapshot.arm->age_s = safety.config().feedback_max_age_s + 0.01;
  snapshot.requested.head = {0.2, -0.1};
  const auto held = safety.filter(snapshot, 1.02);
  expect_held_pair(held, seed.output);
  EXPECT_EQ(held.output.head, snapshot.requested.head);
  safety.committed(held, 1.02);
  snapshot.requested.kind = Output::Kind::kNone;
  snapshot.requested.head.clear();
  const auto missing_head = safety.filter(snapshot, 1.04);
  expect_held_pair(missing_head, seed.output);
  EXPECT_EQ(missing_head.output.head, held.output.head);
  EXPECT_FALSE(missing_head.accept_new);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
}

TEST(CbfSafety, ExpiredPublicationDeadlineKeepsTheCommittedPairWithoutRunningAnyKernelStage)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  ASSERT_FALSE(snapshot.publication_deadline.has_value());  // Existing numerical callers need no shared clock.
  const auto seed = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(seed.accept_new);
  safety.committed(seed, 1.0);
  const auto reuses_before = safety.diagnostics().verification_reuses;
  const auto hand_holds_before = safety.diagnostics().hand_budget_holds;

  snapshot.requested.arm[0] += 0.004;
  snapshot.requested.hand[0] += 0.001;
  snapshot.requested.hand[2] = 0.2;
  snapshot.publication_deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  const auto expired = safety.filter(snapshot, 1.02);
  expect_held_pair(expired, seed.output);
  EXPECT_FALSE(expired.recovery_verified);
  EXPECT_LE(safety.diagnostics().available_budget_us, 0.0);
  EXPECT_DOUBLE_EQ(safety.diagnostics().hand_check_us, 0.0);
  EXPECT_DOUBLE_EQ(safety.diagnostics().verification_us, 0.0);
  EXPECT_DOUBLE_EQ(safety.diagnostics().projection_us, 0.0);
  EXPECT_EQ(safety.diagnostics().verification_reuses, reuses_before);
  EXPECT_EQ(safety.diagnostics().hand_budget_holds, hand_holds_before);
  safety.committed(expired, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().held, 1u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);

  // Node calls inactive on an ownership/context exit. Neither late fallback
  // API may revive a cached command after that explicit exit, even in the same epoch.
  safety.inactive(snapshot.epoch);
  EXPECT_EQ(safety.deadline_fallback(expired).output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.feedback_fallback(expired, "mirror_input_stale").output.kind, Output::Kind::kNone);
  ++snapshot.epoch;
  const auto expired_new_epoch = safety.filter(snapshot, 1.04);
  EXPECT_FALSE(expired_new_epoch.accept_new);
  EXPECT_EQ(expired_new_epoch.output.kind, Output::Kind::kNone);
  EXPECT_DOUBLE_EQ(safety.diagnostics().hand_check_us, 0.0);
  EXPECT_DOUBLE_EQ(safety.diagnostics().verification_us, 0.0);
  EXPECT_DOUBLE_EQ(safety.diagnostics().projection_us, 0.0);

  snapshot.publication_deadline.reset();
  snapshot.requested = seed.output;
  const auto fresh_epoch = safety.filter(snapshot, 1.06);
  ASSERT_TRUE(fresh_epoch.accept_new) << safety.diagnostics().reason;
  safety.committed(fresh_epoch, 1.06);
  EXPECT_EQ(safety.deadline_fallback(expired).output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.diagnostics().accepted, 2u);
}

TEST(CbfSafety, HandSubBudgetTimeoutDefersTheWholeGestureButLeavesBudgetForArmWithdrawal)
{
  auto value = freeze091650_config();
  value["deadline_us"] = 100000.0;
  value["deadline_guard_us"] = 40000.0;
  const auto config = fixture_config(value);
  CbfSafety safety(config);
  auto observed = freeze094711_snapshot(config);
  const auto seed = safety.filter(airwall_tracking_lag_seed(observed), 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  safety.committed(seed, 1.0);
  const auto budget_holds_before = safety.diagnostics().hand_budget_holds;

  observed.requested = seed.output;
  observed.requested.arm[3] += 0.01;
  constexpr std::array<double, 2> captured_model_target{0.06153846153846155, -1.519323076923077};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    observed.requested.hand[axis] =
      (captured_model_target[axis] - config.command_offset[axis]) / config.command_scale[axis];
  }
  observed.requested.hand[2] = 0.3;
  ASSERT_NE(observed.requested.hand, seed.output.hand);
  // At most (100 ms - 40 ms) / 4 = 15 ms is offered to the hand query.
  // Its existing 40 ms guard makes that sub-budget expire immediately, without
  // sleeping or relying on a microsecond timing race. The arm retains ~60 ms compute.
  const auto retreat = safety.filter(observed, 1.02);
  ASSERT_EQ(retreat.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
  ASSERT_TRUE(retreat.accept_new) << safety.diagnostics().reason;
  EXPECT_TRUE(retreat.hand_held);
  EXPECT_TRUE(retreat.recovery_verified);
  EXPECT_EQ(retreat.output.hand, seed.output.hand);  // Every finger is deferred with the new roots.
  EXPECT_GT(retreat.output.arm[3] - seed.output.arm[3], 1e-6);
  EXPECT_GT(expect_freeze091650_command(retreat.output, seed.output, config), 1e-6);
  EXPECT_EQ(safety.diagnostics().hand_budget_holds, budget_holds_before + 1);
  EXPECT_GT(safety.diagnostics().available_budget_us, config.guard_us);
  EXPECT_TRUE(std::isfinite(safety.diagnostics().hand_check_us));
  EXPECT_GE(safety.diagnostics().hand_check_us, 0.0);
  EXPECT_TRUE(std::isfinite(safety.diagnostics().verification_us));
  EXPECT_GE(safety.diagnostics().verification_us, 0.0);
  EXPECT_EQ(safety.diagnostics().reason.find("deadline"), std::string::npos);

  auto published_path = observed;
  published_path.requested = retreat.output;
  const auto verified = freeze091650_path(published_path, config, true);
  ASSERT_FALSE(verified.deadline_missed);
  EXPECT_TRUE(verified.thumb_target_accepted);
  EXPECT_TRUE(verified.nonlinear_verified);
  safety.committed(retreat, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 2u);
  EXPECT_EQ(safety.diagnostics().published, 2u);
  EXPECT_EQ(safety.diagnostics().held, 0u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
}

TEST(CbfSafety, ProjectedRecoveryReusesItsKernelProofForTheUnchangedActualSnapshot)
{
  const auto config = fixture_config(freeze091650_config());
  CbfSafety safety(config);
  auto observed = freeze094711_snapshot(config);
  // The existing stationary-recovery fixture establishes a verified tangent
  // target at h<0. This is numerical history, not a captured previous SDK command.
  const auto seed = safety.filter(observed, 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  ASSERT_TRUE(seed.recovery_verified);
  safety.committed(seed, 1.0);
  const auto reuses_before = safety.diagnostics().verification_reuses;
  observed.requested = seed.output;
  observed.requested.arm[3] -= 0.005;  // Existing captured-pose inward regression forces projection.
  const auto original_arm = observed.arm->position;
  const auto original_thumb = observed.thumb->position;
  const auto nominal = freeze091650_path(observed, config, true);
  ASSERT_FALSE(nominal.deadline_missed);
  ASSERT_FALSE(nominal.thumb_target_accepted);

  const auto projected = safety.filter(observed, 1.02);
  ASSERT_EQ(projected.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
  ASSERT_TRUE(projected.accept_new) << safety.diagnostics().reason;
  EXPECT_TRUE(projected.geometry_modified);
  EXPECT_TRUE(projected.recovery_verified);
  EXPECT_FALSE(projected.hand_held);
  EXPECT_GT(safety.diagnostics().verification_reuses, reuses_before);
  EXPECT_EQ(projected.output.hand, seed.output.hand);
  expect_freeze091650_command(projected.output, seed.output, config);
  EXPECT_EQ(observed.arm->position, original_arm);
  EXPECT_EQ(observed.thumb->position, original_thumb);

  // Independent exact verification retains the SAME original measured pose;
  // replacing it with the candidate would conceal a stale/incorrect proof reuse.
  auto published_path = observed;
  published_path.requested = projected.output;
  const auto checked = freeze091650_path(published_path, config, true);
  ASSERT_FALSE(checked.deadline_missed);
  EXPECT_TRUE(checked.thumb_target_accepted);
  EXPECT_TRUE(checked.nonlinear_verified);
  EXPECT_TRUE(checked.recovery_verified);
  EXPECT_TRUE(std::isfinite(safety.diagnostics().projection_us));
  EXPECT_GE(safety.diagnostics().projection_us, 0.0);
  safety.committed(projected, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 2u);
  EXPECT_EQ(safety.diagnostics().withheld, 0u);
}

TEST(CbfSafety, HoldReleaseConfigurationIsBounded)
{
  const auto defaults = fixture_config();
  EXPECT_DOUBLE_EQ(defaults.hold_release_after_s, 0.2);
  EXPECT_DOUBLE_EQ(defaults.collision_release_after_s, 2.0);
  EXPECT_DOUBLE_EQ(defaults.collision_release_without_history_s, 0.5);
  EXPECT_DOUBLE_EQ(defaults.collision_release_speed_scale, 0.3);
  const std::vector<std::tuple<std::string, double, std::vector<double>>> fields{
    {"hold_release_after_s", 0.5, {-0.01, 1.01}},
    {"collision_release_after_s", 3.0, {-0.01, 10.01}},
    {"collision_release_without_history_s", 1.0, {-0.01, 5.01}},
    {"collision_release_speed_scale", 1.0, {0.0, -0.1, 1.01}},
  };
  for (const auto & [field, valid, invalids] : fields) {
    SCOPED_TRACE(field);
    auto value = urdf_fixture_json();
    value[field] = valid;
    const auto config = fixture_config(value);
    const double read = field == "hold_release_after_s" ? config.hold_release_after_s :
      field == "collision_release_after_s" ? config.collision_release_after_s :
      field == "collision_release_without_history_s" ? config.collision_release_without_history_s :
      config.collision_release_speed_scale;
    EXPECT_DOUBLE_EQ(read, valid);
    for (double invalid : invalids) {
      value[field] = invalid;
      EXPECT_THROW(fixture_config(value), std::invalid_argument) << invalid;
    }
  }
}

TEST(CbfSafety, FaultedOrMissingThumbFeedbackUsesTheLastPublishedCommandInTheSameFrame)
{
  CbfSafety safety(fixture_config());
  const auto & band = safety.config().thumb_model_limits;
  // Before any publication the requested command stands in for the reading.
  auto first = urdf_snapshot();
  first.thumb.reset();
  first.requested.hand[0] = 0.5 * (band[0][0] + band[0][1]);
  const auto seed = safety.filter(first, 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, "unavailable");
  EXPECT_DOUBLE_EQ(safety.diagnostics().thumb_current_urdf[0], first.requested.hand[0]);
  safety.committed(seed, 1.0);

  for (int faulted_case = 0; faulted_case < 2; ++faulted_case) {
    SCOPED_TRACE(faulted_case);
    auto snapshot = urdf_snapshot();
    if (faulted_case == 0) {
      snapshot.thumb_faulted = {true, true};  // Both hands faulted: every reading is replaced.
      snapshot.thumb->position[0] = 5.0;  // A faulted hand's reading is not used even clamped.
    } else {
      snapshot.thumb.reset();
    }
    const auto decision = safety.filter(snapshot, 1.02 + 0.02 * faulted_case);
    ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
    EXPECT_EQ(decision.output.hand, snapshot.requested.hand);
    EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, faulted_case == 0 ? "faulted" : "unavailable");
    EXPECT_DOUBLE_EQ(safety.diagnostics().thumb_current_urdf[0], seed.output.hand[0]);
    safety.committed(decision, 1.02 + 0.02 * faulted_case);
  }
  EXPECT_EQ(safety.diagnostics().held, 0u);
}

TEST(CbfSafety, UnreachableThumbTargetsAreClampedIntoTheModelRangeWithoutNativeLimits)
{
  CbfSafety safety(fixture_config());
  ASSERT_FALSE(safety.config().native_limits_available);
  const auto & band = safety.config().thumb_model_limits;
  auto snapshot = urdf_snapshot();
  snapshot.requested.hand[0] = band[0][1] + 0.35;
  snapshot.requested.hand[11] = band[3][0] - 0.35;
  const auto decision = safety.filter(snapshot, 1.0);
  ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
  EXPECT_TRUE(decision.modified);
  EXPECT_DOUBLE_EQ(decision.output.hand[0], band[0][1]);
  EXPECT_DOUBLE_EQ(decision.output.hand[11], band[3][0]);
  EXPECT_EQ(decision.output.arm, snapshot.requested.arm);
}

TEST(CbfSafety, NonCollisionHoldsReleaseAfterTheConfiguredTimeWithinJointAndVelocityLimits)
{
  CbfSafety safety(fixture_config());
  const auto & config = safety.config();
  const auto seed = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(seed.accept_new);
  safety.committed(seed, 1.0);

  auto goal = urdf_snapshot();
  goal.requested.arm[0] = std::min(config.upper[0], goal.requested.arm[0] + 0.3);
  goal.requested.arm[3] = config.lower[3] - 1.0;  // Past the joint limit: published clamped.
  goal.requested.hand[0] = config.thumb_model_limits[0][1] + 0.2;
  goal.requested.hand[5] = 0.4;
  Output previous = seed.output;
  int held_frames = 0;
  int released_frames = 0;
  double first_release = -1.0;
  for (int frame = 0; frame < 50; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.02 + config.period_s * frame;
    // Every non-verdict hold kind counts toward the same episode.
    auto snapshot = goal;
    CbfDecision decision;
    if (frame % 3 == 0) {
      snapshot.arm->age_s = config.feedback_max_age_s + 0.001;
      decision = safety.filter(snapshot, now);
    } else if (frame % 3 == 1) {
      decision = safety.deadline_fallback(safety.filter(snapshot, now));
    } else {
      decision = safety.feedback_fallback(safety.filter(snapshot, now), "feedback_unavailable");
    }
    ASSERT_FALSE(decision.accept_new);
    ASSERT_FALSE(decision.collision_refusal);
    decision = safety.release_if_stuck(decision, snapshot, now);
    ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
    if (now - 1.02 + 1e-9 < config.hold_release_after_s) {
      expect_held_pair(decision, seed.output);
      EXPECT_FALSE(safety.diagnostics().releasing);
      ++held_frames;
    } else {
      ASSERT_TRUE(decision.accept_new);
      EXPECT_TRUE(decision.released);
      EXPECT_TRUE(safety.diagnostics().releasing);
      EXPECT_EQ(safety.diagnostics().reason.find("released/"), 0u) << safety.diagnostics().reason;
      if (first_release < 0.0) {first_release = now;}
      for (std::size_t joint = 0; joint < 14; ++joint) {
        SCOPED_TRACE(joint);
        EXPECT_GE(decision.output.arm[joint], config.lower[joint] - 1e-12);
        EXPECT_LE(decision.output.arm[joint], config.upper[joint] + 1e-12);
        EXPECT_LE(std::abs(decision.output.arm[joint] - previous.arm[joint]),
          config.velocity[joint] * config.period_s + 1e-9);
      }
      EXPECT_DOUBLE_EQ(decision.output.hand[0], config.thumb_model_limits[0][1]);
      EXPECT_DOUBLE_EQ(decision.output.hand[5], 0.4);
      ++released_frames;
    }
    safety.committed(decision, now);
    previous = decision.output;
  }
  EXPECT_NEAR(first_release, 1.22, 1e-9);
  EXPECT_EQ(held_frames, 10);
  EXPECT_EQ(released_frames, 40);
  EXPECT_EQ(safety.diagnostics().release_episodes, 1u);
  EXPECT_EQ(safety.diagnostics().released, 40u);
  // The release walked the arm to the reachable goal.
  EXPECT_DOUBLE_EQ(previous.arm[0], goal.requested.arm[0]);
  EXPECT_DOUBLE_EQ(previous.arm[3], config.lower[3]);

  // A verified frame ends the episode; a new hold starts a new 0.2 s count.
  auto valid = urdf_snapshot();
  valid.arm->position = previous.arm;
  valid.requested.arm = previous.arm;
  auto verified = safety.release_if_stuck(safety.filter(valid, 2.02), valid, 2.02);
  ASSERT_TRUE(verified.accept_new) << safety.diagnostics().reason;
  EXPECT_FALSE(verified.released);
  EXPECT_FALSE(safety.diagnostics().releasing);
  EXPECT_DOUBLE_EQ(safety.diagnostics().stuck_s, 0.0);
  safety.committed(verified, 2.02);
  valid.arm->age_s = config.feedback_max_age_s + 0.001;
  EXPECT_FALSE(safety.release_if_stuck(safety.filter(valid, 2.04), valid, 2.04).accept_new);
  EXPECT_TRUE(safety.release_if_stuck(safety.filter(valid, 2.24), valid, 2.24).released);
  EXPECT_EQ(safety.diagnostics().release_episodes, 2u);

  safety.inactive(valid.epoch);
  EXPECT_EQ(safety.diagnostics().reason, "inactive");
  EXPECT_FALSE(safety.diagnostics().releasing);
  EXPECT_DOUBLE_EQ(safety.diagnostics().stuck_s, 0.0);
}

TEST(CbfSafety, ReleaseWithoutAPublishedFrameStepsFromTheMeasuredArm)
{
  CbfSafety safety(fixture_config());
  const auto & config = safety.config();
  auto snapshot = urdf_snapshot();
  snapshot.requested.arm[0] = std::min(config.upper[0], snapshot.arm->position[0] + 0.3);
  snapshot.publication_deadline = std::chrono::steady_clock::now();  // Every frame misses its budget.
  auto decision = safety.release_if_stuck(safety.filter(snapshot, 1.0), snapshot, 1.0);
  EXPECT_EQ(decision.output.kind, Output::Kind::kNone);
  EXPECT_FALSE(decision.collision_refusal);
  decision = safety.release_if_stuck(safety.filter(snapshot, 1.2), snapshot, 1.2);
  ASSERT_TRUE(decision.released) << safety.diagnostics().reason;
  ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
  EXPECT_NEAR(decision.output.arm[0], snapshot.arm->position[0] + config.velocity[0] * config.period_s, 1e-12);
  EXPECT_EQ(decision.output.hand, snapshot.requested.hand);

  // Nothing complete to publish: an invalid command keeps the hold however long.
  snapshot.requested.hand[3] = std::numeric_limits<double>::quiet_NaN();
  for (double now : {1.4, 1.6, 2.0}) {
    EXPECT_FALSE(safety.release_if_stuck(safety.filter(snapshot, now), snapshot, now).accept_new);
  }
}

TEST(CbfSafety, ACollisionVerdictHoldsAtTheBoundaryForItsOwnTimeThenReleasesSlowly)
{
  const auto config = fixture_config(freeze091650_config());
  ASSERT_DOUBLE_EQ(config.collision_release_after_s, 2.0);
  ASSERT_DOUBLE_EQ(config.collision_release_speed_scale, 0.3);
  const auto frozen = freeze094711_snapshot(config);
  CbfSafety safety(config);
  const auto seed = safety.filter(frozen, 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  safety.committed(seed, 1.0);

  // A non-verdict hold starts the release clock ...
  auto stale = frozen;
  stale.arm->age_s = config.feedback_max_age_s + 0.001;
  EXPECT_FALSE(safety.release_if_stuck(safety.filter(stale, 1.0), stale, 1.0).accept_new);
  EXPECT_DOUBLE_EQ(safety.diagnostics().stuck_s, 0.0);
  EXPECT_FALSE(safety.diagnostics().stuck_collision);

  // ... and a verdict joins it, so the whole hold now waits the collision time.
  // The measured elbow has left the committed one inside the wall; at the
  // instant of the last publication the command box has zero width, so the only
  // candidate is a path back into the wall. Offline this is the reproducible
  // verdict; the release clock is advanced independently.
  auto lagging = frozen;
  lagging.arm->position[3] += 0.05;
  for (double release_clock : {1.1, 1.25, 1.5, 2.99}) {
    SCOPED_TRACE(release_clock);
    const auto verdict = safety.filter(lagging, 1.0);
    ASSERT_TRUE(verdict.collision_refusal) << safety.diagnostics().reason;
    ASSERT_FALSE(verdict.accept_new);
    const auto kept = safety.release_if_stuck(verdict, lagging, release_clock);
    EXPECT_FALSE(kept.released);
    expect_held_pair(kept, seed.output);
    EXPECT_FALSE(safety.diagnostics().releasing);
    EXPECT_TRUE(safety.diagnostics().stuck_collision);
    EXPECT_NEAR(safety.diagnostics().stuck_s, release_clock - 1.0, 1e-9);
    // The node's late recheck keeps the verdict flag on the cached hold.
    EXPECT_TRUE(safety.deadline_fallback(verdict).collision_refusal);
    // A non-verdict frame amid the verdicts cannot cut the hold to 0.2 s.
    const auto amid = safety.release_if_stuck(safety.filter(stale, 1.0), stale, release_clock);
    EXPECT_FALSE(amid.released);
    expect_held_pair(amid, seed.output);
  }
  EXPECT_EQ(safety.diagnostics().release_episodes, 0u);

  // At 2 s the request goes out, the arm at 0.3x the velocity step.
  const auto verdict = safety.filter(lagging, 1.0);
  ASSERT_TRUE(verdict.collision_refusal);
  const auto released = safety.release_if_stuck(verdict, lagging, 3.0);
  ASSERT_TRUE(released.accept_new);
  ASSERT_TRUE(released.released);
  EXPECT_TRUE(safety.diagnostics().releasing);
  EXPECT_EQ(safety.diagnostics().release_episodes, 1u);
  EXPECT_EQ(safety.diagnostics().collision_release_episodes, 1u);
  for (std::size_t joint = 0; joint < 14; ++joint) {
    SCOPED_TRACE(joint);
    EXPECT_LE(std::abs(released.output.arm[joint] - seed.output.arm[joint]),
      0.3 * config.velocity[joint] * config.period_s + 1e-12);
  }
  EXPECT_EQ(released.output.hand, lagging.requested.hand);
  safety.committed(seed, 1.0);

  // One frame later the operator moves away and the frame is accepted.
  auto away = lagging;
  away.requested.arm[3] = lagging.arm->position[3] + 0.02;
  const auto moved = safety.release_if_stuck(safety.filter(away, 1.02), away, 3.02);
  ASSERT_TRUE(moved.accept_new) << safety.diagnostics().reason;
  EXPECT_FALSE(moved.released);
  EXPECT_FALSE(safety.diagnostics().releasing);
  EXPECT_FALSE(safety.diagnostics().stuck_collision);
  EXPECT_DOUBLE_EQ(safety.diagnostics().stuck_s, 0.0);
  EXPECT_GT(moved.output.arm[3], seed.output.arm[3] + 1e-6);
  auto path = away;
  path.requested = moved.output;
  const auto verified = freeze091650_path(path, config, true);
  EXPECT_TRUE(verified.nonlinear_verified);
}

TEST(CbfSafety, AReleaseUnderWayIsNotStoppedByAVerdictOnlySlowedDown)
{
  const auto config = fixture_config(freeze091650_config());
  const auto frozen = freeze094711_snapshot(config);
  CbfSafety safety(config);
  const auto seed = safety.filter(frozen, 1.0);
  ASSERT_TRUE(seed.accept_new) << safety.diagnostics().reason;
  safety.committed(seed, 1.0);
  auto stale = frozen;
  stale.arm->age_s = config.feedback_max_age_s + 0.001;
  stale.requested.arm[3] -= 0.1;
  EXPECT_FALSE(safety.release_if_stuck(safety.filter(stale, 1.0), stale, 1.0).accept_new);
  const auto plain = safety.release_if_stuck(safety.filter(stale, 1.0), stale, 1.2);
  ASSERT_TRUE(plain.released);
  EXPECT_FALSE(safety.diagnostics().stuck_collision);
  EXPECT_NEAR(std::abs(plain.output.arm[3] - seed.output.arm[3]), config.velocity[3] * config.period_s, 1e-12);

  // The same release, not committed, meets the 094711 verdict on the next frame.
  auto lagging = frozen;
  lagging.arm->position[3] += 0.05;
  const auto verdict = safety.filter(lagging, 1.0);
  ASSERT_TRUE(verdict.collision_refusal) << safety.diagnostics().reason;
  const auto slowed = safety.release_if_stuck(verdict, lagging, 1.22);
  ASSERT_TRUE(slowed.released);
  EXPECT_TRUE(safety.diagnostics().releasing);
  EXPECT_TRUE(safety.diagnostics().stuck_collision);
  for (std::size_t joint = 0; joint < 14; ++joint) {
    SCOPED_TRACE(joint);
    EXPECT_LE(std::abs(slowed.output.arm[joint] - seed.output.arm[joint]),
      0.3 * config.velocity[joint] * config.period_s + 1e-12);
  }
  EXPECT_EQ(safety.diagnostics().release_episodes, 1u);
  EXPECT_EQ(safety.diagnostics().collision_release_episodes, 0u);  // It began as a plain release.
}

TEST(CbfSafety, AVerdictBeforeAnythingWasPublishedIsReleasedAfterHalfASecondAndKeepsGoing)
{
  const auto config = fixture_config(freeze091650_config());
  ASSERT_DOUBLE_EQ(config.collision_release_without_history_s, 0.5);
  // A fresh epoch: nothing published yet, so a verdict has no pair to hold and
  // publishes nothing, frame after frame. Measured elbow 0.15 rad lower than
  // the 094711 snapshot, request 0.1 rad further down: the thumb sweep from the
  // reading to the request is rejected on the first frame.
  auto start = freeze094711_snapshot(config);
  start.arm->position[3] -= 0.15;
  start.requested.arm = start.arm->position;
  start.requested.arm[3] -= 0.1;
  CbfSafety safety(config);
  Output previous;
  int empty_frames = 0;
  int released_frames = 0;
  for (int frame = 0; frame < 60; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.0 + config.period_s * frame;
    const auto decision = safety.filter(start, now);
    if (frame == 0) {
      ASSERT_TRUE(decision.collision_refusal) << safety.diagnostics().reason;
      ASSERT_EQ(decision.output.kind, Output::Kind::kNone);
    }
    const auto out = safety.release_if_stuck(decision, start, now);
    if (now - 1.0 + 1e-9 < 0.5) {
      EXPECT_EQ(out.output.kind, Output::Kind::kNone);
      EXPECT_FALSE(safety.diagnostics().releasing);
      ++empty_frames;
    } else {
      // Once begun, the release goes on through the verdicts that follow.
      ASSERT_TRUE(out.released) << safety.diagnostics().reason;
      ASSERT_EQ(out.output.kind, Output::Kind::kUpperBody);
      const auto & from = released_frames == 0 ? start.arm->position : previous.arm;
      for (std::size_t joint = 0; joint < 14; ++joint) {
        SCOPED_TRACE(joint);
        EXPECT_LE(std::abs(out.output.arm[joint] - from[joint]),
          0.3 * config.velocity[joint] * config.period_s + 1e-12);
      }
      ++released_frames;
    }
    safety.committed(out, now);
    if (out.output.kind == Output::Kind::kUpperBody) {previous = out.output;}
  }
  EXPECT_EQ(empty_frames, 25);
  EXPECT_EQ(released_frames, 35);
  EXPECT_EQ(safety.diagnostics().release_episodes, 1u);
  EXPECT_EQ(safety.diagnostics().collision_release_episodes, 1u);
  // The slowed release reached the requested elbow.
  EXPECT_NEAR(previous.arm[3], start.requested.arm[3], 1e-9);

  // A new epoch starts a new count: nothing published in it yet.
  auto next = start;
  next.epoch = start.epoch + 1;
  EXPECT_EQ(safety.release_if_stuck(safety.filter(next, 3.0), next, 3.0).output.kind, Output::Kind::kNone);
  EXPECT_FALSE(safety.diagnostics().releasing);
  EXPECT_DOUBLE_EQ(safety.diagnostics().stuck_s, 0.0);
  EXPECT_EQ(safety.release_if_stuck(safety.filter(next, 3.48), next, 3.48).output.kind, Output::Kind::kNone);
  EXPECT_TRUE(safety.release_if_stuck(safety.filter(next, 3.5), next, 3.5).released);
}

TEST(CbfSafety, ABlockedArmNeverTurnsMoveAwayOrStillRequestsIntoAHold)
{
  // The robot does not follow the published command (blocked or lagging) while
  // the operator keeps moving. At 50 Hz none of these frames may hold.
  const auto config = fixture_config(freeze091650_config());
  const auto frozen = freeze094711_snapshot(config);
  for (double offset : {-0.05, 0.0, 0.05, 0.2}) {
    for (double direction : {0.0, 0.01}) {
      SCOPED_TRACE(offset);
      SCOPED_TRACE(direction);
      CbfSafety safety(config);
      const auto seed = safety.filter(frozen, 1.0);
      ASSERT_TRUE(seed.accept_new);
      safety.committed(seed, 1.0);
      auto snapshot = frozen;
      snapshot.arm->position[3] += offset;
      for (int frame = 0; frame < 40; ++frame) {
        const double now = 1.02 + config.period_s * frame;
        snapshot.requested.arm[3] = std::min(config.upper[3], frozen.requested.arm[3] + direction * (frame + 1));
        const auto decision = safety.release_if_stuck(safety.filter(snapshot, now), snapshot, now);
        ASSERT_TRUE(decision.accept_new) << frame << " " << safety.diagnostics().reason;
        EXPECT_FALSE(decision.released);
        safety.committed(decision, now);
      }
    }
  }
}

// Every reason a frame can go unverified other than a collision verdict, held
// for the whole run. The same moving request is sent with and without it: the
// control is accepted from the first frame, the fault is held for at most the
// release time and then followed. Thumb readings never hold a frame at all.
struct PersistentFault
{
  const char * name;
  bool holds;
  const char * reason;  // The hold's reason; nullptr when the frame never reached the end of the filter.
  std::function<void(CbfSnapshot &, CbfSafety &)> input = nullptr;
  // The node's checks after the filter, as dataplane_node.cpp applies them.
  std::function<CbfDecision(CbfSafety &, const CbfDecision &)> node = nullptr;
  std::function<void(json &)> configure = nullptr;
  std::function<void(const CbfDiagnostics &)> while_held = nullptr;
};

void expect_follows(const PersistentFault & fault, bool faulty)
{
  auto value = urdf_fixture_json();
  if (faulty && fault.configure) {fault.configure(value);}
  CbfSafety safety(fixture_config(value));
  const auto & config = safety.config();
  const auto standing = urdf_snapshot();
  const auto seed = safety.filter(standing, 1.0);
  if (seed.accept_new) {safety.committed(seed, 1.0);}
  ASSERT_EQ(seed.accept_new, !(faulty && fault.configure));

  auto goal = standing;
  goal.requested.arm[0] = std::min(config.upper[0], goal.requested.arm[0] + 0.1);
  goal.requested.hand[5] = 0.4;
  Output previous = standing.requested;
  int held = 0;
  for (int frame = 0; frame < 50; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.02 + config.period_s * frame;
    auto snapshot = goal;
    // A new thumb target every frame, so the thumb ordering check runs every frame.
    snapshot.requested.hand[0] += frame % 2 == 0 ? 0.05 : -0.05;
    snapshot.arm->position = previous.arm;  // The robot follows what was published.
    safety.limit_kernel_calls_for_test(-1);
    if (faulty && fault.input) {fault.input(snapshot, safety);}
    auto decision = safety.filter(snapshot, now);
    if (faulty && fault.node) {decision = fault.node(safety, decision);}
    decision = safety.release_if_stuck(decision, snapshot, now);
    EXPECT_FALSE(decision.collision_refusal) << safety.diagnostics().reason;
    if (faulty && fault.holds && now - 1.02 + 1e-9 < config.hold_release_after_s) {
      EXPECT_FALSE(decision.accept_new) << safety.diagnostics().reason;
      if (fault.reason) {EXPECT_EQ(safety.diagnostics().reason, fault.reason);}
      if (fault.while_held) {fault.while_held(safety.diagnostics());}
      ++held;
    } else {
      ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
      EXPECT_EQ(decision.released, faulty && fault.holds) << safety.diagnostics().reason;
      ASSERT_EQ(decision.output.kind, Output::Kind::kUpperBody);
      for (std::size_t joint = 0; joint < 14; ++joint) {
        SCOPED_TRACE(joint);
        EXPECT_LE(std::abs(decision.output.arm[joint] - previous.arm[joint]),
          config.velocity[joint] * config.period_s + 1e-9);
      }
      EXPECT_DOUBLE_EQ(decision.output.hand[5], 0.4);
      previous = decision.output;
    }
    safety.committed(decision, now);
  }
  EXPECT_EQ(held, faulty && fault.holds ? 10 : 0);
  // The filter may stop short of a request near a constraint; a release publishes it.
  EXPECT_GT(previous.arm[0], standing.requested.arm[0] + 0.08);
  if (faulty && fault.holds) {EXPECT_DOUBLE_EQ(previous.arm[0], goal.requested.arm[0]);}
}

std::vector<PersistentFault> persistent_faults()
{
  const auto deadline_fallback = [](CbfSafety & safety, const CbfDecision & decision) {
      return safety.deadline_fallback(decision);
    };
  const auto publication = [](const char * reason) {
      return [reason](CbfSafety & safety, const CbfDecision & decision) {
               return safety.feedback_fallback(decision, reason);
             };
    };
  const auto thumb_axis_band = [](const CbfSafety & safety, int axis) {
      return safety.config().thumb_model_limits[axis];
    };
  return {
    {"arm_feedback_stale", true, "feedback_unavailable", [](CbfSnapshot & s, CbfSafety & safety) {
        s.arm->age_s = safety.config().feedback_max_age_s + 0.001;
      }},
    {"arm_feedback_missing", true, "feedback_unavailable", [](CbfSnapshot & s, CbfSafety &) {s.arm.reset();}},
    {"budget_spent_before_the_kernel", true, "deadline", [](CbfSnapshot & s, CbfSafety &) {
        s.publication_deadline = std::chrono::steady_clock::now();
      }},
    {"budget_spent_after_the_thumb_check", true, "deadline", [](CbfSnapshot &, CbfSafety & safety) {
        safety.limit_kernel_calls_for_test(1);
      }, nullptr, nullptr, [](const CbfDiagnostics & d) {
        EXPECT_GT(d.hand_check_us, 0.0);
        EXPECT_DOUBLE_EQ(d.verification_us, 0.0);
      }},
    {"budget_spent_after_the_nominal_check", true, "deadline", [](CbfSnapshot &, CbfSafety & safety) {
        safety.limit_kernel_calls_for_test(2);
      }, nullptr, nullptr, [](const CbfDiagnostics & d) {
        EXPECT_GT(d.hand_check_us, 0.0);
        EXPECT_GT(d.verification_us, 0.0);
      }},
    {"node_deadline", true, "node_deadline", nullptr, deadline_fallback},
    {"publication_snapshot_feedback_expired", true, "publication_snapshot_feedback_expired", nullptr, publication("snapshot_feedback_expired")},
    {"mirror_input_stale", true, "mirror_input_stale", [](CbfSnapshot & s, CbfSafety &) {s.source_ready = false;}},
    {"publication_mirror_input_stale", true, "publication_mirror_input_stale", nullptr, publication("mirror_input_stale")},
    {"hand_configuration_mismatch", true, "hand_configuration_mismatch", [](CbfSnapshot & s, CbfSafety &) {
        s.hand_configuration_matches = false;
      }},
    {"publication_hand_configuration_mismatch", true,
      "publication_hand_configuration_mismatch", nullptr, publication("hand_configuration_mismatch")},
    {"mixed_named_arm", true, "mixed_named_arm", [](CbfSnapshot & s, CbfSafety &) {s.named_arm_cache = true;}},
    {"mapping_required", true, "mapping_required", nullptr, nullptr, [](json & value) {value.erase("thumb_mapping");}},
    // dataplane_node.cpp's catch: the frame carries only its epoch.
    {"filter_exception", true, nullptr, nullptr, [](CbfSafety &, const CbfDecision & decision) {
        CbfDecision failed;
        failed.epoch = decision.epoch;
        return failed;
      }},
    {"unknown_reason", true, "publication_a_reason_added_later", nullptr, publication("a_reason_added_later")},
    {"thumb_out_of_range", false, nullptr, [thumb_axis_band](CbfSnapshot & s, CbfSafety & safety) {
        s.thumb->position[0] = thumb_axis_band(safety, 0)[1] + 0.35;
        s.thumb->position[3] = thumb_axis_band(safety, 3)[0] - 0.35;
      }},
    {"thumb_saturated", false, nullptr, [](CbfSnapshot & s, CbfSafety &) {
        std::fill(s.thumb->position.begin(), s.thumb->position.end(), 1.74);
      }},
    {"thumb_nan", false, nullptr, [](CbfSnapshot & s, CbfSafety &) {
        s.thumb->position[1] = std::numeric_limits<double>::quiet_NaN();
      }},
    {"thumb_faulted", false, nullptr, [](CbfSnapshot & s, CbfSafety &) {s.thumb_faulted = {true, true};}},
    {"thumb_missing", false, nullptr, [](CbfSnapshot & s, CbfSafety &) {s.thumb.reset();}},
    {"thumb_stale", false, nullptr, [](CbfSnapshot & s, CbfSafety & safety) {
        s.thumb->age_s = safety.config().feedback_max_age_s + 0.001;
      }},
  };
}

TEST(CbfSafety, EveryPersistentNonCollisionReasonIsFollowedAfterTheReleaseTime)
{
  for (const auto & fault : persistent_faults()) {
    SCOPED_TRACE(fault.name);
    {
      SCOPED_TRACE("control");
      expect_follows(fault, false);
    }
    expect_follows(fault, true);
  }
}

namespace
{

struct NoStallStats
{
  int frames = 0;
  int verdicts = 0;
  int collision_releases = 0;
  int releases = 0;
  double longest_unpublished_s = 0.0;  // Longest run of frames that accepted no new pair.
};

// Random reaching motions, including the arms across and the hands together,
// with the robot following what was published. No hold may outlast the longest
// configured release time by more than a frame.
NoStallStats run_random_reaching(const CbfSafetyConfig & config, CbfSnapshot snapshot, unsigned seed, int frames)
{
  CbfSafety safety(config);
  std::srand(seed);
  const auto uniform = [](double lo, double hi) {return lo + (hi - lo) * (std::rand() / (RAND_MAX + 1.0));};
  NoStallStats stats;
  std::array<double, 14> goal{};
  std::array<double, 4> thumb_goal{};
  for (std::size_t j = 0; j < 14; ++j) {goal[j] = snapshot.requested.arm[j];}
  double stuck_since = -1.0;
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  for (int frame = 0; frame < frames; ++frame) {
    const double now = 1.0 + config.period_s * frame;
    if (frame % 40 == 0) {
      for (std::size_t j = 0; j < 14; ++j) {goal[j] = uniform(config.lower[j], config.upper[j]);}
    }
    if (frame % 7 == 0) {
      for (std::size_t a = 0; a < 4; ++a) {
        thumb_goal[a] = uniform(config.thumb_model_limits[a][0], config.thumb_model_limits[a][1]);
      }
    }
    // The operator moves fast: 0.1 rad a frame towards the goal, well past the arm step.
    for (std::size_t j = 0; j < 14; ++j) {
      const double delta = std::clamp(goal[j] - snapshot.requested.arm[j], -0.1, 0.1);
      snapshot.requested.arm[j] += delta;
    }
    for (std::size_t a = 0; a < 4; ++a) {
      snapshot.requested.hand[slots[a]] = (thumb_goal[a] - config.command_offset[a]) / config.command_scale[a];
    }
    auto decision = safety.filter(snapshot, now);
    if (decision.collision_refusal) {++stats.verdicts;}
    if (frame % 17 == 0) {decision = safety.deadline_fallback(decision);}
    const bool was_releasing = safety.diagnostics().releasing;
    decision = safety.release_if_stuck(decision, snapshot, now);
    if (safety.diagnostics().releasing && !was_releasing) {
      ++stats.releases;
      if (safety.diagnostics().stuck_collision) {++stats.collision_releases;}
    }
    safety.committed(decision, now);
    if (decision.accept_new) {
      stuck_since = -1.0;
    } else {
      if (stuck_since < 0.0) {stuck_since = now;}
      stats.longest_unpublished_s = std::max(stats.longest_unpublished_s, now - stuck_since);
    }
    // The robot lags: each frame it covers half the way to the published arm.
    if (decision.output.kind == Output::Kind::kUpperBody) {
      for (std::size_t j = 0; j < 14; ++j) {
        snapshot.arm->position[j] += 0.5 * (decision.output.arm[j] - snapshot.arm->position[j]);
      }
      for (std::size_t a = 0; a < snapshot.thumb->position.size(); ++a) {
        auto & thumb = snapshot.thumb->position[a];
        thumb += 0.5 * (decision.output.hand[slots[a]] - thumb);
      }
    }
    ++stats.frames;
  }
  return stats;
}

}  // namespace

TEST(CbfSafety, RandomReachingWithOneOrTwoHandsNeverHoldsLongerThanTheReleaseTime)
{
  struct Case {const char * name; CbfSafetyConfig config; CbfSnapshot start;};
  const auto left = fixture_config(freeze091650_config());
  const std::vector<Case> cases{
    {"two hands", fixture_config(), urdf_snapshot()},
    {"left hand", left, freeze094711_snapshot(left)},
  };
  for (const auto & item : cases) {
    SCOPED_TRACE(item.name);
    int verdicts = 0;
    int collision_releases = 0;
    for (unsigned seed = 1; seed <= 10; ++seed) {
      SCOPED_TRACE(seed);
      const auto stats = run_random_reaching(item.config, item.start, seed, 1500);
      std::printf("%s seed=%u frames=%d verdicts=%d releases=%d collision_releases=%d longest_unpublished_s=%.3f\n",
        item.name, seed, stats.frames, stats.verdicts, stats.releases, stats.collision_releases,
        stats.longest_unpublished_s);
      EXPECT_LE(stats.longest_unpublished_s, item.config.collision_release_after_s + 1e-9);
      verdicts += stats.verdicts;
      collision_releases += stats.collision_releases;
    }
    // The random motions must actually reach collision verdicts, or this proves nothing.
    EXPECT_GT(verdicts, 0);
  }
}

TEST(CbfSafety, WithTwoHandsAFirstFrameThumbVerdictNoLongerPublishesNothingForever)
{
  // Two installed hands, a random arm pose and thumb request (probe seed 11,
  // trial 5): the first frame of the epoch rejects the thumb sweep from the
  // reading. Nothing has been published, so the hold publishes nothing.
  const auto config = fixture_config();
  ASSERT_TRUE(config.hands_installed[0] && config.hands_installed[1]);
  auto snapshot = urdf_snapshot();
  const std::array<double, 14> arm{
    0.97098955171977375, 0.0015500553156590283, -1.0116668777822755, -0.86858823246654793,
    2.2962224066114643, -0.012511402474913114, -0.21896841497806308, -2.8793513228937471,
    -1.3056623927009205, -0.70820119348490729, -1.2248622998793119, -0.56394577834023996,
    -0.40764830845765165, -0.19043920759682131};
  const std::array<double, 4> thumb{0.26006678639704239, -0.0635755207687001, 0.34644515390078245, -0.97902161376583652};
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  for (std::size_t j = 0; j < 14; ++j) {snapshot.arm->position[j] = snapshot.requested.arm[j] = arm[j];}
  for (std::size_t a = 0; a < 4; ++a) {snapshot.requested.hand[slots[a]] = thumb[a];}
  CbfSafety safety(config);
  int empty_frames = 0;
  int released_frames = 0;
  int verified_frames = 0;
  for (int frame = 0; frame < 150; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.0 + config.period_s * frame;
    const auto decision = safety.filter(snapshot, now);
    if (frame == 0) {
      ASSERT_TRUE(decision.collision_refusal) << safety.diagnostics().reason;
      ASSERT_EQ(decision.output.kind, Output::Kind::kNone);
    }
    const auto out = safety.release_if_stuck(decision, snapshot, now);
    safety.committed(out, now);
    if (now - 1.0 + 1e-9 < 0.5) {
      EXPECT_EQ(out.output.kind, Output::Kind::kNone);
      ++empty_frames;
      continue;
    }
    // From 0.5 s every frame publishes a complete pair.
    ASSERT_EQ(out.output.kind, Output::Kind::kUpperBody) << safety.diagnostics().reason;
    out.released ? ++released_frames : ++verified_frames;
    // The robot covers half the way to what was published.
    for (std::size_t j = 0; j < 14; ++j) {
      snapshot.arm->position[j] += 0.5 * (out.output.arm[j] - snapshot.arm->position[j]);
    }
    for (std::size_t a = 0; a < 4; ++a) {
      snapshot.thumb->position[a] += 0.5 * (out.output.hand[slots[a]] - snapshot.thumb->position[a]);
    }
  }
  std::printf("two-hand first frame: empty=%d released=%d verified=%d\n", empty_frames, released_frames, verified_frames);
  EXPECT_EQ(empty_frames, 25);
  EXPECT_EQ(released_frames + verified_frames, 125);
  EXPECT_GT(released_frames, 0);
  EXPECT_EQ(safety.diagnostics().collision_release_episodes, 1u);
}

// --- Two hands configured, one of them possibly an empty flange ---------------

TEST(CbfSafety, AnInstalledSideReportedAsNoHandIsAPhantomHandNotAMismatch)
{
  const auto both = fixture_config();
  ASSERT_TRUE(both.hands_installed[0] && both.hands_installed[1]);
  // Type 0 on an installed side is an empty flange (the hand is not fitted yet).
  EXPECT_TRUE(both.matches_hand_types({1, 1}));
  EXPECT_TRUE(both.matches_hand_types({1, 0}));
  EXPECT_TRUE(both.matches_hand_types({0, 1}));
  EXPECT_TRUE(both.matches_hand_types({0, 0}));
}

TEST(CbfSafety, AHandTheGeometryWasNotBuiltForIsStillAMismatch)
{
  // Any hand other than the OmniHand (type 1) has no collision body here.
  const auto both = fixture_config();
  for (int other : {2, 3, 4, 5, 0xFF, -1}) {
    SCOPED_TRACE(other);
    EXPECT_FALSE(both.matches_hand_types({1, other}));
    EXPECT_FALSE(both.matches_hand_types({other, 1}));
    EXPECT_FALSE(both.matches_hand_types({0, other}));
  }
  EXPECT_FALSE(both.matches_hand_types({1}));
  EXPECT_FALSE(both.matches_hand_types({1, 1, 1}));
  EXPECT_FALSE(both.matches_hand_types({}));
  // Control: the same calls with the OmniHand on both sides match.
  EXPECT_TRUE(both.matches_hand_types({1, 1}));
}

TEST(CbfSafety, AHandOnASideConfiguredWithoutOneIsStillAMismatch)
{
  // The reverse of the phantom hand: the platform reports a hand where the
  // configuration has no collision body, so that hand would move unchecked.
  auto value = urdf_fixture_json();
  value["hands_installed"] = {true, false};
  const auto left_only = fixture_config(value);
  EXPECT_FALSE(left_only.matches_hand_types({1, 1}));
  EXPECT_FALSE(left_only.matches_hand_types({0, 1}));
  EXPECT_FALSE(left_only.matches_hand_types({1, 2}));
  // Control: none on that side matches.
  EXPECT_TRUE(left_only.matches_hand_types({1, 0}));
  value["hands_installed"] = {false, true};
  const auto right_only = fixture_config(value);
  EXPECT_FALSE(right_only.matches_hand_types({1, 1}));
  EXPECT_FALSE(right_only.matches_hand_types({1, 0}));
  EXPECT_TRUE(right_only.matches_hand_types({0, 1}));
  EXPECT_TRUE(right_only.matches_hand_types({0, 0}));
}

namespace
{

// Two hands configured; every thumb reading inside the model band and off the command.
CbfSnapshot two_hand_readings(const CbfSafetyConfig & config)
{
  auto snapshot = urdf_snapshot();
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    const auto & band = config.thumb_model_limits[axis];
    const double command = snapshot.requested.hand[slots[axis]];
    // A quarter of the way to the farther bound: inside the band, never the command.
    const double bound = band[1] - command > command - band[0] ? band[1] : band[0];
    snapshot.thumb->position[axis] = command + 0.25 * (bound - command);
  }
  return snapshot;
}

}  // namespace

// The request does not change here, so the published phantom thumb is the
// request on every frame; a changing request is held (see below).
TEST(CbfSafety, APhantomRightThumbIsThePublishedCommandAndNoFeedbackProblem)
{
  const auto config = fixture_config();
  CbfSafety safety(config);
  auto snapshot = two_hand_readings(config);
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  snapshot.hand_reported_absent = {false, true};
  // An empty flange has no readings; any that arrive are not this hand's.
  snapshot.thumb->position[2] = std::numeric_limits<double>::quiet_NaN();
  snapshot.thumb->position[3] = 5.0;
  for (int frame = 0; frame < 3; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.0 + 0.02 * frame;
    const auto decision = safety.filter(snapshot, now);
    ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
    const auto & d = safety.diagnostics();
    for (std::size_t axis = 0; axis < 2; ++axis) {
      EXPECT_DOUBLE_EQ(d.thumb_current_urdf[axis], snapshot.thumb->position[axis]) << axis;
      ASSERT_NE(snapshot.thumb->position[axis], snapshot.requested.hand[slots[axis]]);
    }
    for (std::size_t axis = 2; axis < 4; ++axis) {
      EXPECT_DOUBLE_EQ(d.thumb_current_urdf[axis], snapshot.requested.hand[slots[axis]]) << axis;
    }
    EXPECT_TRUE(d.thumb_feedback_issue.empty()) << d.thumb_feedback_issue;
    EXPECT_EQ(d.thumb_feedback_frames, 0u);
    EXPECT_FALSE(d.last_thumb_feedback_event.has_value());
    safety.committed(decision, now);
  }
  // Control: the same readings on a hand the platform reports are used, and flagged.
  CbfSafety reported(config);
  snapshot.hand_reported_absent = {false, false};
  const auto decision = reported.filter(snapshot, 1.0);
  ASSERT_TRUE(decision.accept_new) << reported.diagnostics().reason;
  EXPECT_EQ(reported.diagnostics().thumb_feedback_issue, "unavailable");
  EXPECT_EQ(reported.diagnostics().last_thumb_feedback_event->axis, 2);
  EXPECT_EQ(reported.diagnostics().thumb_feedback_frames, 1u);
}

TEST(CbfSafety, AFaultedThumbReplacesOnlyItsOwnSide)
{
  const auto config = fixture_config();
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  for (std::size_t faulted : {0u, 1u}) {
    SCOPED_TRACE(faulted);
    CbfSafety safety(config);
    auto snapshot = two_hand_readings(config);
    snapshot.thumb_faulted[faulted] = true;
    const auto decision = safety.filter(snapshot, 1.0);
    ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
    const auto & d = safety.diagnostics();
    for (std::size_t axis = 0; axis < 4; ++axis) {
      SCOPED_TRACE(axis);
      const double expected = axis / 2 == faulted ? snapshot.requested.hand[slots[axis]] : snapshot.thumb->position[axis];
      EXPECT_DOUBLE_EQ(d.thumb_current_urdf[axis], expected);
    }
    EXPECT_EQ(d.thumb_feedback_issue, "faulted");
    ASSERT_TRUE(d.last_thumb_feedback_event.has_value());
    EXPECT_EQ(d.last_thumb_feedback_event->axis, static_cast<int>(2 * faulted));
  }
}

TEST(CbfSafety, AMissingSideLeavesTheOtherSidesReadingInUse)
{
  // The node fills a side without readings with NaN; the other side still counts.
  const auto config = fixture_config();
  constexpr std::array<std::size_t, 4> slots{0, 1, 10, 11};
  for (std::size_t missing : {0u, 1u}) {
    SCOPED_TRACE(missing);
    CbfSafety safety(config);
    auto snapshot = two_hand_readings(config);
    snapshot.thumb->position[2 * missing] = std::numeric_limits<double>::quiet_NaN();
    snapshot.thumb->position[2 * missing + 1] = std::numeric_limits<double>::quiet_NaN();
    const auto decision = safety.filter(snapshot, 1.0);
    ASSERT_TRUE(decision.accept_new) << safety.diagnostics().reason;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      SCOPED_TRACE(axis);
      const double expected = axis / 2 == missing ? snapshot.requested.hand[slots[axis]] : snapshot.thumb->position[axis];
      EXPECT_DOUBLE_EQ(safety.diagnostics().thumb_current_urdf[axis], expected);
    }
    EXPECT_EQ(safety.diagnostics().thumb_feedback_issue, "unavailable");
  }
}

namespace
{

// Everything a left-only frame decides and counts, timing excluded.
std::string decision_record(const CbfDecision & decision, const CbfDiagnostics & d)
{
  std::string out;
  const auto add = [&out](double value) {
      char text[40];
      std::snprintf(text, sizeof(text), "%a,", value);
      out += text;
    };
  out += std::to_string(static_cast<int>(decision.output.kind)) + ":" + std::to_string(decision.accept_new) +
    std::to_string(decision.modified) + std::to_string(decision.geometry_modified) +
    std::to_string(decision.recovery_verified) + std::to_string(decision.hand_held) +
    std::to_string(decision.collision_refusal) + std::to_string(decision.released) + ":";
  for (double v : decision.output.arm) {add(v);}
  for (double v : decision.output.hand) {add(v);}
  out += "|" + d.reason + "|" + d.thumb_feedback_issue + "|";
  for (auto v : {d.calls, d.accepted, d.published, d.modified, d.geometry_modified, d.held, d.withheld,
      d.deadlines, d.invalid_input_frames, d.thumb_feedback_frames, d.released, d.release_episodes,
      d.collision_release_episodes, d.hand_budget_holds})
  {
    out += std::to_string(v) + ",";
  }
  for (double v : d.thumb_current_urdf) {add(v);}
  for (double v : d.thumb_target_urdf) {add(v);}
  add(d.distance_valid ? d.minimum_distance_m : -99.0);
  if (d.last_thumb_feedback_event) {
    out += d.last_thumb_feedback_event->issue + std::to_string(d.last_thumb_feedback_event->axis) +
      std::to_string(d.last_thumb_feedback_event->event);
  }
  return out;
}

}  // namespace

TEST(CbfSafety, RightSideFlagsCannotReachALeftOnlyConfiguration)
{
  // Left-only control for the per-side flags: on a side configured without a
  // hand they are inert, so the left-only behaviour is the one of e965d015.
  const auto config = fixture_config(historical_left_candidate_fixture_json());
  CbfSafety plain(config);
  CbfSafety flagged(config);
  auto base = historical_left_native_snapshot();
  for (int frame = 0; frame < 12; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.0 + 0.02 * frame;
    auto snapshot = base;
    snapshot.requested.arm[0] += 0.01 * frame;
    if (frame == 4) {snapshot.thumb_faulted[0] = true;}
    if (frame == 6) {snapshot.thumb->position[1] = std::numeric_limits<double>::quiet_NaN();}
    if (frame == 8) {snapshot.thumb.reset();}
    auto with_right = snapshot;
    with_right.thumb_faulted[1] = true;
    with_right.hand_reported_absent[1] = true;
    const auto a = plain.filter(snapshot, now);
    const auto b = flagged.filter(with_right, now);
    EXPECT_EQ(decision_record(a, plain.diagnostics()), decision_record(b, flagged.diagnostics()));
    plain.committed(a, now);
    flagged.committed(b, now);
  }
  EXPECT_GT(plain.diagnostics().thumb_feedback_frames, 0u);
}

TEST(CbfSafety, APhantomRightHandKeepsItsCollisionPairsInTheDecision)
{
  // Found offline (random right-arm poses, seed 3, trial 197): from natural
  // standing, a straight reach that only the right thumb's collision body
  // stops. Its path is clear for the arms and the left hand.
  const auto & standing = x2::cbf::generated::kNaturalStanding;
  std::array<double, 14> reach{};
  std::copy_n(standing.begin(), 7, reach.begin());
  const std::array<double, 7> right{
    0.66750223117580409, -0.35727957073251027, 0.49318808792707802, -1.1582442514935436,
    0.64058981282216676, -0.1618123346891257, -0.13997082739639233};
  std::copy(right.begin(), right.end(), reach.begin() + 7);
  const auto right_hand_pair = [](std::size_t index) {
      const auto pair = x2::cbf::kPrimitivePairs[index];
      const auto hand = [](x2::cbf::PrimitiveId id) {
          return id == x2::cbf::PrimitiveId::RightPalm || id == x2::cbf::PrimitiveId::RightThumb;
        };
      return hand(pair.first) || hand(pair.second);
    };
  const auto distances = [](const std::array<double, 14> & arm, std::array<bool, 2> hands) {
      x2::cbf::KinematicsInput input;
      input.q_arm = arm;
      input.q_thumb_urdf = x2::cbf::kRigidThumbBaselineUrdf;  // The commanded thumbs below.
      input.thumb_pose_valid = true;
      input.hands_installed = hands;
      return x2::cbf::evaluate_proxy_distances(input);
    };
  {
    const auto with_right = distances(reach, {true, true});
    ASSERT_LT(with_right.minimum_distance, -0.01);
    ASSERT_TRUE(right_hand_pair(with_right.minimum_pair_index));
    ASSERT_GT(distances(reach, {true, false}).minimum_distance, 0.0);
  }
  const auto walk = [&](const CbfSafetyConfig & config, bool phantom_right, double & worst_right_pair) {
      CbfSafety safety(config);
      auto snapshot = urdf_snapshot();
      snapshot.requested.arm.assign(reach.begin(), reach.end());
      if (phantom_right) {
        snapshot.hand_reported_absent = {false, true};
        snapshot.thumb->position[2] = snapshot.thumb->position[3] = std::numeric_limits<double>::quiet_NaN();
      } else {
        snapshot.thumb->position.resize(2);
      }
      std::array<double, 14> arm{};
      std::copy_n(standing.begin(), 14, arm.begin());
      worst_right_pair = std::numeric_limits<double>::infinity();
      for (int frame = 0; frame < 90; ++frame) {
        const double now = 1.0 + 0.02 * frame;
        snapshot.arm->position.assign(arm.begin(), arm.end());  // The arm tracks exactly.
        auto decision = safety.filter(snapshot, now);
        decision = safety.release_if_stuck(decision, snapshot, now);
        EXPECT_FALSE(decision.released) << frame << " " << safety.diagnostics().reason;
        safety.committed(decision, now);
        if (decision.output.kind != Output::Kind::kUpperBody) {continue;}
        std::copy_n(decision.output.arm.begin(), 14, arm.begin());
        const auto report = distances(arm, {true, true});
        for (std::size_t index = 0; index < report.distances.size(); ++index) {
          if (right_hand_pair(index)) {
            worst_right_pair = std::min(worst_right_pair, report.distances[index].signed_distance);
          }
        }
      }
      double error = 0.0;
      for (std::size_t j = 0; j < 14; ++j) {error = std::max(error, std::abs(arm[j] - reach[j]));}
      return error;
    };
  auto value = urdf_fixture_json();
  value["hands_installed"] = {true, false};
  double left_only_worst = 0.0;
  // Control: without the right hand's geometry the reach is published as requested.
  EXPECT_LT(walk(fixture_config(value), false, left_only_worst), 1e-9);
  EXPECT_LT(left_only_worst, 0.0);
  // Two hands configured, the right one an empty flange: stopped short, never inside.
  double phantom_worst = 0.0;
  EXPECT_GT(walk(fixture_config(), true, phantom_worst), 0.05);
  EXPECT_GE(phantom_worst, 0.0);
}

namespace
{

bool right_hand_pair(std::size_t index)
{
  const auto pair = x2::cbf::kPrimitivePairs[index];
  const auto hand = [](x2::cbf::PrimitiveId id) {
      return id == x2::cbf::PrimitiveId::RightPalm || id == x2::cbf::PrimitiveId::RightThumb;
    };
  return hand(pair.first) || hand(pair.second);
}

// The closest pair involving the right palm or thumb, with both hands in the geometry.
double right_hand_distance(const std::vector<double> & arm, const std::array<double, 6> & thumbs)
{
  x2::cbf::KinematicsInput input;
  std::copy_n(arm.begin(), 14, input.q_arm.begin());
  input.q_thumb_urdf = thumbs;
  input.thumb_pose_valid = true;
  input.hands_installed = {true, true};
  const auto report = x2::cbf::evaluate_proxy_distances(input);
  double out = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < report.distances.size(); ++index) {
    if (right_hand_pair(index)) {out = std::min(out, report.distances[index].signed_distance);}
  }
  return out;
}

}  // namespace

TEST(CbfSafety, APhantomThumbCommandNeverHoldsTheRealHandsGesture)
{
  // Found offline (random right-arm poses, seed 3, trial 270): with the left arm
  // standing, a right thumb angle that puts the right thumb into the body while
  // the baseline thumbs are clear. The left thumb changes gesture in the same frame.
  const auto & standing = x2::cbf::generated::kNaturalStanding;
  std::vector<double> arm(standing.begin(), standing.end());
  const std::array<double, 7> right{
    1.2195245293972587, -0.35745308639855455, 0.77877897171611288, -1.3525817857779296,
    -1.474963573344094, -0.19896486789711637, -0.4473240963224453};
  std::copy(right.begin(), right.end(), arm.begin() + 7);
  const std::array<double, 2> right_thumb{0.44924661356620199, 0.0078834578599495941};
  const std::array<double, 2> left_gesture{0.66863211354711227, -0.35277882194107857};
  {
    auto thumbs = x2::cbf::kRigidThumbBaselineUrdf;
    ASSERT_GT(right_hand_distance(arm, thumbs), 0.0);
    thumbs[3] = right_thumb[0];
    thumbs[4] = right_thumb[1];
    ASSERT_LT(right_hand_distance(arm, thumbs), -0.02);
  }
  const auto config = fixture_config();
  // Frame 0 publishes both baseline thumbs; frame 1 asks for the left gesture,
  // and the right thumb either stays or moves into the body.
  const auto second_frame = [&](bool phantom_right, bool move_right) {
      CbfSafety safety(config);
      auto snapshot = urdf_snapshot();
      snapshot.requested.arm = arm;
      snapshot.arm = Measured{arm, 0.0};
      if (phantom_right) {
        snapshot.hand_reported_absent = {false, true};
        snapshot.thumb->position[2] = snapshot.thumb->position[3] = std::numeric_limits<double>::quiet_NaN();
      }
      const auto first = safety.filter(snapshot, 1.0);
      EXPECT_TRUE(first.accept_new) << safety.diagnostics().reason;
      safety.committed(first, 1.0);
      snapshot.requested.sequence = 8;
      snapshot.requested.hand[0] = left_gesture[0];
      snapshot.requested.hand[1] = left_gesture[1];
      if (move_right) {
        snapshot.requested.hand[10] = right_thumb[0];
        snapshot.requested.hand[11] = right_thumb[1];
      }
      return std::make_pair(safety.filter(snapshot, 1.02), first.output);
    };
  // Control: on a reported right hand that thumb angle is real, and holds the whole gesture.
  {
    const auto [held, first] = second_frame(false, true);
    ASSERT_TRUE(held.accept_new);
    EXPECT_TRUE(held.hand_held);
    EXPECT_EQ(held.output.hand, first.hand);
    const auto [still, unused] = second_frame(false, false);
    ASSERT_TRUE(still.accept_new);
    EXPECT_FALSE(still.hand_held);
    EXPECT_EQ(still.output.hand[0], left_gesture[0]);
  }
  // A phantom right thumb stays where it was published: the left gesture passes
  // in that frame, bit for bit as when the right thumb command does not change.
  const auto [moved, first] = second_frame(true, true);
  const auto [still, unused] = second_frame(true, false);
  ASSERT_TRUE(moved.accept_new);
  EXPECT_FALSE(moved.hand_held);
  EXPECT_EQ(moved.output.hand[0], left_gesture[0]);
  EXPECT_EQ(moved.output.hand[1], left_gesture[1]);
  EXPECT_EQ(moved.output.hand[10], first.hand[10]);
  EXPECT_EQ(moved.output.hand[11], first.hand[11]);
  EXPECT_EQ(moved.output.arm, still.output.arm);
  EXPECT_EQ(moved.output.hand, still.output.hand);
  EXPECT_EQ(moved.recovery_verified, still.recovery_verified);
  EXPECT_EQ(moved.geometry_modified, still.geometry_modified);
}

TEST(CbfSafety, APhantomRightHandChangesTheLeftHandAndArmsOnlyWhereItsGeometryIsNear)
{
  // Two hands configured with the right one an empty flange, against the
  // left-only configuration with the platform reporting {1, 0}. The latter is
  // the e965d015 behaviour: an offline driver linking both builds printed every
  // decision of 600 frames bit for bit identical (RightSideFlags... above keeps
  // the per-side flags out of it in this suite).
  //
  // Same requests, right thumb command unchanged. The two can only differ
  // where the phantom right palm or thumb is in the geometry: a pair involving
  // it within the activation distance (0.08 m, plus 0.02 m for the path
  // between the checked poses) of the body, the thighs or the left arm. Far
  // from them the outputs are bit for bit the same. Near them the phantom hand
  // may stop or bend the arm the left-only filter lets through; that is what
  // an empty flange costs (geometry_modified 389 against 1107 frames in the
  // 3000-frame timing input), and the reason the reverse is unsafe.
  //
  // After a frame that differs the two robots are put back together (a new
  // epoch, both at the phantom run's arm), so every compared frame starts from
  // the same state.
  auto left_value = urdf_fixture_json();
  left_value["hands_installed"] = {true, false};
  const auto left_config = fixture_config(left_value);
  const auto phantom_config = fixture_config();
  ASSERT_TRUE(left_config.matches_hand_types({1, 0}));
  ASSERT_TRUE(phantom_config.matches_hand_types({1, 0}));
  CbfSafety left_only(left_config);
  CbfSafety phantom(phantom_config);
  const auto & standing = x2::cbf::generated::kNaturalStanding;
  const std::array<double, 7> reach{
    0.66750223117580409, -0.35727957073251027, 0.49318808792707802, -1.1582442514935436,
    0.64058981282216676, -0.1618123346891257, -0.13997082739639233};
  std::uint64_t state = 88172645463325252ull;
  const auto uniform = [&state](double lo, double hi) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      return lo + (hi - lo) * static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    };
  const double activation = x2::cbf::SafetyFilterConfig{}.activation_distance_m;
  auto snapshot = urdf_snapshot();
  std::vector<double> requested(standing.begin(), standing.end());
  std::vector<double> goal = requested, measured = requested;
  std::array<double, 2> left_thumb_goal{snapshot.requested.hand[0], snapshot.requested.hand[1]};
  std::array<double, 2> left_thumb{left_thumb_goal};
  std::uint64_t epoch = 1;
  int same_far = 0, same_near = 0, differed = 0;
  double farthest_difference = -std::numeric_limits<double>::infinity();
  for (int frame = 0; frame < 1500; ++frame) {
    SCOPED_TRACE(frame);
    const double now = 1.0 + 0.02 * frame;
    if (frame % 40 == 0) {
      if (frame < 500) {  // Left arm reaching anywhere, right arm standing.
        for (std::size_t j = 0; j < 7; ++j) {goal[j] = uniform(left_config.lower[j], left_config.upper[j]);}
        std::copy_n(standing.begin() + 7, 7, goal.begin() + 7);
      } else if (frame < 1000) {  // Right arm towards the reach only its hand stops.
        std::copy_n(standing.begin(), 7, goal.begin());
        for (std::size_t j = 0; j < 7; ++j) {goal[7 + j] = reach[j] + uniform(-0.2, 0.2);}
      } else {  // Both arms reaching anywhere.
        for (std::size_t j = 0; j < 14; ++j) {goal[j] = uniform(left_config.lower[j], left_config.upper[j]);}
      }
    }
    if (frame % 7 == 0) {
      for (std::size_t a = 0; a < 2; ++a) {
        left_thumb_goal[a] = uniform(left_config.thumb_model_limits[a][0], left_config.thumb_model_limits[a][1]);
      }
    }
    for (std::size_t j = 0; j < 14; ++j) {requested[j] += std::clamp(goal[j] - requested[j], -0.1, 0.1);}
    snapshot.epoch = epoch;
    snapshot.requested.sequence = static_cast<std::uint32_t>(frame + 10);
    snapshot.requested.arm = requested;
    snapshot.requested.hand[0] = left_thumb_goal[0];
    snapshot.requested.hand[1] = left_thumb_goal[1];
    snapshot.arm = Measured{measured, 0.0};
    auto left_snapshot = snapshot;
    left_snapshot.thumb->position = {left_thumb[0], left_thumb[1]};
    auto phantom_snapshot = snapshot;
    phantom_snapshot.thumb->position = {left_thumb[0], left_thumb[1],
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    phantom_snapshot.hand_reported_absent = {false, true};
    const auto a = left_only.filter(left_snapshot, now);
    const auto b = phantom.filter(phantom_snapshot, now);
    left_only.committed(a, now);
    phantom.committed(b, now);
    const auto left_hand = [](const CbfDecision & d) {
        const auto & hand = d.output.hand;  // Empty when nothing is published.
        return std::vector<double>(hand.begin(), hand.begin() + std::min<std::size_t>(10, hand.size()));
      };
    const bool same = a.accept_new == b.accept_new && a.output.kind == b.output.kind &&
      a.output.arm == b.output.arm && left_hand(a) == left_hand(b);
    auto thumbs = x2::cbf::kRigidThumbBaselineUrdf;
    thumbs[0] = snapshot.requested.hand[0];
    thumbs[1] = snapshot.requested.hand[1];
    double near = right_hand_distance(measured, thumbs);
    near = std::min(near, right_hand_distance(requested, thumbs));
    for (const auto * d : {&a, &b}) {
      if (d->output.kind == Output::Kind::kUpperBody) {near = std::min(near, right_hand_distance(d->output.arm, thumbs));}
    }
    const bool is_near = near < activation + 0.02;
    if (same) {
      (is_near ? same_near : same_far) += 1;
    } else {
      ++differed;
      farthest_difference = std::max(farthest_difference, near);
      EXPECT_TRUE(is_near) << "outputs differ with the phantom hand " << near << " m away";
    }
    const auto & lead = same ? a : b;
    if (lead.output.kind == Output::Kind::kUpperBody) {
      for (std::size_t j = 0; j < 14; ++j) {measured[j] += 0.5 * (lead.output.arm[j] - measured[j]);}
      for (std::size_t k = 0; k < 2; ++k) {left_thumb[k] += 0.5 * (lead.output.hand[k] - left_thumb[k]);}
    }
    if (!same) {++epoch;}  // Both start the next frame from the same state again.
  }
  std::printf("same_far=%d same_near=%d differed=%d farthest_difference_m=%.4f\n", same_far, same_near, differed,
    farthest_difference);
  // The input must reach both cases, or this proves nothing.
  EXPECT_GT(same_far, 100);
  EXPECT_GT(differed, 10);
}
