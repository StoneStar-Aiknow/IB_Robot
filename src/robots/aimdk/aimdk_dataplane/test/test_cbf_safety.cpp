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
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "cbf/generated_constants.hpp"
#include "cbf/rigid_thumb.hpp"
#include "cbf/safety_filter.hpp"
#include "nlohmann/json.hpp"

namespace
{

using aimdk_dataplane::CbfDecision;
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
  snapshot.requested.hand[0] = x2::cbf::kRigidThumbLimits[0][1] + 0.01;
  const auto rejected = safety.filter(snapshot, 1.02);

  EXPECT_FALSE(rejected.accept_new);
  EXPECT_EQ(rejected.output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.diagnostics().accepted, 0u);
  EXPECT_EQ(safety.diagnostics().published, 0u);
}

TEST(CbfSafety, InvalidThumbAndNanNominalHoldTheCommittedArmAndHandTogether)
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
      // Existing phase-C shoulder request plus the original invalid-thumb case.
      snapshot.requested.arm[0] += 0.04;
      snapshot.requested.hand[0] = x2::cbf::kRigidThumbLimits[0][1] + 0.01;
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

TEST(CbfSafety, StaleArmOrThumbFeedbackKeepsTheCommittedPairWithoutAcceptingNewTargets)
{
  CbfSafety safety(fixture_config());
  const auto initial = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(initial.accept_new);
  safety.committed(initial, 1.0);

  for (int stale_group = 0; stale_group < 2; ++stale_group) {
    SCOPED_TRACE(stale_group);
    auto snapshot = urdf_snapshot();
    auto & feedback = stale_group == 0 ? snapshot.arm : snapshot.thumb;
    feedback->age_s = safety.config().feedback_max_age_s + 0.001;
    const auto decision = safety.filter(snapshot, 1.02);
    expect_held_pair(decision, initial.output);
    EXPECT_EQ(safety.diagnostics().reason, "feedback_unavailable");
  }
  EXPECT_EQ(safety.diagnostics().published, 1u);
}

TEST(CbfSafety, FreshFeedbackWithExcessiveArmThumbSkewIsRejected)
{
  CbfSafety safety(fixture_config());
  auto snapshot = urdf_snapshot();
  snapshot.arm->age_s = 0.001;
  snapshot.thumb->age_s = 0.06;
  EXPECT_DOUBLE_EQ(safety.config().state_skew_max_s, 0.04);
  ASSERT_LT(snapshot.arm->age_s, safety.config().feedback_max_age_s);
  ASSERT_LT(snapshot.thumb->age_s, safety.config().feedback_max_age_s);
  ASSERT_GT(std::abs(snapshot.arm->age_s - snapshot.thumb->age_s), safety.config().state_skew_max_s);

  const auto rejected = safety.filter(snapshot, 1.0);
  EXPECT_FALSE(rejected.accept_new);
  EXPECT_EQ(rejected.output.kind, Output::Kind::kNone);
  EXPECT_EQ(safety.diagnostics().reason, "feedback_skew");
}

TEST(CbfSafety, ANewEpochCannotReuseThePreviousCommittedPair)
{
  CbfSafety safety(fixture_config());
  const auto initial = safety.filter(urdf_snapshot(), 1.0);
  ASSERT_TRUE(initial.accept_new);
  safety.committed(initial, 1.0);

  auto snapshot = urdf_snapshot();
  snapshot.epoch = 2;
  snapshot.requested.hand[0] = x2::cbf::kRigidThumbLimits[0][1] + 0.01;
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

TEST(CbfSafety, LeftCandidateAcceptsSafeCompactFeedbackAndHoldsTheCommittedPairForAnInvalidTarget)
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

  auto invalid_target = snapshot;
  invalid_target.requested.arm[0] += 0.04;
  invalid_target.requested.hand[0] = safety.config().native_upper[0] + 0.01;
  const auto held = safety.filter(invalid_target, 1.02);
  expect_held_pair(held, decision.output);
  EXPECT_EQ(safety.diagnostics().reason, "unverified_mapping/invalid_command");
  ASSERT_EQ(held.output.kind, Output::Kind::kUpperBody);
  safety.committed(held, 1.02);
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().held, 1u);
  EXPECT_EQ(safety.diagnostics().published, 2u);
  EXPECT_FALSE(safety.config().mapping_verified);
}

TEST(CbfSafety, LeftCandidateHoldsTheCommittedPairForOutOfRangeNanAndStaleFeedback)
{
  CbfSafety safety(fixture_config(historical_left_candidate_fixture_json()));
  const auto first = safety.filter(historical_left_native_snapshot(), 1.0);
  ASSERT_TRUE(first.accept_new);
  safety.committed(first, 1.0);
  for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
    SCOPED_TRACE(invalid_case);
    auto snapshot = historical_left_native_snapshot();
    const char * reason;
    if (invalid_case == 0) {
      snapshot.thumb->position[0] = safety.config().native_lower[0] - 0.01;
      reason = "unverified_mapping/native_feedback_out_of_range";
    } else if (invalid_case == 1) {
      snapshot.thumb->position[0] = std::numeric_limits<double>::quiet_NaN();
      reason = "unverified_mapping/feedback_unavailable";
    } else {
      snapshot.thumb->age_s = safety.config().feedback_max_age_s + 0.001;
      reason = "unverified_mapping/feedback_unavailable";
    }
    const auto rejected = safety.filter(snapshot, 1.02 + 0.02 * invalid_case);
    expect_held_pair(rejected, first.output);
    EXPECT_EQ(safety.diagnostics().reason, reason);
    EXPECT_FALSE(safety.config().mapping_verified);
  }
  EXPECT_EQ(safety.diagnostics().accepted, 1u);
  EXPECT_EQ(safety.diagnostics().published, 1u);
}

TEST(CbfSafety, ExplicitLeftHandOnlyUsesTwoMeasuredRootsAndPreservesHand20)
{
  auto value = urdf_fixture_json();
  value["hands_installed"] = {true, false};
  const auto config = fixture_config(value);
  EXPECT_TRUE(config.matches_hand_types({1, 0}));
  EXPECT_FALSE(config.matches_hand_types({0, 0}));
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
  expect_held_pair(lost_left, decision.output);
  EXPECT_EQ(safety.diagnostics().reason, "feedback_unavailable");
}

TEST(CbfSafety, AnInstalledHandCannotSilentlyDisappearOrChangeHardware)
{
  CbfSafety both(fixture_config());
  auto snapshot = urdf_snapshot();
  snapshot.thumb->position.resize(2);
  const auto missing = both.filter(snapshot, 1.0);
  EXPECT_EQ(missing.output.kind, Output::Kind::kNone);
  EXPECT_EQ(both.diagnostics().reason, "feedback_unavailable");

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

TEST(CbfSafety, PublicationFeedbackGateRejectsNewInvalidSamplesButAcceptsNormalUpdates)
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
  EXPECT_EQ(config.feedback_error(latest), "feedback_unavailable");
  latest.thumb.reset();
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

TEST(CbfSafety, ThumbFeedbackAllowanceDoesNotRelaxTargetLimitsOrAcceptExcessiveFeedback)
{
  auto value = historical_left_candidate_fixture_json();
  CbfSafety strict(fixture_config(value));
  constexpr double one_degree = 0.017453292519943295;
  value["feedback_limit_tolerance_rad"] = one_degree;
  CbfSafety observation_allowance(fixture_config(value));
  const double native_upper = strict.config().native_upper[0];
  auto within_allowance = historical_left_native_snapshot();
  ASSERT_EQ(within_allowance.thumb->position.size(), 2u);
  within_allowance.thumb->position[0] = native_upper + 0.0006;
  EXPECT_EQ(strict.config().feedback_error(within_allowance), "native_feedback_out_of_range");
  const auto strict_rejection = strict.filter(within_allowance, 1.0);
  EXPECT_FALSE(strict_rejection.accept_new);
  EXPECT_EQ(strict_rejection.output.kind, Output::Kind::kNone);
  EXPECT_TRUE(observation_allowance.config().feedback_error(within_allowance).empty());

  const auto numerical = observation_allowance.filter(within_allowance, 1.0);
  (void)numerical;  // Geometry may reject; an observation allowance is not a collision-safety pass.
  EXPECT_EQ(observation_allowance.diagnostics().invalid_input_frames, 0u);
  EXPECT_EQ(observation_allowance.diagnostics().reason.find("INVALID_INPUT_FALLBACK"), std::string::npos);
  EXPECT_NE(observation_allowance.diagnostics().reason, "unverified_mapping/native_feedback_out_of_range");
  ASSERT_TRUE(observation_allowance.diagnostics().thumb_angles_valid);
  const double raw_model_current = observation_allowance.config().feedback_scale[0] *
    within_allowance.thumb->position[0] + observation_allowance.config().feedback_offset[0];
  ASSERT_LT(raw_model_current, observation_allowance.config().thumb_model_limits[0][0]);
  EXPECT_DOUBLE_EQ(observation_allowance.diagnostics().thumb_current_urdf[0], raw_model_current);

  auto beyond_allowance = within_allowance;
  beyond_allowance.thumb->position[0] = native_upper + one_degree + 0.0006;
  EXPECT_EQ(observation_allowance.config().feedback_error(beyond_allowance), "native_feedback_out_of_range");
  const auto feedback_rejection = observation_allowance.filter(beyond_allowance, 1.02);
  EXPECT_FALSE(feedback_rejection.accept_new);
  EXPECT_EQ(feedback_rejection.output.kind, Output::Kind::kNone);
  EXPECT_EQ(observation_allowance.diagnostics().reason, "unverified_mapping/native_feedback_out_of_range");

  // No result above was committed: a target outside the unchanged command
  // range cannot authorize a new frame or invent a fallback pair.
  auto invalid_target = historical_left_native_snapshot();
  invalid_target.requested.hand[0] = native_upper + 0.0006;
  ASSERT_TRUE(observation_allowance.config().feedback_error(invalid_target).empty());
  const auto target_rejection = observation_allowance.filter(invalid_target, 1.04);
  EXPECT_FALSE(target_rejection.accept_new);
  EXPECT_EQ(target_rejection.output.kind, Output::Kind::kNone);
  EXPECT_EQ(observation_allowance.diagnostics().reason, "unverified_mapping/invalid_command");
  EXPECT_FALSE(observation_allowance.config().mapping_verified);
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
  input.current_thumb_joint_limits_urdf = input.thumb_joint_limits_urdf;
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const auto slot = x2::cbf::kRigidThumbIndices[axis];
    input.q_thumb_current_urdf[slot] = config.feedback_scale[axis] * snapshot.thumb->position[axis] +
      config.feedback_offset[axis];
    input.q_thumb_urdf[slot] = config.command_scale[axis] * snapshot.requested.hand[axis] + config.command_offset[axis];
    const double allowance = std::abs(config.feedback_scale[axis]) * config.feedback_limit_tolerance_rad;
    (*input.current_thumb_joint_limits_urdf)[axis][0] -= allowance;
    (*input.current_thumb_joint_limits_urdf)[axis][1] += allowance;
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
      bad.thumb->age_s = config.state_skew_max_s + 0.001;
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
  fresh.thumb.reset();
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
    if (frame == 1) {supplied.thumb->age_s = config.feedback_max_age_s + 0.001;}
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
