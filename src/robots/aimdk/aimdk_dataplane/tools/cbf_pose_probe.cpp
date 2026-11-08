// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// Link with aimdk_dataplane_core. This standalone numerical query has no ROS,
// SDK, publisher, actuator, or robot connection. It never authorizes motion.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "cbf/kinematics.hpp"
#include "cbf/proxy_model.hpp"
#include "cbf/safety_filter.hpp"
#include "nlohmann/json.hpp"

namespace
{

using nlohmann::json;
constexpr std::array<std::size_t, 4> kNativeRootSlots{0, 1, 10, 11};
constexpr std::array<double, 4> kCandidateScale{-1.0, 1.0, 1.0, -1.0};
constexpr std::array<double, 4> kCandidateOffset{0.0, -1.6424, -0.576, -1.6424};
constexpr std::array<double, 4> kNativeLower{-1.117, -0.05, -0.03, -1.64};
constexpr std::array<double, 4> kNativeUpper{0.03, 1.64, 0.87, 0.05};

struct Pose
{
  std::array<double, 14> arm{};
  std::array<double, 20> hand{};
  std::array<bool, 2> hands_installed{true, true};
};

json scope()
{
  return {
    {"probe", "static_frozen_proxy_numeric_query_only"},
    {"verified", false},
    {"candidate_only", true},
    {"robot_motion", false},
    {"physical_validation_passed", false},
    {"calibration_passed", false},
    {"warning", "Unverified candidate mapping. Numerical acceptance is not physical safety or calibration approval."},
  };
}

Pose default_pose()
{
  Pose pose;
  // aimdk_x2_magicmirror.yaml reset_pose / smplx2robot.postprocess.BASE_POSE_RAD.
  pose.arm = {
    0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0,
    0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0,
  };
  // mirror_session/hand_overlay.py: HAND_NEUTRAL = [P * LEFT_FIST,
  // P * LEFT_FIST * MIRROR]. The default result lies within its native bounds.
  constexpr std::array<double, 10> fist{-0.5, 1.0, -0.75, 0.0, 1.57, 1.57, 0.0, 1.57, 0.0, 1.57};
  constexpr std::array<double, 10> mirror{-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
  constexpr double curl = 0.33;
  constexpr double thumb = (curl - 0.25) / 0.65;
  constexpr std::array<double, 10> weights{thumb, thumb, thumb, 0.0, curl, curl, 0.0, curl, 0.0, curl};
  for (std::size_t joint = 0; joint < fist.size(); ++joint) {
    pose.hand[joint] = weights[joint] * fist[joint];
    pose.hand[joint + fist.size()] = pose.hand[joint] * mirror[joint];
  }
  return pose;
}

template<std::size_t N>
std::array<double, N> input_array(const json & document, const char * key)
{
  const auto & values = document.at(key);
  if (!values.is_array() || values.size() != N) {
    throw std::invalid_argument(std::string(key) + " must be an array of " + std::to_string(N) + " finite numbers");
  }
  std::array<double, N> result{};
  for (std::size_t index = 0; index < N; ++index) {
    if (!values[index].is_number()) {
      throw std::invalid_argument(std::string(key) + " contains a nonnumeric value");
    }
    result[index] = values[index].get<double>();
    if (!std::isfinite(result[index])) {
      throw std::invalid_argument(std::string(key) + " contains a nonfinite value");
    }
  }
  return result;
}

Pose load_pose(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream) {throw std::invalid_argument("cannot open input JSON file");}
  const auto document = json::parse(stream);
  if (!document.is_object()) {throw std::invalid_argument("input JSON must be an object with arm14 and hand20");}
  Pose pose;
  pose.arm = input_array<14>(document, "arm14");
  if (document.contains("left_hand10")) {
    if (document.contains("hand20")) {throw std::invalid_argument("use left_hand10 or hand20, not both");}
    pose.hands_installed = {true, false};
    const auto left = input_array<10>(document, "left_hand10");
    std::copy(left.begin(), left.end(), pose.hand.begin());
  } else {
    pose.hand = input_array<20>(document, "hand20");
  }
  return pose;
}

aimdk_dataplane::CbfSafetyConfig candidate_config(std::array<bool, 2> hands_installed = {true, true})
{
  const json value = {
    {"enabled", true},
    {"mode", "shadow"},
    {"protected_mirror_mode", "policy_stream"},
    {"deadline_us", 100000.0},
    {"hands_installed", hands_installed},
    {"thumb_mapping", {
      {"verified", false},
      {"command_scale", kCandidateScale},
      {"command_offset", kCandidateOffset},
      {"feedback_scale", kCandidateScale},
      {"feedback_offset", kCandidateOffset},
      {"native_lower", kNativeLower},
      {"native_upper", kNativeUpper},
    }},
  };
  return aimdk_dataplane::CbfSafetyConfig::from_json(value.dump(), 50.0);
}

json pair_description(std::size_t index)
{
  if (index >= x2::cbf::kPrimitivePairs.size()) {return nullptr;}
  const auto pair = x2::cbf::kPrimitivePairs[index];
  return {
    {"index", index},
    {"first", x2::cbf::primitive_name(pair.first)},
    {"second", x2::cbf::primitive_name(pair.second)},
  };
}

json probe(const Pose & pose, const std::string & source)
{
  const auto mapping = candidate_config(pose.hands_installed);
  if (mapping.mapping_verified || !mapping.mapping_available || !mapping.native_limits_available) {
    throw std::logic_error("probe requires available candidate mapping and bounds, with verified=false");
  }
  x2::cbf::SafetyFilterInput input;
  input.q_current = pose.arm;
  input.q_nominal = pose.arm;
  input.q_thumb_urdf = x2::cbf::kRigidThumbBaselineUrdf;
  input.hands_installed = pose.hands_installed;
  input.thumb_current_valid = true;
  input.thumb_pose_valid = true;  // Numerical coordinates available; NOT SDK calibration verification.
  input.thumb_joint_limits_urdf = mapping.thumb_model_limits;
  std::array<double, 4> native_roots{};
  std::array<double, 4> model_roots{};
  bool native_roots_within_limits = true;
  for (std::size_t axis = 0; axis < native_roots.size(); ++axis) {
    if (!pose.hands_installed[axis / 2]) {
      input.thumb_joint_limits_urdf[axis] = x2::cbf::kRigidThumbLimits[axis];
      continue;
    }
    native_roots[axis] = pose.hand[kNativeRootSlots[axis]];
    model_roots[axis] = mapping.command_scale[axis] * native_roots[axis] + mapping.command_offset[axis];
    input.q_thumb_urdf[x2::cbf::kRigidThumbIndices[axis]] = model_roots[axis];
    native_roots_within_limits = native_roots_within_limits &&
      native_roots[axis] >= mapping.native_lower[axis] && native_roots[axis] <= mapping.native_upper[axis];
  }
  input.q_thumb_current_urdf = input.q_thumb_urdf;

  x2::cbf::SafetyFilterConfig config;
  config.deadline_us = 100000.0;  // Numerical probe budget, not a performance claim.
  config.max_step_rad = 0.0;  // Query this pose; do not seek a different arm pose.
  config.path_check_substeps = 64;
  const auto result = x2::cbf::filter_arm_target(input, config);  // Exactly one filter call.

  auto report = scope();
  report["input_source"] = source;
  report["arm14"] = pose.arm;
  report["hand20"] = pose.hand;
  report["hands_installed"] = pose.hands_installed;
  report["root_order"] = {"L_roll", "L_abad", "R_roll", "R_abad"};
  report["native_root_slots"] = kNativeRootSlots;
  report["angle_units"] = "rad";
  report["native_roots"] = native_roots;
  report["model_roots"] = model_roots;
  report["model_thumb6"] = input.q_thumb_urdf;
  for (std::size_t side = 0; side < 2; ++side) {
    if (pose.hands_installed[side]) {continue;}
    for (std::size_t slot = 10 * side; slot < 10 * (side + 1); ++slot) {report["hand20"][slot] = nullptr;}
    for (std::size_t axis = 2 * side; axis < 2 * (side + 1); ++axis) {
      report["native_roots"][axis] = nullptr;
      report["model_roots"][axis] = nullptr;
    }
    for (std::size_t axis = 3 * side; axis < 3 * (side + 1); ++axis) {report["model_thumb6"][axis] = nullptr;}
  }
  report["native_roots_within_limits"] = native_roots_within_limits;
  report["model_roots_within_limits"] =
    x2::cbf::valid_rigid_thumb_pose(input.q_thumb_urdf, input.thumb_joint_limits_urdf);
  report["mapping"] = {
    {"verified", mapping.mapping_verified}, {"candidate_only", true},
    {"scale", mapping.command_scale}, {"offset_rad", mapping.command_offset},
    {"native_lower", mapping.native_lower}, {"native_upper", mapping.native_upper},
    {"thumb_model_limits", input.thumb_joint_limits_urdf},
    {"offset_evidence", {"fk_inferred", "fk_inferred", "hypothesis", "hypothesis"}},
  };
  report["status"] = x2::cbf::filter_status_name(result.status);
  report["nonlinear_verified"] = result.nonlinear_verified;
  report["target_accepted"] = result.thumb_target_accepted;
  report["deadline_missed"] = result.deadline_missed;
  report["deadline_us"] = config.deadline_us;
  report["elapsed_us"] = result.elapsed_us;
  report["performance_benchmark"] = false;
  report["returned_arm14"] = result.q_safe;
  report["arm_unchanged"] = result.q_safe == pose.arm;

  // Invalid inputs leave the kernel's numeric fields at defaults; do not print
  // those zero defaults as measured distances or an evaluated pair index.
  const bool evaluated = result.status != x2::cbf::FilterStatus::InvalidInputFallback;
  report["kernel_metrics_valid"] = evaluated;
  report["distance_units"] = "m";
  report["min_distance"] = evaluated ? json(result.input_min_distance_m) : json(nullptr);
  report["min_safety_margin"] = evaluated ? json(result.input_min_safety_margin_m) : json(nullptr);
  report["minimum_margin_pair"] = evaluated ? pair_description(result.input_min_pair) : json(nullptr);
  report["nearest_pair_index"] = nullptr;
  report["nearest_pair"] = nullptr;
  if (evaluated) {
    // The kernel's input_min_pair identifies the minimum SAFETY MARGIN, which
    // need not be the minimum raw distance. Ask the existing geometry API for
    // the actual nearest-distance pair, without another filter invocation.
    x2::cbf::KinematicsInput geometry;
    geometry.q_arm = input.q_current;
    geometry.q_thumb_urdf = input.q_thumb_current_urdf;
    geometry.thumb_pose_valid = true;
    geometry.hands_installed = pose.hands_installed;
    const auto kinematics = x2::cbf::forward_kinematics(geometry);
    if (kinematics.valid) {
      const auto distances = x2::cbf::evaluate_proxy_distances(kinematics);
      const bool distances_valid = std::all_of(
        distances.distances.begin(), distances.distances.end(),
        [](const x2::cbf::DistanceResult & distance) {
          return distance.valid && std::isfinite(distance.signed_distance);
        });
      if (distances_valid && std::isfinite(distances.minimum_distance)) {
        report["nearest_pair_index"] = distances.minimum_pair_index;
        report["nearest_pair"] = pair_description(distances.minimum_pair_index);
      }
    }
  }
  report["query_completed"] = true;
  return report;
}

// This replay is in frozen URDF coordinates only. Its identity mapping marked
// verified=true is a TEST FIXTURE, never evidence about a real SDK mapping.
json replay(std::size_t frames, bool left_hand_only = false)
{
  using aimdk_dataplane::CbfSafety;
  using aimdk_dataplane::CbfSnapshot;
  using aimdk_dataplane::Measured;
  using aimdk_dataplane::Output;
  const json settings = {
    {"enabled", true}, {"mode", "enforce"}, {"protected_mirror_mode", "policy_stream"},
    {"deadline_us", 10000.0}, {"deadline_guard_us", 500.0},
    {"hands_installed", {true, !left_hand_only}},
    {"thumb_mapping", {
      {"verified", true}, {"command_scale", {1.0, 1.0, 1.0, 1.0}},
      {"command_offset", {0.0, 0.0, 0.0, 0.0}}, {"feedback_scale", {1.0, 1.0, 1.0, 1.0}},
      {"feedback_offset", {0.0, 0.0, 0.0, 0.0}},
    }},
  };
  const auto config = aimdk_dataplane::CbfSafetyConfig::from_json(settings.dump(), 50.0);
  CbfSafety safety(config);
  Output actual;
  actual.kind = Output::Kind::kUpperBody;
  actual.hand_sub_mode = aimdk_dataplane::kHandSubModeDexterousJoint;
  actual.head = {0.0, 0.0};
  const auto & standing = x2::cbf::generated::kNaturalStanding;
  actual.arm.assign(standing.begin(), standing.end());
  actual.hand.assign(20, 0.0);
  for (std::size_t axis = 0; axis < 4; ++axis) {
    actual.hand[kNativeRootSlots[axis]] = x2::cbf::kRigidThumbBaselineUrdf[x2::cbf::kRigidThumbIndices[axis]];
  }
  const Output initial = actual;
  // Saved .5 formal frame 39137, seed 20261108: this endpoint penetrates
  // right_forearm/head, a primitive pair independent of thumb angles.
  Output held_goal = initial;
  held_goal.arm = {
    0.35188810177026719, 3.05552669153681, -0.43347949253335571, -0.94260207595231216,
    -0.23915094010658641, -0.067878510294076336, 0.24717439849856296,
    -0.44410450928103035, -2.9146311500677924, -1.3323410446378825, -2.1121877003161917,
    0.35548866369794113, -0.30264380550710107, 0.24458409666527142};
  std::vector<double> elapsed;
  std::size_t finite_errors = 0, limit_errors = 0, step_errors = 0, pair_errors = 0;
  std::size_t publication_pair_errors = 0;
  std::size_t collision_pairs = 0, collision_frames = 0, published_checked = 0;
  std::size_t local_walk_frames = 0, replayed = 0;
  double arm_travel = 0.0, hand_travel = 0.0, max_arm_excursion = 0.0, max_arm_step = 0.0;
  bool finite_timing = true;
  const uint64_t epoch = 1;
  auto distances_for = [&config](const Output & output) {
      x2::cbf::KinematicsInput geometry;
      std::copy_n(output.arm.begin(), 14, geometry.q_arm.begin());
      geometry.q_thumb_urdf.fill(0.0);
      geometry.thumb_pose_valid = true;
      geometry.hands_installed = config.hands_installed;
      for (std::size_t axis = 0; axis < 4; ++axis) {
        geometry.q_thumb_urdf[x2::cbf::kRigidThumbIndices[axis]] = output.hand[kNativeRootSlots[axis]];
      }
      return x2::cbf::evaluate_proxy_distances(geometry);
    };
  auto validate = [&](const Output & output) {
      if (output.arm.size() != 14 || output.hand.size() != 20 ||
        output.kind != Output::Kind::kUpperBody ||
        output.hand_sub_mode != aimdk_dataplane::kHandSubModeDexterousJoint)
      {
        ++finite_errors;
        return false;
      }
      const auto finite_values = [](const auto & values) {
          return std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);});
        };
      if (!finite_values(output.arm) || !finite_values(output.hand) ||
        !finite_values(output.head) || !finite_values(output.twist))
      {
        ++finite_errors;
        return false;
      }
      for (std::size_t joint = 0; joint < 14; ++joint) {
        const double step = std::abs(output.arm[joint] - actual.arm[joint]);
        max_arm_step = std::max(max_arm_step, step);
        if (output.arm[joint] < config.lower[joint] - 1e-10 ||
          output.arm[joint] > config.upper[joint] + 1e-10) {++limit_errors;}
        if (step > std::min(0.04, config.velocity[joint] * config.period_s) + 1e-9) {++step_errors;}
      }
      for (std::size_t joint = 0; joint < 20; ++joint) {
        // Non-root slots stay at zero in this fixture; no SDK hand limits are inferred.
        const auto root = std::find(kNativeRootSlots.begin(), kNativeRootSlots.end(), joint);
        if (root == kNativeRootSlots.end()) {
          if (output.hand[joint] != 0.0) {++limit_errors;}
        } else {
          const auto axis = static_cast<std::size_t>(root - kNativeRootSlots.begin());
          if (output.hand[joint] < config.thumb_model_limits[axis][0] - 1e-12 ||
            output.hand[joint] > config.thumb_model_limits[axis][1] + 1e-12) {++limit_errors;}
          // A bound of this synthetic +/-0.02-rad fixture, not a production hand limiter.
          if (std::abs(output.hand[joint] - actual.hand[joint]) > 0.04 + 1e-12) {++step_errors;}
        }
      }
      const auto distances = distances_for(output);
      if (!std::isfinite(distances.minimum_distance)) {++pair_errors;}
      std::size_t collisions = 0;
      for (const auto & distance : distances.distances) {
        if (!distance.valid || !std::isfinite(distance.signed_distance)) {++pair_errors;}
        else if (distance.signed_distance < 0.0) {++collisions;}
      }
      collision_pairs += collisions;
      collision_frames += collisions > 0;
      ++published_checked;
      return finite_errors == 0 && limit_errors == 0 && step_errors == 0 && pair_errors == 0 && collisions == 0;
    };
  auto snapshot_for = [&](const Output & goal) {
      CbfSnapshot snapshot;
      snapshot.epoch = epoch;
      snapshot.requested = goal;
      snapshot.arm = Measured{actual.arm, 0.0};
      Measured thumb;
      for (std::size_t axis = 0; axis < kNativeRootSlots.size(); ++axis) {
        if (config.hands_installed[axis / 2]) {thumb.position.push_back(actual.hand[kNativeRootSlots[axis]]);}
      }
      snapshot.thumb = thumb;
      return snapshot;
    };
  auto sample_timing = [&]() {
      const double value = safety.diagnostics().elapsed_us;
      if (!std::isfinite(value) || value < 0.0) {finite_timing = false;}
      else {elapsed.push_back(value);}
    };

  // Seed an accepted, committed standing target before introducing a held goal.
  const auto seed = safety.filter(snapshot_for(actual), 1.0);
  sample_timing();
  const bool seeded = seed.accept_new && validate(seed.output);
  if (seeded) {safety.committed(seed, 1.0); actual = seed.output;}
  else if (seed.output.kind == Output::Kind::kNone) {safety.committed(seed, 1.0);}
  else {safety.discarded();}
  if (seeded) {
    for (std::size_t frame = 0; frame < frames; ++frame) {
      ++replayed;
      const bool local_walk = frames >= 20 && frame >= frames / 2 && frame < frames / 2 + 5;
      Output goal = held_goal;
      goal.sequence = static_cast<uint32_t>(frame + 1);
      if (local_walk) {
        ++local_walk_frames;
        goal.arm = actual.arm;
        for (std::size_t joint = 0; joint < goal.arm.size(); ++joint) {
          goal.arm[joint] += 0.01 * std::sin(
            0.7 * static_cast<double>(frame - frames / 2 + 1) + 0.5 * static_cast<double>(joint));
        }
      }
      if ((frame / 40) % 4 == 1) {
        for (std::size_t axis = 0; axis < 4; ++axis) {
          goal.hand[kNativeRootSlots[axis]] += 0.02 * std::sin(0.05 * frame + 0.5 * axis);
        }
      }
      const double now = 1.0 + config.period_s * static_cast<double>(frame + 1);
      const auto decision = safety.filter(snapshot_for(goal), now);
      sample_timing();
      if (decision.output.kind == Output::Kind::kNone) {
        if (decision.accept_new) {++publication_pair_errors; safety.discarded(); break;}
        safety.committed(decision, now);  // Counts withheld; actual remains the previous published pair.
        continue;
      }
      const auto & expected_hand = decision.hand_held ? actual.hand : goal.hand;
      if ((decision.accept_new && decision.output.hand != expected_hand) ||
        (!decision.accept_new && (decision.output.arm != actual.arm || decision.output.hand != actual.hand)))
      {
        ++publication_pair_errors;  // Only an explicitly deferred whole gesture may accompany a new arm target.
        safety.discarded();
        break;
      }
      if (!validate(decision.output)) {safety.discarded(); break;}
      for (std::size_t joint = 0; joint < 14; ++joint) {
        arm_travel += std::abs(decision.output.arm[joint] - actual.arm[joint]);
        max_arm_excursion = std::max(max_arm_excursion, std::abs(decision.output.arm[joint] - initial.arm[joint]));
      }
      for (std::size_t joint = 0; joint < 20; ++joint) {
        hand_travel += std::abs(decision.output.hand[joint] - actual.hand[joint]);
      }
      safety.committed(decision, now);
      actual = decision.output;  // Ideal numeric tracking of the published COMPLETE pair only.
    }
  }
  const auto final_distances = distances_for(actual);
  const auto held_goal_distances = distances_for(held_goal);
  if (!std::isfinite(final_distances.minimum_distance)) {++pair_errors;}
  std::size_t final_collisions = 0;
  for (const auto & distance : final_distances.distances) {
    if (!distance.valid || !std::isfinite(distance.signed_distance)) {++pair_errors;}
    else if (distance.signed_distance < 0.0) {++final_collisions;}
  }
  std::sort(elapsed.begin(), elapsed.end());
  const auto percentile = [&](double fraction) {
      if (elapsed.empty()) {return 0.0;}
      return elapsed[static_cast<std::size_t>(std::ceil(fraction * elapsed.size())) - 1];
    };
  const auto & stats = safety.diagnostics();
  const bool moved = arm_travel > 1e-5 && max_arm_excursion > 1e-5 && stats.accepted > 1;
  const bool geometry_intervention_observed = stats.geometry_modified > 0;
  const bool passed = seeded && replayed == frames && moved && geometry_intervention_observed &&
    finite_timing && finite_errors == 0 &&
    limit_errors == 0 && step_errors == 0 && pair_errors == 0 && publication_pair_errors == 0 &&
    collision_pairs == 0 && final_collisions == 0 && held_goal_distances.minimum_distance < 0.0;
  auto report = scope();
  report["probe"] = "bounded_urdf_identity_numeric_replay";
  report["warning"] = "URDF identity test fixture only; no real SDK, dynamics, calibration or physical safety validation.";
  report["identity_fixture_verified"] = true;  // Only the test's internal coordinate identity.
  report["fixture_config"] = settings;
  report["frames_requested"] = frames;
  report["frames_replayed"] = replayed;
  report["local_walk_frames"] = local_walk_frames;
  report["local_walk_model"] = "five local joint-space perturbations about actual.arm, amplitude 0.01 rad, same mirror epoch";
  report["sleep"] = false;
  report["seed_accepted_and_committed"] = seeded;
  report["timing_samples_include_seed"] = true;
  report["adapter_elapsed_us"] = {
    {"samples", elapsed.size()}, {"p50", percentile(0.50)}, {"p95", percentile(0.95)},
    {"p99", percentile(0.99)}, {"max", elapsed.empty() ? 0.0 : elapsed.back()},
  };
  report["counts_include_seed"] = true;
  report["held_goal_min_raw_distance_m"] = held_goal_distances.minimum_distance;
  report["held_goal_is_raw_collision"] = held_goal_distances.minimum_distance < 0.0;
  report["timing_within_10ms"] = !elapsed.empty() && elapsed.back() <= config.deadline_us;
  report["counts"] = {
    {"calls", stats.calls}, {"accepted", stats.accepted}, {"published", stats.published}, {"modified", stats.modified},
    {"geometry_modified", stats.geometry_modified}, {"held", stats.held}, {"withheld", stats.withheld},
    {"deadlines", stats.deadlines},
    {"verification_reuses", stats.verification_reuses}, {"hand_budget_holds", stats.hand_budget_holds},
  };
  report["stage_max_us"] = {
    {"hand_check", stats.max_hand_check_us}, {"verification", stats.max_verification_us},
    {"projection", stats.max_projection_us},
  };
  report["motion_rad"] = {
    {"arm_total_absolute", arm_travel}, {"hand_total_absolute", hand_travel},
    {"max_arm_excursion", max_arm_excursion}, {"max_arm_step", max_arm_step},
  };
  report["checks"] = {
    {"finite_errors", finite_errors}, {"limit_errors", limit_errors}, {"step_errors", step_errors},
    {"pair_errors", pair_errors}, {"publication_pair_errors", publication_pair_errors},
    {"published_checked", published_checked},
    {"raw_collision_pairs_total", collision_pairs}, {"raw_collision_frames", collision_frames},
    {"final_raw_collision_pairs", final_collisions}, {"actual_motion_observed", moved},
  };
  report["final_min_distance"] = final_distances.minimum_distance;
  report["final_arm14"] = actual.arm;
  report["final_hand20"] = actual.hand;
  report["geometry_intervention_observed"] = geometry_intervention_observed;
  report["passed"] = passed;
  report["pass_scope"] = "finite bounded published numeric poses, nonzero motion, geometry intervention, zero raw proxy collisions; not hardware certification";
  return report;
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      auto help = scope();
      help["usage"] = "cbf_pose_probe [--input FILE.json | --replay N [--left-hand-only] | --left-hand-only]";
      help["input_fields"] = {{"arm14", "14 finite numbers"}, {"hand20", "20 finite numbers, dual-hand input"},
        {"left_hand10", "10 finite numbers instead of hand20 for explicitly left-only hardware"}};
      help["default"] = "Mirror reset arm14 and hand_overlay.py HAND_NEUTRAL in native command coordinates";
      help["exit_codes"] = {
        {"0", "static query completed, help, or numeric replay passed; never a physical safety pass"},
        {"1", "numeric replay failed its checks"}, {"2", "input/configuration error"},
      };
      std::cout << help.dump(2) << '\n';
      return 0;
    }
    if ((argc == 3 || (argc == 4 && std::string(argv[3]) == "--left-hand-only")) &&
      std::string(argv[1]) == "--replay")
    {
      std::size_t consumed = 0;
      const std::string value = argv[2];
      const int frames = std::stoi(value, &consumed);
      if (consumed != value.size() || frames < 1 || frames > 1000) {
        throw std::invalid_argument("--replay N requires an integer from 1 through 1000");
      }
      const auto report = replay(static_cast<std::size_t>(frames), argc == 4);
      std::cout << report.dump(2) << '\n';
      return report.at("passed").get<bool>() ? 0 : 1;
    }
    Pose pose;
    std::string source;
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--left-hand-only")) {
      pose = default_pose();
      if (argc == 2) {pose.hands_installed = {true, false};}
      source = "mirror_reset_and_hand_overlay_neutral";
    } else if (argc == 3 && std::string(argv[1]) == "--input") {
      source = argv[2];
      pose = load_pose(source);
    } else {
      throw std::invalid_argument("usage: cbf_pose_probe [--input FILE.json | --replay N]");
    }
    std::cout << probe(pose, source).dump(2) << '\n';
    return 0;  // Successful numerical query, including unsafe/withheld results.
  } catch (const std::exception & error) {
    auto report = scope();
    report["query_completed"] = false;
    report["error"] = error.what();
    std::cout << report.dump(2) << '\n';
    return 2;
  }
}
