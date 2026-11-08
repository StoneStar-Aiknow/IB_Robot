// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
#include "aimdk_dataplane/cbf_safety.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cbf/safety_filter.hpp"
#include "nlohmann/json.hpp"

namespace aimdk_dataplane
{
namespace
{
using Clock = std::chrono::steady_clock;
using nlohmann::json;
constexpr std::array<size_t, 4> kHandSlots{0, 1, 10, 11};
constexpr std::array<size_t, 4> kModelSlots{0, 1, 3, 4};

template<size_t N>
bool finite(const std::array<double, N> & values)
{
  return std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);});
}

bool finite(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);});
}

double elapsed(Clock::time_point start)
{
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

template<size_t N>
std::array<double, N> array_from(const json & value, const char * key)
{
  const auto & entry = value.at(key);
  if (!entry.is_array() || entry.size() != N) {
    throw std::invalid_argument(std::string("wrong CBF array length: ") + key);
  }
  const auto result = entry.get<std::array<double, N>>();
  if (!finite(result)) {throw std::invalid_argument(std::string("nonfinite CBF array: ") + key);}
  return result;
}

bool same_arm(const std::array<double, 14> & a, const std::vector<double> & b)
{
  if (b.size() != a.size()) {return false;}
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > 1e-10) {return false;}
  }
  return true;
}
}  // namespace

CbfSafetyConfig CbfSafetyConfig::from_json(const std::string & text, double rate_hz)
{
  CbfSafetyConfig out;
  if (text.empty()) {return out;}
  const auto value = json::parse(text);
  out.enabled = value.value("enabled", false);
  if (!out.enabled) {return out;}
  const auto mode = value.value("mode", std::string("shadow"));
  out.enforce = mode == "enforce";
  out.mirror_mode = value.at("protected_mirror_mode").get<std::string>();
  out.deadline_us = value.value("deadline_us", out.deadline_us);
  out.guard_us = value.value("deadline_guard_us", out.guard_us);
  out.feedback_max_age_s = value.value("feedback_max_age_s", out.feedback_max_age_s);
  out.state_skew_max_s = value.value("state_skew_max_s", out.state_skew_max_s);
  out.feedback_limit_tolerance_rad = value.value("feedback_limit_tolerance_rad", 0.0);
  out.check_arm_feedback_goal_range = value.value("check_arm_feedback_goal_range", true);
  if ((mode != "shadow" && mode != "enforce") || out.mirror_mode.empty() ||
    !std::isfinite(rate_hz) || rate_hz <= 0.0 || !std::isfinite(out.deadline_us) ||
    !std::isfinite(out.guard_us) || out.deadline_us <= 0.0 || out.guard_us < 0.0 ||
    out.guard_us >= out.deadline_us || !std::isfinite(out.feedback_max_age_s) || out.feedback_max_age_s <= 0.0 ||
    !std::isfinite(out.state_skew_max_s) || out.state_skew_max_s < 0.0 ||
    !std::isfinite(out.feedback_limit_tolerance_rad) || out.feedback_limit_tolerance_rad < 0.0 ||
    out.feedback_limit_tolerance_rad > 0.1)
  {
    throw std::invalid_argument("invalid CBF mode, period or budget");
  }
  out.period_s = 1.0 / rate_hz;
  if (value.contains("hands_installed")) {
    const auto & hands = value.at("hands_installed");
    if (!hands.is_array() || hands.size() != 2 || !hands[0].is_boolean() || !hands[1].is_boolean()) {
      throw std::invalid_argument("CBF hands_installed must contain two booleans: left, right");
    }
    out.hands_installed = hands.get<std::array<bool, 2>>();
  }
  if (!out.hands_installed[0] && !out.hands_installed[1]) {
    throw std::invalid_argument("CBF v0.5 requires at least one installed OmniHand");
  }
  out.thumb_model_limits = x2::cbf::kRigidThumbLimits;
  for (size_t j = 0; j < 14; ++j) {
    out.lower[j] = x2::cbf::generated::kArmLimits[j][0];
    out.upper[j] = x2::cbf::generated::kArmLimits[j][1];
    out.velocity[j] = 0.04 / out.period_s;
    out.arm_feedback_limits[j] = x2::cbf::generated::kArmLimits[j];
  }
  if (value.contains("arm_feedback_lower_limits") || value.contains("arm_feedback_upper_limits")) {
    const auto lower = array_from<14>(value, "arm_feedback_lower_limits");
    const auto upper = array_from<14>(value, "arm_feedback_upper_limits");
    for (size_t j = 0; j < 14; ++j) {
      if (lower[j] >= upper[j] || !std::isfinite(upper[j] - lower[j])) {
        throw std::invalid_argument("invalid CBF feedback joint interval");
      }
      out.arm_feedback_limits[j] = {lower[j], upper[j]};
      out.lower[j] = std::max(out.lower[j], lower[j]);
      out.upper[j] = std::min(out.upper[j], upper[j]);
    }
  }
  if (value.contains("arm_lower_limits")) {
    const auto lower = array_from<14>(value, "arm_lower_limits");
    const auto upper = array_from<14>(value, "arm_upper_limits");
    if (!finite(lower) || !finite(upper)) {throw std::invalid_argument("nonfinite CBF arm limits");}
    for (size_t j = 0; j < 14; ++j) {
      out.lower[j] = std::max(out.lower[j], lower[j]);
      out.upper[j] = std::min(out.upper[j], upper[j]);
    }
  }
  // Match the existing operator/model command intersection and its one-degree margin,
  // including direct callers that did not supply application URDF bands.
  constexpr double margin = 0.017453292519943295;
  for (size_t j = 0; j < 14; ++j) {
    out.lower[j] += margin;
    out.upper[j] -= margin;
    if (out.lower[j] > out.upper[j]) {throw std::invalid_argument("empty CBF joint interval");}
  }
  if (value.contains("arm_velocity_limits")) {
    const auto speeds = array_from<14>(value, "arm_velocity_limits");
    if (!finite(speeds)) {throw std::invalid_argument("nonfinite CBF velocity limits");}
    for (size_t j = 0; j < 14; ++j) {
      if (speeds[j] <= 0.0) {throw std::invalid_argument("nonpositive CBF velocity limit");}
      out.velocity[j] = std::min(out.velocity[j], speeds[j]);
    }
  }
  if (value.contains("thumb_mapping")) {
    const auto & mapping = value.at("thumb_mapping");
    out.mapping_verified = mapping.value("verified", false);
    out.mapping_available = mapping.contains("command_scale") || mapping.contains("command_offset") ||
      mapping.contains("feedback_scale") || mapping.contains("feedback_offset");
    if (out.mapping_verified || out.mapping_available) {
      out.command_scale = array_from<4>(mapping, "command_scale");
      out.command_offset = array_from<4>(mapping, "command_offset");
      out.feedback_scale = array_from<4>(mapping, "feedback_scale");
      out.feedback_offset = array_from<4>(mapping, "feedback_offset");
      if (!finite(out.command_scale) || !finite(out.command_offset) ||
        !finite(out.feedback_scale) || !finite(out.feedback_offset))
      {
        throw std::invalid_argument("nonfinite CBF thumb mapping");
      }
      for (size_t j = 0; j < 4; ++j) {
        if (out.command_scale[j] == 0.0 || out.feedback_scale[j] == 0.0) {
          throw std::invalid_argument("zero CBF thumb mapping scale");
        }
      }
      out.mapping_available = true;
    }
    if (mapping.contains("native_lower") || mapping.contains("native_upper")) {
      if (!out.mapping_available) {throw std::invalid_argument("native CBF bounds require a mapping");}
      out.native_lower = array_from<4>(mapping, "native_lower");
      out.native_upper = array_from<4>(mapping, "native_upper");
      out.native_limits_available = true;
      for (size_t j = 0; j < 4; ++j) {
        if (out.native_lower[j] > out.native_upper[j]) {throw std::invalid_argument("reversed native CBF bounds");}
        const double ca = out.command_scale[j] * out.native_lower[j] + out.command_offset[j];
        const double cb = out.command_scale[j] * out.native_upper[j] + out.command_offset[j];
        const double fa = out.feedback_scale[j] * out.native_lower[j] + out.feedback_offset[j];
        const double fb = out.feedback_scale[j] * out.native_upper[j] + out.feedback_offset[j];
        out.thumb_model_limits[j] = {std::min({ca, cb, fa, fb}), std::max({ca, cb, fa, fb})};
        const double span = out.thumb_model_limits[j][1] - out.thumb_model_limits[j][0];
        if (!std::isfinite(out.thumb_model_limits[j][0]) || !std::isfinite(out.thumb_model_limits[j][1]) ||
          !std::isfinite(span) || span > 6.283185307179586)
        {
          throw std::invalid_argument("nonfinite mapped CBF bounds");
        }
      }
    }
  }
  return out;
}

bool CbfSafetyConfig::matches_hand_types(const std::vector<int> & types) const
{
  if (types.size() != 2) {return false;}
  for (size_t side = 0; side < hands_installed.size(); ++side) {
    // This geometry and mapping are for the OmniHand reported as SDK type 1.
    // A dropout or a newly attached hand must never remove collision pairs.
    if (types[side] != (hands_installed[side] ? 1 : 0)) {return false;}
  }
  return true;
}

std::string CbfSafetyConfig::feedback_error(const CbfSnapshot & snapshot) const
{
  const size_t count = 2 * static_cast<size_t>(hands_installed[0]) +
    2 * static_cast<size_t>(hands_installed[1]);
  if (!snapshot.arm || !snapshot.thumb || snapshot.arm->position.size() != 14 ||
    snapshot.thumb->position.size() != count || !finite(snapshot.arm->position) || !finite(snapshot.thumb->position) ||
    !std::isfinite(snapshot.arm->age_s) || !std::isfinite(snapshot.thumb->age_s) ||
    snapshot.arm->age_s < 0.0 || snapshot.thumb->age_s < 0.0 ||
    snapshot.arm->age_s > feedback_max_age_s || snapshot.thumb->age_s > feedback_max_age_s)
  {
    return "feedback_unavailable";
  }
  if (std::abs(snapshot.arm->age_s - snapshot.thumb->age_s) > state_skew_max_s) {return "feedback_skew";}
  if (native_limits_available) {
    size_t axis = 0;
    for (size_t j = 0; j < 4; ++j) {
      if (!hands_installed[j / 2]) {continue;}
      const double measured = snapshot.thumb->position[axis++];
      if (measured < native_lower[j] - feedback_limit_tolerance_rad - 1e-9 ||
        measured > native_upper[j] + feedback_limit_tolerance_rad + 1e-9)
      {
        return "native_feedback_out_of_range";
      }
    }
  }
  return {};
}

CbfSafety::CbfSafety(CbfSafetyConfig config)
: config_(std::move(config)) {}

void CbfSafety::inactive(uint64_t epoch)
{
  invalid_input_episode_ = false;
  if (epoch != epoch_ || diagnostics_.active) {
    accepted_.reset();
    epoch_ = epoch;
  }
  diagnostics_.active = false;
  diagnostics_.recovery_verified = false;
  diagnostics_.hand_held = false;
}

CbfDecision CbfSafety::filter(const CbfSnapshot & snapshot, double now)
{
  const auto start = Clock::now();
  const auto configured_deadline = start + std::chrono::duration_cast<Clock::duration>(
    std::chrono::duration<double, std::micro>(config_.deadline_us));
  const auto deadline = snapshot.publication_deadline ?
    std::min(configured_deadline, *snapshot.publication_deadline) : configured_deadline;
  const auto remaining = [&]() {
      return std::chrono::duration<double, std::micro>(deadline - Clock::now()).count();
    };
  if (snapshot.epoch != epoch_) {
    accepted_.reset();
    invalid_input_episode_ = false;
    epoch_ = snapshot.epoch;
  }
  ++diagnostics_.calls;
  diagnostics_.active = true;
  diagnostics_.distance_valid = false;
  diagnostics_.thumb_angles_valid = false;
  diagnostics_.available_budget_us = std::max(0.0, remaining());
  diagnostics_.hand_check_us = 0.0;
  diagnostics_.verification_us = 0.0;
  diagnostics_.projection_us = 0.0;
  diagnostics_.arm_age_s = snapshot.arm ? snapshot.arm->age_s : -1.0;
  diagnostics_.thumb_age_s = snapshot.thumb ? snapshot.thumb->age_s : -1.0;
  CbfDecision decision;
  decision.epoch = snapshot.epoch;
  bool invalid_input_this_frame = false;
  const auto finish = [&](const std::string & reason) {
      if (!invalid_input_this_frame) {invalid_input_episode_ = false;}
      diagnostics_.recovery_verified = decision.accept_new && decision.recovery_verified;
      diagnostics_.hand_held = decision.accept_new && decision.hand_held;
      // Calibration provenance is diagnostic, not an additional motion gate.
      const auto detail = decision.hand_held ? "arm_recovery_hand_held/" + reason : reason;
      diagnostics_.reason = config_.mapping_available && !config_.mapping_verified ?
        "unverified_mapping/" + detail : detail;
      diagnostics_.elapsed_us = elapsed(start);
      diagnostics_.max_elapsed_us = std::max(diagnostics_.max_elapsed_us, diagnostics_.elapsed_us);
      diagnostics_.max_hand_check_us = std::max(diagnostics_.max_hand_check_us, diagnostics_.hand_check_us);
      diagnostics_.max_verification_us = std::max(diagnostics_.max_verification_us, diagnostics_.verification_us);
      diagnostics_.max_projection_us = std::max(diagnostics_.max_projection_us, diagnostics_.projection_us);
      if (!config_.enforce) {
        decision.output = snapshot.requested;
        decision.accept_new = false;
      }
      return decision;
    };
  if (!config_.mapping_available) {return finish("mapping_required");}
  if (!snapshot.hand_configuration_matches) {return finish("hand_configuration_mismatch");}
  if (snapshot.named_arm_cache) {return finish("mixed_named_arm");}
  // This is a retransmission of a committed command, not a newly verified
  // trajectory. Prepare it before feedback/solver checks so their failures do
  // not stop the controller stream. Authorization exits above cannot use it.
  if (accepted_ && (snapshot.requested.kind == Output::Kind::kUpperBody ||
    snapshot.requested.kind == Output::Kind::kNone))
  {
    decision.prepared_hold = *accepted_;
    if (snapshot.requested.head.size() == 2 && finite(snapshot.requested.head)) {
      decision.prepared_hold.head = snapshot.requested.head;
    }
    decision.prepared_hold.sequence = snapshot.requested.sequence;
  }
  const auto fallback = [&](const std::string & reason) {
      decision.output = decision.prepared_hold;
      decision.accept_new = false;
      decision.modified = false;
      decision.geometry_modified = false;
      decision.recovery_verified = false;
      decision.hand_held = false;
      diagnostics_.distance_valid = false;
      return finish(reason);
    };
  if (!snapshot.source_ready) {
    return fallback(accepted_ ? "mirror_input_stale" : "awaiting_mirror_input");
  }
  if (remaining() <= config_.guard_us) {
    ++diagnostics_.deadlines;
    return fallback("deadline");
  }
  bool command_valid = snapshot.requested.kind == Output::Kind::kUpperBody &&
    snapshot.requested.arm.size() == 14 && snapshot.requested.hand_sub_mode == kHandSubModeDexterousJoint &&
    snapshot.requested.hand.size() == 20 && finite(snapshot.requested.arm) && finite(snapshot.requested.hand);
  const auto feedback_problem = config_.feedback_error(snapshot);
  if (!feedback_problem.empty()) {return fallback(feedback_problem);}
  if (config_.native_limits_available) {
    for (size_t j = 0; j < 4; ++j) {
      if (!config_.hands_installed[j / 2]) {continue;}
      if (command_valid && (snapshot.requested.hand[kHandSlots[j]] < config_.native_lower[j] - 1e-9 ||
        snapshot.requested.hand[kHandSlots[j]] > config_.native_upper[j] + 1e-9))
      {
        command_valid = false;
      }
    }
  }
  std::array<std::array<double, 2>, 14> command_box;
  for (size_t j = 0; j < command_box.size(); ++j) {
    command_box[j] = {config_.lower[j], config_.upper[j]};
  }
  auto input_for = [&](const Output & target) {
      x2::cbf::SafetyFilterInput input;
      std::copy_n(snapshot.arm->position.begin(), 14, input.q_current.begin());
      std::copy_n(target.arm.begin(), 14, input.q_nominal.begin());
      // Absent-hand transforms are numerical placeholders excluded by the
      // explicit geometry mask; they are never labelled as measurements.
      input.q_thumb_urdf = x2::cbf::kRigidThumbBaselineUrdf;
      input.q_thumb_current_urdf = x2::cbf::kRigidThumbBaselineUrdf;
      input.thumb_current_valid = true;
      input.thumb_joint_limits_urdf = config_.thumb_model_limits;
      input.current_arm_joint_limits = config_.arm_feedback_limits;
      input.check_current_arm_goal_range = config_.check_arm_feedback_goal_range;
      input.command_joint_limits = command_box;
      // Profile travel limits constrain goals, not exact encoder samples. Keep
      // raw observations for every FK/path query; only admission has an allowance.
      for (auto & band : input.current_arm_joint_limits) {
        band[0] -= config_.feedback_limit_tolerance_rad;
        band[1] += config_.feedback_limit_tolerance_rad;
      }
      input.hands_installed = config_.hands_installed;
      size_t feedback_axis = 0;
      for (size_t j = 0; j < 4; ++j) {
        if (!config_.hands_installed[j / 2]) {
          input.thumb_joint_limits_urdf[j] = x2::cbf::kRigidThumbLimits[j];
          continue;
        }
        input.q_thumb_urdf[kModelSlots[j]] =
          config_.command_scale[j] * target.hand[kHandSlots[j]] + config_.command_offset[j];
        input.q_thumb_current_urdf[kModelSlots[j]] =
          config_.feedback_scale[j] * snapshot.thumb->position[feedback_axis++] + config_.feedback_offset[j];
      }
      input.current_thumb_joint_limits_urdf = input.thumb_joint_limits_urdf;
      for (size_t j = 0; j < 4; ++j) {
        if (!config_.hands_installed[j / 2]) {continue;}
        const double allowance = std::abs(config_.feedback_scale[j]) * config_.feedback_limit_tolerance_rad;
        (*input.current_thumb_joint_limits_urdf)[j][0] -= allowance;
        (*input.current_thumb_joint_limits_urdf)[j][1] += allowance;
      }
      return input;
    };
  if (command_valid) {
    const auto preview = input_for(snapshot.requested);
    for (size_t j = 0; j < 4; ++j) {
      diagnostics_.thumb_current_urdf[j] = preview.q_thumb_current_urdf[kModelSlots[j]];
      diagnostics_.thumb_target_urdf[j] = preview.q_thumb_urdf[kModelSlots[j]];
    }
    diagnostics_.thumb_angles_valid = true;
  }
  const auto run = [&](const Output & target, double step, bool verify_only, const char * stage,
      bool thumb_only = false, double budget_cap_us = -1.0) {
      const auto call_start = Clock::now();
      const auto input = input_for(target);
      x2::cbf::SafetyFilterConfig cfg;
      const double available = remaining();
      cfg.deadline_us = std::max(1.0, budget_cap_us < 0.0 ? available : std::min(available, budget_cap_us));
      cfg.deadline_guard_us = std::min(config_.guard_us, cfg.deadline_us);
      cfg.max_step_rad = step;
      cfg.path_check_substeps = 64;
      cfg.allow_airwall_recovery = true;
      const auto result = thumb_only ? x2::cbf::verify_thumb_target_with_arm_held(input, cfg) :
        (verify_only ? x2::cbf::verify_arm_target(input, cfg) : x2::cbf::filter_arm_target(input, cfg));
      const double cost = elapsed(call_start);
      if (thumb_only) {diagnostics_.hand_check_us += cost;}
      else if (verify_only) {diagnostics_.verification_us += cost;}
      else {diagnostics_.projection_us += cost;}
      if (result.status == x2::cbf::FilterStatus::InvalidInputFallback && !invalid_input_this_frame) {
        invalid_input_this_frame = true;
        ++diagnostics_.invalid_input_frames;
        if (!invalid_input_episode_ || !diagnostics_.last_input_fault ||
          diagnostics_.last_input_fault->issue != result.invalid_input_reason ||
          diagnostics_.last_input_fault->index != result.invalid_input_index)
        {
          CbfInputFault fault;
          fault.event = diagnostics_.last_input_fault ? diagnostics_.last_input_fault->event + 1 : 1;
          fault.call = diagnostics_.calls;
          fault.epoch = snapshot.epoch;
          fault.command_sequence = snapshot.requested.sequence;
          fault.at_s = now;
          fault.stage = stage;
          fault.issue = result.invalid_input_reason;
          fault.index = result.invalid_input_index;
          fault.value = result.invalid_input_value;
          fault.lower = result.invalid_input_lower;
          fault.upper = result.invalid_input_upper;
          fault.arm_current = input.q_current;
          fault.arm_target = input.q_nominal;
          fault.arm_feedback_limits = input.current_arm_joint_limits;
          fault.arm_requested = snapshot.requested.arm;
          fault.thumb_current_urdf = input.q_thumb_current_urdf;
          fault.thumb_target_urdf = input.q_thumb_urdf;
          fault.thumb_limits_urdf = input.thumb_joint_limits_urdf;
          fault.thumb_current_limits_urdf = *input.current_thumb_joint_limits_urdf;
          fault.thumb_native_current = snapshot.thumb->position;
          for (size_t axis = 0; axis < kHandSlots.size(); ++axis) {
            fault.thumb_native_target[axis] = target.hand[kHandSlots[axis]];
          }
          fault.arm_age_s = snapshot.arm->age_s;
          fault.thumb_age_s = snapshot.thumb->age_s;
          diagnostics_.last_input_fault = std::move(fault);
          invalid_input_episode_ = true;
        }
      }
      return result;
    };

  if (!command_valid) {
    return fallback(snapshot.requested.kind == Output::Kind::kNone ? "candidate_unavailable" : "invalid_command");
  }
  if (remaining() <= config_.guard_us) {
    ++diagnostics_.deadlines;
    return fallback("deadline");
  }
  Output target = snapshot.requested;
  bool hand_held = false;
  bool new_thumb_target = false;
  if (config_.enforce && accepted_) {
    for (size_t axis = 0; axis < kHandSlots.size(); ++axis) {
      if (config_.hands_installed[axis / 2] && target.hand[kHandSlots[axis]] != accepted_->hand[kHandSlots[axis]]) {
        new_thumb_target = true;
      }
    }
  }
  if (new_thumb_target) {
    // The hand may reach its new pose before the arm moves. This API checks
    // that ordering at raw measured arm coordinates; its q_safe is NEVER used
    // as a command. An unsafe gesture waits while the arm can still withdraw.
    // Gesture ordering cannot consume the whole frame while the user needs
    // the arm to withdraw. Reserve three quarters of the available computation
    // window for arm verification/projection, without changing the frame limit.
    const double hand_budget = 0.25 * std::max(0.0, remaining() - config_.guard_us);
    const auto hand_first = run(target, 6.5, true, "thumb_ordering", true, hand_budget);
    if (remaining() <= config_.guard_us) {
      ++diagnostics_.deadlines;
      return fallback("deadline");
    }
    if (hand_first.deadline_missed) {
      ++diagnostics_.hand_budget_holds;
      target.hand = accepted_->hand;
      hand_held = true;
    } else if (hand_first.status == x2::cbf::FilterStatus::InvalidInputFallback) {
      return fallback(x2::cbf::filter_status_name(hand_first.status));
    } else if (!hand_first.thumb_target_accepted || !hand_first.nonlinear_verified) {
      target.hand = accepted_->hand;
      hand_held = true;
    }
  }
  const auto & reference = accepted_ ? accepted_->arm : snapshot.arm->position;
  const double dt = accepted_ ? std::min(config_.period_s, std::max(0.0, now - last_published_)) : config_.period_s;
  for (size_t j = 0; j < 14; ++j) {
    const double step = config_.velocity[j] * dt;
    const double low = std::max(config_.lower[j], reference[j] - step);
    const double high = std::min(config_.upper[j], reference[j] + step);
    if (low > high) {return fallback("original_motion_limit");}
    command_box[j] = {low, high};
    target.arm[j] = std::clamp(target.arm[j], low, high);
  }
  const auto bounded_arm = target.arm;
  // One retry may defer the entire new gesture, never alter its internal
  // angles. Both attempts share the same command box, deadline and verifier.
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto result = run(target, 6.5, true, hand_held ? "held_hand_nominal" : "nominal");
    bool projected = false;
    if (!result.thumb_target_accepted || !result.nonlinear_verified) {
      if (result.deadline_missed || remaining() <= config_.guard_us) {
        ++diagnostics_.deadlines;
        return fallback("deadline");
      }
      if (result.status != x2::cbf::FilterStatus::InvalidInputFallback) {
        result = run(target, 6.5, false, hand_held ? "held_hand_projection" : "projection");
        projected = true;
      }
    }
    if (result.deadline_missed || remaining() <= config_.guard_us) {
      ++diagnostics_.deadlines;
      return fallback("deadline");
    }
    if (result.thumb_target_accepted && result.nonlinear_verified) {
      // The solver works in this box, and this independent guard also applies
      // to exact nominal acceptance. Measured positions remain raw FK inputs.
      for (size_t j = 0; j < command_box.size(); ++j) {
        if (!std::isfinite(result.q_safe[j]) || result.q_safe[j] < command_box[j][0] - 1e-10 ||
          result.q_safe[j] > command_box[j][1] + 1e-10)
        {
          return fallback("original_motion_limit");
        }
      }
      bool same_hand_geometry = accepted_ && !hand_held && !result.recovery_verified;
      if (same_hand_geometry) {
        for (const auto slot : kHandSlots) {
          if (!config_.hands_installed[slot / 10]) {continue;}
          same_hand_geometry = same_hand_geometry && target.hand[slot] == accepted_->hand[slot];
        }
      }
      double new_error = 0.0, hold_error = 0.0;
      bool preferred_reference = false;
      if (projected && same_hand_geometry) {
        for (size_t j = 0; j < 14; ++j) {
          new_error += std::pow(result.q_safe[j] - snapshot.requested.arm[j], 2);
          hold_error += std::pow(accepted_->arm[j] - snapshot.requested.arm[j], 2);
        }
        if (new_error > hold_error + 1e-12) {
          // Only this error comparison needs a fresh old-target report. Do not
          // spend a normal nominal frame's budget verifying a cached hold.
          const auto hold_report = run(*accepted_, 6.5, true, "hold");
          if (hold_report.deadline_missed || remaining() <= config_.guard_us) {
            ++diagnostics_.deadlines;
            return fallback("deadline");
          }
          if (hold_report.thumb_target_accepted && hold_report.nonlinear_verified &&
            same_arm(hold_report.q_safe, accepted_->arm))
          {
            // Non-root finger commands do not affect thumb geometry. Preserve
            // their updates and exactly verify this preferred arm below.
            std::copy_n(accepted_->arm.begin(), 14, result.q_safe.begin());
            preferred_reference = true;
          }
        }
      }
      decision.output = target;
      decision.output.arm.assign(result.q_safe.begin(), result.q_safe.end());
      // The kernel already checked this exact output against this snapshot and
      // command box. Only a reference replacement changes that proven target.
      const auto verified = preferred_reference ? run(decision.output, 6.5, true, "final") : result;
      if (projected && !preferred_reference) {++diagnostics_.verification_reuses;}
      if (verified.deadline_missed || remaining() <= config_.guard_us) {
        ++diagnostics_.deadlines;
        return fallback("deadline");
      }
      if (verified.thumb_target_accepted && verified.nonlinear_verified) {
        // A verified recovery may still have negative endpoint margin. Its
        // sampled non-worsening recovery constraints are the acceptance proof.
        decision.accept_new = true;
        decision.hand_held = hand_held;
        decision.modified = decision.output.arm != snapshot.requested.arm ||
          decision.output.hand != snapshot.requested.hand;
        decision.geometry_modified = hand_held || !same_arm(result.q_safe, bounded_arm);
        decision.recovery_verified = verified.recovery_verified;
        diagnostics_.minimum_distance_m = verified.output_min_distance_m;
        diagnostics_.distance_valid = true;
        return finish(preferred_reference ? "safe_reference_preferred" :
          (!projected && !verified.recovery_verified ? "verified_nominal_path" :
          x2::cbf::filter_status_name(result.status)));
      }
      result = verified;
    }
    // Only a geometry failure can defer a valid new gesture. Metadata/input
    // failures retain the full old pair and never authorize an arm-only retry.
    if (attempt == 0 && config_.enforce && accepted_ &&
      result.status != x2::cbf::FilterStatus::InvalidInputFallback &&
      target.hand != accepted_->hand && remaining() > config_.guard_us)
    {
      target.hand = accepted_->hand;
      hand_held = true;
      continue;
    }
    return fallback(x2::cbf::filter_status_name(result.status));
  }
  return fallback("candidate_path_rejected");
}

void CbfSafety::committed(const CbfDecision & decision, double now)
{
  if (config_.enforce && (decision.epoch != epoch_ || !diagnostics_.active)) {
    ++diagnostics_.withheld;
    return;
  }
  if (config_.enforce && decision.output.kind == Output::Kind::kNone) {
    ++diagnostics_.withheld;
    return;
  }
  if (config_.enforce && decision.output.kind == Output::Kind::kUpperBody) {
    ++diagnostics_.published;
    if (decision.accept_new) {
      accepted_ = decision.output;
      ++diagnostics_.accepted;
      if (decision.modified) {++diagnostics_.modified;}
      if (decision.geometry_modified) {++diagnostics_.geometry_modified;}
    } else {
      ++diagnostics_.held;
      // A held arm/hand pair may carry a newer independently controlled head.
      // Remember what was actually published so a later missing head cannot
      // roll that unrelated channel back; this does not accept a new arm/hand.
      if (accepted_ && decision.output.head.size() == 2 && finite(decision.output.head)) {
        accepted_->head = decision.output.head;
      }
    }
    last_published_ = now;
  }
}

CbfDecision CbfSafety::deadline_fallback(const CbfDecision & decision)
{
  if (!config_.enforce) {return decision;}
  ++diagnostics_.node_deadlines;
  return cached_fallback(decision, "node_deadline");
}

CbfDecision CbfSafety::feedback_fallback(const CbfDecision & decision, const std::string & reason)
{
  if (!config_.enforce) {return decision;}
  return cached_fallback(decision, "publication_" + reason);
}

CbfDecision CbfSafety::cached_fallback(const CbfDecision & decision, const std::string & reason)
{
  CbfDecision held;
  held.epoch = decision.epoch;
  // A decision prepared before an ownership/epoch exit cannot revive history.
  if (diagnostics_.active && decision.epoch == epoch_ && accepted_ &&
    decision.prepared_hold.kind == Output::Kind::kUpperBody &&
    decision.prepared_hold.arm == accepted_->arm && decision.prepared_hold.hand == accepted_->hand &&
    decision.prepared_hold.hand_sub_mode == accepted_->hand_sub_mode)
  {
    held.output = decision.prepared_hold;
    held.prepared_hold = decision.prepared_hold;
  }
  diagnostics_.reason = config_.mapping_available && !config_.mapping_verified ?
    "unverified_mapping/" + reason : reason;
  diagnostics_.distance_valid = false;
  diagnostics_.recovery_verified = false;
  diagnostics_.hand_held = false;
  return held;
}

void CbfSafety::tick_finished(double elapsed_us)
{
  diagnostics_.tick_us = elapsed_us;
  diagnostics_.max_tick_us = std::max(diagnostics_.max_tick_us, elapsed_us);
  if (elapsed_us > config_.deadline_us) {++diagnostics_.publish_overruns;}
}

void CbfSafety::discarded()
{
  ++diagnostics_.withheld;
  diagnostics_.reason = "stale_snapshot_or_epoch";
  diagnostics_.recovery_verified = false;
  diagnostics_.hand_held = false;
  diagnostics_.distance_valid = false;
}
}  // namespace aimdk_dataplane
