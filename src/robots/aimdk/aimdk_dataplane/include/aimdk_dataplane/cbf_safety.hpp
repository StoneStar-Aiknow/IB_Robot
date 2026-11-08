// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef AIMDK_DATAPLANE__CBF_SAFETY_HPP_
#define AIMDK_DATAPLANE__CBF_SAFETY_HPP_

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "aimdk_dataplane/core.hpp"

namespace aimdk_dataplane
{

struct CbfSnapshot;

struct CbfSafetyConfig
{
  bool enabled = false;
  bool enforce = false;
  std::string mirror_mode;
  double deadline_us = 10000.0;
  double guard_us = 500.0;
  double feedback_max_age_s = 0.1;
  double state_skew_max_s = 0.04;
  // Sensor/tracking allowance only. Never added to command or clearance limits.
  double feedback_limit_tolerance_rad = 0.0;
  bool check_arm_feedback_goal_range = true;
  double period_s = 0.02;
  bool mapping_verified = false;
  bool mapping_available = false;
  bool native_limits_available = false;
  std::array<bool, 2> hands_installed{true, true};  // Explicit hardware, never inferred from missing feedback.
  std::array<double, 4> native_lower{};
  std::array<double, 4> native_upper{};
  std::array<std::array<double, 2>, 4> thumb_model_limits{};
  std::array<double, 4> command_scale{1.0, 1.0, 1.0, 1.0};
  std::array<double, 4> command_offset{};
  std::array<double, 4> feedback_scale{1.0, 1.0, 1.0, 1.0};
  std::array<double, 4> feedback_offset{};
  std::array<double, 14> lower{};
  std::array<double, 14> upper{};
  std::array<double, 14> velocity{};
  std::array<std::array<double, 2>, 14> arm_feedback_limits{};

  static CbfSafetyConfig from_json(const std::string & text, double rate_hz);
  bool matches_hand_types(const std::vector<int> & types) const;
  std::string feedback_error(const CbfSnapshot & snapshot) const;
};

struct CbfSnapshot
{
  Output requested;
  uint64_t epoch = 0;
  bool named_arm_cache = false;
  bool source_ready = true;
  bool hand_configuration_matches = true;
  std::optional<Measured> arm;
  std::optional<Measured> thumb;  // Installed sides only: L roll/abad then R roll/abad, native radians.
  // Shared with the publishing tick. Numerical callers may omit this and use
  // the configured relative budget; wall-clock/ROS time must never be used here.
  std::optional<std::chrono::steady_clock::time_point> publication_deadline;
};

struct CbfDecision
{
  Output output;
  uint64_t epoch = 0;
  bool accept_new = false;
  bool modified = false;
  bool geometry_modified = false;
  bool recovery_verified = false;
  bool hand_held = false;
  Output prepared_hold;
};

struct CbfInputFault
{
  uint64_t event = 0;
  uint64_t call = 0;
  uint64_t epoch = 0;
  uint32_t command_sequence = 0;
  double at_s = 0.0;
  std::string stage;
  std::string issue;
  int index = -1;
  double value = 0.0;
  double lower = 0.0;
  double upper = 0.0;
  std::array<double, 14> arm_current{};
  std::array<double, 14> arm_target{};
  std::array<std::array<double, 2>, 14> arm_feedback_limits{};
  std::vector<double> arm_requested;
  std::array<double, 6> thumb_current_urdf{};
  std::array<double, 6> thumb_target_urdf{};
  std::array<std::array<double, 2>, 4> thumb_limits_urdf{};
  std::array<std::array<double, 2>, 4> thumb_current_limits_urdf{};
  std::vector<double> thumb_native_current;
  std::array<double, 4> thumb_native_target{};
  double arm_age_s = 0.0;
  double thumb_age_s = 0.0;
};

struct CbfDiagnostics
{
  uint64_t calls = 0;
  uint64_t accepted = 0;
  uint64_t published = 0;
  uint64_t modified = 0;
  uint64_t geometry_modified = 0;
  uint64_t held = 0;
  uint64_t withheld = 0;
  uint64_t deadlines = 0;
  uint64_t node_deadlines = 0;
  uint64_t publish_overruns = 0;
  double elapsed_us = 0.0;
  double max_elapsed_us = 0.0;
  double tick_us = 0.0;
  double max_tick_us = 0.0;
  double available_budget_us = 0.0;
  double hand_check_us = 0.0;
  double verification_us = 0.0;
  double projection_us = 0.0;
  double max_hand_check_us = 0.0;
  double max_verification_us = 0.0;
  double max_projection_us = 0.0;
  uint64_t verification_reuses = 0;
  uint64_t hand_budget_holds = 0;
  double minimum_distance_m = 0.0;
  bool distance_valid = false;
  bool active = false;
  bool recovery_verified = false;
  bool hand_held = false;
  bool thumb_angles_valid = false;
  double arm_age_s = -1.0;
  double thumb_age_s = -1.0;
  std::array<double, 4> thumb_current_urdf{};
  std::array<double, 4> thumb_target_urdf{};
  uint64_t invalid_input_frames = 0;
  std::optional<CbfInputFault> last_input_fault;
  std::string reason = "inactive";
};

// No ROS or vendor types. Only the caller may commit a successfully published frame.
class CbfSafety
{
public:
  explicit CbfSafety(CbfSafetyConfig config);
  const CbfSafetyConfig & config() const {return config_;}
  const CbfDiagnostics & diagnostics() const {return diagnostics_;}
  CbfDecision filter(const CbfSnapshot & snapshot, double now);
  void committed(const CbfDecision & decision, double now);
  CbfDecision deadline_fallback(const CbfDecision & decision);
  CbfDecision feedback_fallback(const CbfDecision & decision, const std::string & reason);
  void tick_finished(double elapsed_us);
  void discarded();
  void inactive(uint64_t epoch);

private:
  CbfDecision cached_fallback(const CbfDecision & decision, const std::string & reason);
  CbfSafetyConfig config_;
  CbfDiagnostics diagnostics_;
  uint64_t epoch_ = 0;
  std::optional<Output> accepted_;
  double last_published_ = 0.0;
  bool invalid_input_episode_ = false;
};

}  // namespace aimdk_dataplane
#endif  // AIMDK_DATAPLANE__CBF_SAFETY_HPP_
