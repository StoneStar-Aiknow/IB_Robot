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
  // A refusal that is not a collision verdict (data, budget, authorization) may
  // hold the committed pair only this long; then the request is published within
  // the original joint and velocity limits, as if the filter were disabled.
  double hold_release_after_s = 0.2;
  // A hold that saw a collision verdict is released too, later and slower: after
  // collision_release_after_s when it began on a published pair (the arm waits at
  // the boundary and a move-away is accepted at once), after
  // collision_release_without_history_s when nothing had been published yet (the
  // hold publishes nothing). The arm step is scaled by collision_release_speed_scale.
  double collision_release_after_s = 2.0;
  double collision_release_without_history_s = 0.5;
  double collision_release_speed_scale = 0.3;
  double period_s = 0.02;
  bool mapping_verified = false;
  bool mapping_available = false;
  bool native_limits_available = false;
  // Hands whose collision geometry is checked. Explicit, never inferred from
  // missing feedback. An installed side may be an empty flange (platform type 0):
  // checked as a phantom hand. A hand on a side configured without one is a mismatch.
  std::array<bool, 2> hands_installed{true, true};
  std::array<double, 4> native_lower{};
  std::array<double, 4> native_upper{};
  std::array<std::array<double, 2>, 4> thumb_model_limits{};
  std::array<double, 4> command_scale{1.0, 1.0, 1.0, 1.0};
  std::array<double, 4> command_offset{};
  std::array<double, 4> feedback_scale{1.0, 1.0, 1.0, 1.0};
  std::array<double, 4> feedback_offset{};
  // Native command angles (L roll, L abad, R roll, R abad) of a phantom hand's
  // thumb, used as its target, its current angle and its published command.
  // Default: the most tucked-in pose in the command band, model roll at its
  // upper and abad at its lower end; with the rigid model band and an identity
  // mapping that is the value below. FK over arm poses whose palm is near the
  // body or the other arm found it farthest from them of the band ends and
  // the baseline. cbf_safety_json may set phantom_thumb_native: [[L roll,
  // L abad], [R roll, R abad]] inside the command band.
  std::array<double, 4> phantom_thumb_native{1.27409035, -1.6424, 0.69813170, -1.464235482612666};
  std::array<double, 14> lower{};
  std::array<double, 14> upper{};
  std::array<double, 14> velocity{};
  std::array<std::array<double, 2>, 14> arm_feedback_limits{};

  static CbfSafetyConfig from_json(const std::string & text, double rate_hz);
  bool matches_hand_types(const std::vector<int> & types) const;
  std::string feedback_error(const CbfSnapshot & snapshot) const;
};

// What a thumb axis (0..3 = L roll, L abad, R roll, R abad) can be commanded to,
// in command values: its native range, else the model range.
std::array<double, 2> cbf_thumb_command_band(const CbfSafetyConfig & config, size_t axis);

struct CbfSnapshot
{
  Output requested;
  uint64_t epoch = 0;
  bool named_arm_cache = false;
  bool source_ready = true;
  bool hand_configuration_matches = true;
  std::optional<Measured> arm;
  // Installed sides only: L roll/abad then R roll/abad, native radians. A side
  // without readings is NaN there; the other side's readings are still used.
  std::optional<Measured> thumb;
  // Per side, left/right: that side's thumb roots reported a nonzero hand
  // faultcode. Only that side is replaced by the command.
  std::array<bool, 2> thumb_faulted{};
  // Per side: configured as installed while the platform reports no hand (SDK
  // type 0), i.e. an empty flange. Its geometry is still checked, as a phantom
  // hand with its thumb at phantom_thumb_native; this is no feedback problem.
  std::array<bool, 2> hand_reported_absent{};
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
  // The kernel judged the request unsafe. These holds are released later and
  // slower than others; a target that does not approach a collision is never
  // refused this way.
  bool collision_refusal = false;
  bool released = false;  // Published by release_if_stuck, not verified by the filter.
  Output prepared_hold;
};

// Thumb feedback is advisory: its first problem in an episode, for logging only.
struct CbfThumbFeedbackEvent
{
  uint64_t event = 0;
  double at_s = 0.0;
  std::string issue;  // out_of_range, unavailable, stale, faulted
  int axis = -1;  // 0..3 = L roll, L abad, R roll, R abad
  double value = 0.0;  // native radians as reported
  double lower = 0.0;
  double upper = 0.0;
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
  // The part of a tick after the node's deadline check (publication, commit):
  // an overrun here cannot be caused by the filter and does not hold a frame.
  double after_check_us = 0.0;
  double max_after_check_us = 0.0;
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
  uint64_t thumb_feedback_frames = 0;  // Frames that clamped or replaced the thumb reading.
  std::string thumb_feedback_issue;  // This frame's issue, empty when the reading was used as is.
  std::optional<CbfThumbFeedbackEvent> last_thumb_feedback_event;
  uint64_t released = 0;
  uint64_t played = 0;  // Frames published by the template player, not by the filter.
  uint64_t release_episodes = 0;
  uint64_t collision_release_episodes = 0;  // Episodes that released after a collision verdict.
  uint64_t a_cap_capped_ticks = 0;  // Released frames whose arm and hand were capped to the frozen point.
  bool releasing = false;
  bool stuck_collision = false;  // The current hold has seen a collision verdict.
  double stuck_s = 0.0;  // Length of the current hold.
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
  // Last step before publication. Every hold ends: see the release times in the config.
  // ``a_cap``: a template A segment holds the cap on this tick. Then the arm and
  // hand are released towards the request of the hold's first tick under the cap
  // (the frozen point), not the latest one; the head follows the latest request.
  // When, how fast and within which limits the release goes is unchanged.
  CbfDecision release_if_stuck(
    const CbfDecision & decision, const CbfSnapshot & snapshot, double now, bool a_cap = false);
  // A frame the template player published instead of the filter. Under enforce it
  // becomes the committed pair of the same epoch, so the filter resumes from where
  // the arm is; it is never verified and ends any hold. A frame of another epoch,
  // or one played while mirroring is not active, is only counted.
  void played(const Output & output, uint64_t epoch, double now);
  void tick_finished(double elapsed_us, double after_check_us = 0.0);
  void discarded();
  void inactive(uint64_t epoch);
  // Tests only: after this many kernel calls in a frame its budget reads as spent.
  void limit_kernel_calls_for_test(int calls) {kernel_call_limit_ = calls;}

private:
  CbfDecision cached_fallback(const CbfDecision & decision, const std::string & reason);
  CbfSafetyConfig config_;
  CbfDiagnostics diagnostics_;
  uint64_t epoch_ = 0;
  std::optional<Output> accepted_;
  double last_published_ = 0.0;
  bool invalid_input_episode_ = false;
  std::optional<double> stuck_since_;
  bool stuck_with_history_ = false;  // The hold began on a published pair.
  bool stuck_released_ = false;  // The hold has begun to release.
  std::optional<Output> a_cap_point_;  // This hold's frozen point, while the cap holds.
  int kernel_call_limit_ = -1;
};

}  // namespace aimdk_dataplane
#endif  // AIMDK_DATAPLANE__CBF_SAFETY_HPP_
