#pragma once

#include "cbf/distance_gradient.hpp"
#include "cbf/generated_constants.hpp"
#include "cbf/rigid_thumb.hpp"

#include <array>
#include <cstddef>
#include <optional>

namespace x2::cbf {

enum class FilterStatus {
  Normal,
  Projected,
  IterationLimitSafe,
  AirwallRecovery,
  PenetrationRecovery,
  DeadlineFallback,
  InvalidInputFallback,
  DegenerateGradientFallback,
  DegradedHold,
  ThumbMotionRejected,
};

struct SafetyFilterConfig {
  double activation_distance_m{0.080};
  double extra_clearance_m{0.002};
  double shoulder_torso_clearance_m{0.0};
  double cbf_gain{0.50};
  double max_step_rad{0.040};
  double gradient_squared_epsilon{1e-12};
  double verification_tolerance_m{1e-5};
  double improvement_epsilon_m{1e-6};
  double deadline_us{3000.0};
  double deadline_guard_us{500.0};
  int projection_iterations{12};
  int nonlinear_recovery_iterations{8};
  int path_check_substeps{4};
  int path_recovery_iterations{6};
  std::size_t max_active_pairs{16};
  bool use_path_separation_culling{true};
  // Live recovery may leave an already violated barrier without worsening it.
  // Existing callers retain strict initial-state admission unless opted in.
  bool allow_airwall_recovery{false};
};

struct SafetyFilterInput {
  std::array<double, 14> q_current{};
  std::array<double, 14> q_nominal{};
  std::array<double, 6> q_thumb_urdf{kRigidThumbBaselineUrdf};
  bool thumb_pose_valid{true};
  std::array<double, 6> q_thumb_current_urdf{kRigidThumbBaselineUrdf};
  bool thumb_current_valid{false};
  std::array<std::array<double, 2>, 4> thumb_joint_limits_urdf{kRigidThumbLimits};
  // Separate observation bounds; targets keep thumb_joint_limits_urdf strict.
  // Unset preserves existing callers, including custom target-model bounds.
  std::optional<std::array<std::array<double, 2>, 4>> current_thumb_joint_limits_urdf;
  std::array<bool, 2> hands_installed{true, true};
  // Observation domain may be wider than the commanded operator envelope.
  std::array<std::array<double, 2>, 14> current_arm_joint_limits = generated::kArmLimits;
  // Library callers may retain strict domains. Live encoder observations are
  // not commands; disable only their goal-range predicate, never finite checks.
  bool check_current_arm_goal_range{true};
  // Caller command box, normally the application limits intersected with its
  // last-published-target rate box. Never applied to measured FK coordinates.
  std::array<std::array<double, 2>, 14> command_joint_limits = generated::kArmLimits;
};

struct SafetyFilterResult {
  std::array<double, 14> q_safe{};
  FilterStatus status{FilterStatus::InvalidInputFallback};
  double input_min_distance_m{0.0};
  double output_min_distance_m{0.0};
  double input_min_safety_margin_m{0.0};
  double output_min_safety_margin_m{0.0};
  double elapsed_us{0.0};
  std::size_t input_min_pair{0};
  std::size_t output_min_pair{0};
  std::size_t active_pair_count{0};
  int projection_iterations{0};
  bool nonlinear_verified{false};
  bool deadline_missed{false};
  bool thumb_target_accepted{false};
  // The sampled path obeys state-relative non-worsening recovery constraints;
  // the endpoint may still be inside an initially violated clearance barrier.
  bool recovery_verified{false};
  // First recovery-path violation, used for bounded endpoint Jacobian repair.
  std::size_t path_violation_pair{kPrimitivePairCount};
  double path_violation_fraction{0.0};
  double path_violation_previous_fraction{0.0};
  double path_violation_deficit_m{0.0};
  // Diagnostics only, populated at an existing invalid-input return site.
  const char* invalid_input_reason{""};
  int invalid_input_index{-1};
  double invalid_input_value{0.0};
  double invalid_input_lower{0.0};
  double invalid_input_upper{0.0};
};

SafetyFilterResult filter_arm_target(
    const SafetyFilterInput& input,
    const SafetyFilterConfig& config = SafetyFilterConfig{});

// Verify exactly this position target and its sampled path; never optimize or
// alter a previously published hold target. Uses the same geometry/clearances.
SafetyFilterResult verify_arm_target(
    const SafetyFilterInput& input,
    const SafetyFilterConfig& config = SafetyFilterConfig{});

// Geometric ordering check only: can the requested thumb move while the arms
// remain at their measured coordinates? The returned arm is NOT a command.
SafetyFilterResult verify_thumb_target_with_arm_held(
    const SafetyFilterInput& input,
    const SafetyFilterConfig& config = SafetyFilterConfig{});

const char* filter_status_name(FilterStatus status);

}  // namespace x2::cbf
