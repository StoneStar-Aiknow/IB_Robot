#include "cbf/safety_filter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>

namespace x2::cbf {
namespace {

using Clock = std::chrono::steady_clock;

struct LinearConstraint {
  ArmGradient gradient{};
  double lower_bound{0.0};
  double priority{0.0};
  std::size_t pair_index{0};
};

double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

bool deadline_budget_exhausted(Clock::time_point start,
                               const SafetyFilterConfig& config) {
  const double guard = std::clamp(config.deadline_guard_us, 0.0,
                                  std::max(0.0, config.deadline_us));
  return elapsed_us(start) >= config.deadline_us - guard;
}

bool finite_array(const std::array<double, 14>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool finite_array(const std::array<double, 6>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

using ArmBounds = std::array<std::array<double, 2>, 14>;

bool valid_command_limits(const ArmBounds& bounds) {
  for (std::size_t j = 0; j < bounds.size(); ++j) {
    if (!std::isfinite(bounds[j][0]) || !std::isfinite(bounds[j][1]) ||
        bounds[j][0] > bounds[j][1] || bounds[j][0] < generated::kArmLimits[j][0] ||
        bounds[j][1] > generated::kArmLimits[j][1]) return false;
  }
  return true;
}

std::array<double, 14> clamp_to_limits(std::array<double, 14> values,
                                      const ArmBounds& bounds = generated::kArmLimits) {
  const bool valid = valid_command_limits(bounds);
  for (std::size_t joint = 0; joint < values.size(); ++joint) {
    values[joint] = std::clamp(values[joint],
                               valid ? bounds[joint][0] : generated::kArmLimits[joint][0],
                               valid ? bounds[joint][1] : generated::kArmLimits[joint][1]);
  }
  return values;
}

bool measured_arm_in_domain(const std::array<double, 14>& values,
                             const std::array<std::array<double, 2>, 14>& bounds,
                             bool check_goal_range) {
  if (!finite_array(values)) return false;
  for (std::size_t joint = 0; joint < values.size(); ++joint) {
    const auto band = bounds[joint];
    if (!std::isfinite(band[0]) || !std::isfinite(band[1]) || band[0] > band[1] ||
        !std::isfinite(band[1] - band[0]) ||
        (check_goal_range && (values[joint] < band[0] || values[joint] > band[1]))) return false;
  }
  return true;
}

enum class InvalidInputSite { Filter, FixedThumb, Verify };

const std::array<std::array<double, 2>, 4>& current_thumb_limits(const SafetyFilterInput& input) {
  return input.current_thumb_joint_limits_urdf ? *input.current_thumb_joint_limits_urdf :
      input.thumb_joint_limits_urdf;
}

void input_issue(SafetyFilterResult& result, const char* reason, int index = -1,
                 double value = 0.0, double lower = 0.0, double upper = 0.0) {
  result.invalid_input_reason = reason;
  result.invalid_input_index = index;
  result.invalid_input_value = value;
  result.invalid_input_lower = lower;
  result.invalid_input_upper = upper;
}

bool diagnose_arm(SafetyFilterResult& result,
                  const std::array<double, 14>& values,
                  const char* nonfinite, const char* outside, bool check_range,
                  const std::array<std::array<double, 2>, 14>& bounds = generated::kArmLimits) {
  // Check the whole array for finiteness before diagnosing the range predicate.
  for (std::size_t joint = 0; joint < values.size(); ++joint) {
    if (!std::isfinite(values[joint])) {
      input_issue(result, nonfinite, static_cast<int>(joint), values[joint],
                  bounds[joint][0], bounds[joint][1]);
      return true;
    }
  }
  {
    for (std::size_t joint = 0; joint < values.size(); ++joint) {
      const auto band = bounds[joint];
      if (!std::isfinite(band[0]) || !std::isfinite(band[1]) || band[0] > band[1] ||
          !std::isfinite(band[1] - band[0])) {
        input_issue(result, "current_arm_limits_invalid", static_cast<int>(joint),
                    values[joint], band[0], band[1]);
        return true;
      }
      if (check_range && (values[joint] < band[0] || values[joint] > band[1])) {
        input_issue(result, outside, static_cast<int>(joint), values[joint], band[0], band[1]);
        return true;
      }
    }
  }
  return false;
}

bool diagnose_thumb(SafetyFilterResult& result,
                    const std::array<double, 6>& pose,
                    const std::array<std::array<double, 2>, 4>& bounds,
                    const char* nonfinite, const char* outside) {
  // Mirror valid_rigid_thumb_pose exactly, including its unchanged tolerance.
  for (std::size_t axis = 0; axis < kRigidThumbIndices.size(); ++axis) {
    const auto slot = kRigidThumbIndices[axis];
    const double lower = bounds[axis][0], upper = bounds[axis][1];
    const double span = upper - lower;
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper ||
        !std::isfinite(span) || span > 6.283185307179586) {
      const double value = !std::isfinite(lower) ? lower :
          (!std::isfinite(upper) ? upper : span);
      input_issue(result, "thumb_limits_invalid", static_cast<int>(slot), value, lower, upper);
      return true;
    }
    if (!std::isfinite(pose[slot])) {
      input_issue(result, nonfinite, static_cast<int>(slot), pose[slot], lower, upper);
      return true;
    }
    if (pose[slot] < lower - 1e-12 || pose[slot] > upper + 1e-12) {
      input_issue(result, outside, static_cast<int>(slot), pose[slot], lower, upper);
      return true;
    }
  }
  return false;
}

void diagnose_invalid_input(SafetyFilterResult& result,
                            const SafetyFilterInput& input,
                            const SafetyFilterConfig& config, InvalidInputSite site) {
  // Called only AFTER an unchanged existing condition has rejected the input.
  const bool fixed = site == InvalidInputSite::FixedThumb;
  const bool verify = site == InvalidInputSite::Verify;
  if (!valid_command_limits(input.command_joint_limits)) {
    input_issue(result, "command_arm_limits_invalid");
    return;
  }
  if (diagnose_arm(result, input.q_current, "current_arm_nonfinite", "current_arm_out_of_range",
                   !fixed && input.check_current_arm_goal_range,
                   input.current_arm_joint_limits)) return;
  if ((verify || fixed) && diagnose_arm(result, input.q_nominal,
      "nominal_arm_nonfinite", "nominal_arm_out_of_range", verify, input.command_joint_limits)) return;
  if (fixed) {
    for (std::size_t slot = 0; slot < input.q_thumb_urdf.size(); ++slot) {
      if (!std::isfinite(input.q_thumb_urdf[slot])) {
        input_issue(result, "target_thumb_nonfinite", static_cast<int>(slot), input.q_thumb_urdf[slot]);
        return;
      }
    }
  }
  if (!input.thumb_pose_valid) {input_issue(result, "thumb_pose_invalid"); return;}
  if (!fixed) {
    if (diagnose_thumb(result, input.q_thumb_current_urdf, current_thumb_limits(input),
                       "current_thumb_nonfinite", "current_thumb_out_of_range")) return;
    if (diagnose_thumb(result, input.q_thumb_urdf, input.thumb_joint_limits_urdf,
                       "target_thumb_nonfinite", "target_thumb_out_of_range")) return;
    if (!verify && diagnose_arm(result, input.q_nominal,
        "nominal_arm_nonfinite", "nominal_arm_out_of_range", false)) return;
  }
  // Each call site retains its original, different config validation expression.
  // Reaching here after rejection means that site's config predicate failed.
  (void)config;
  input_issue(result, "config_invalid");
}

std::array<double, 14> clamp_to_step(
    std::array<double, 14> values, const std::array<double, 14>& current,
    double max_step_rad, const ArmBounds& bounds = generated::kArmLimits) {
  const double step = std::max(0.0, max_step_rad);
  for (std::size_t joint = 0; joint < values.size(); ++joint) {
    values[joint] = std::clamp(values[joint], current[joint] - step,
                               current[joint] + step);
  }
  return clamp_to_limits(values, bounds);
}

std::array<double, 14> interpolate_arm(
    const std::array<double, 14>& first,
    const std::array<double, 14>& second, double amount) {
  std::array<double, 14> result{};
  for (std::size_t joint = 0; joint < result.size(); ++joint) {
    result[joint] = first[joint] + amount * (second[joint] - first[joint]);
  }
  return result;
}

double dot_gradient_delta(const ArmGradient& gradient,
                          const std::array<double, 14>& candidate,
                          const std::array<double, 14>& current) {
  double value = 0.0;
  for (std::size_t joint = 0; joint < gradient.size(); ++joint) {
    value += gradient[joint] * (candidate[joint] - current[joint]);
  }
  return value;
}

double maximum_array_difference(const std::array<double, 14>& first,
                                const std::array<double, 14>& second) {
  double maximum = 0.0;
  for (std::size_t joint = 0; joint < first.size(); ++joint) {
    maximum = std::max(maximum, std::abs(first[joint] - second[joint]));
  }
  return maximum;
}

double pair_required_clearance(const KinematicsOutput& kinematics,
                               std::size_t pair_index,
                               const SafetyFilterConfig& config) {
  const PrimitivePair pair = kPrimitivePairs[pair_index];
  const ProxyPrimitive& first =
      kinematics.primitives[static_cast<std::size_t>(pair.first)];
  const ProxyPrimitive& second =
      kinematics.primitives[static_cast<std::size_t>(pair.second)];
  const bool shoulder_torso =
      pair.second == PrimitiveId::Torso &&
      (pair.first == PrimitiveId::LeftShoulder ||
       pair.first == PrimitiveId::RightShoulder);
  if (shoulder_torso) {
    return std::max(0.0, config.shoulder_torso_clearance_m);
  }
  return std::max(first.margin, second.margin) +
         std::max(0.0, config.extra_clearance_m);
}

double minimum_safety_margin(const KinematicsOutput& kinematics,
                             const PrimitiveDistanceReport& distances,
                             const SafetyFilterConfig& config,
                             std::size_t& minimum_pair) {
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t pair_index = 0; pair_index < kPrimitivePairCount;
       ++pair_index) {
    if (!proxy_pair_enabled(kinematics, pair_index)) continue;
    const double margin = distances.distances[pair_index].signed_distance -
                          pair_required_clearance(kinematics, pair_index, config);
    if (margin < minimum) {
      minimum = margin;
      minimum_pair = pair_index;
    }
  }
  return minimum;
}

KinematicsInput make_kinematics_input(
    const std::array<double, 14>& q_arm,
    const SafetyFilterInput& input);

struct SafetyEvaluation {
  KinematicsOutput kinematics{};
  PrimitiveDistanceReport distances{};
  double minimum_safety_margin{0.0};
  std::size_t minimum_pair{0};
  bool safe{false};
};

SafetyEvaluation evaluate_safety(const std::array<double, 14>& arm,
                                 const SafetyFilterInput& input,
                                 const SafetyFilterConfig& config) {
  SafetyEvaluation evaluation;
  evaluation.kinematics =
      forward_kinematics(make_kinematics_input(arm, input));
  evaluation.distances = evaluate_proxy_distances(evaluation.kinematics);
  evaluation.minimum_safety_margin = minimum_safety_margin(
      evaluation.kinematics, evaluation.distances, config,
      evaluation.minimum_pair);
  evaluation.safe = evaluation.distances.minimum_distance >= 0.0 &&
                    evaluation.minimum_safety_margin >=
                        -config.verification_tolerance_m;
  return evaluation;
}

Vec3 rotation_column(const Mat3& rotation, std::size_t column) {
  return {rotation.value[0][column], rotation.value[1][column],
          rotation.value[2][column]};
}

double projection_radius(const ProxyPrimitive& primitive, const Vec3& axis) {
  if (primitive.kind == PrimitiveKind::Sphere) return primitive.radius;
  if (primitive.kind == PrimitiveKind::Capsule) {
    return primitive.radius + 0.5 * primitive.length *
        std::abs(dot(rotation_column(primitive.pose.rotation, 2), axis));
  }
  const Vec3 half = primitive.size * 0.5;
  return half.x * std::abs(dot(rotation_column(primitive.pose.rotation, 0), axis)) +
         half.y * std::abs(dot(rotation_column(primitive.pose.rotation, 1), axis)) +
         half.z * std::abs(dot(rotation_column(primitive.pose.rotation, 2), axis));
}

bool box_axis_proves_clearance(const ProxyPrimitive& first,
                               const ProxyPrimitive& second,
                               const ProxyPrimitive& box, double threshold) {
  const Vec3 centers = second.pose.translation - first.pose.translation;
  for (std::size_t column = 0; column < 3; ++column) {
    const Vec3 axis = normalized(rotation_column(box.pose.rotation, column));
    const double gap = std::abs(dot(centers, axis)) -
        projection_radius(first, axis) - projection_radius(second, axis);
    if (gap >= threshold + 1e-9) return true;
  }
  return false;
}

bool path_sample_is_safe(const std::array<double, 14>& arm,
                         const SafetyFilterInput& input,
                         const SafetyFilterConfig& config, double path_guard) {
  const KinematicsOutput kinematics =
      forward_kinematics(make_kinematics_input(arm, input));
  if (!kinematics.valid) return false;
  for (std::size_t pair_index = 0; pair_index < kPrimitivePairCount; ++pair_index) {
    if (!proxy_pair_enabled(kinematics, pair_index)) continue;
    const PrimitivePair pair = kPrimitivePairs[pair_index];
    const auto& first = kinematics.primitives[static_cast<std::size_t>(pair.first)];
    const auto& second = kinematics.primitives[static_cast<std::size_t>(pair.second)];
    const double required = pair_required_clearance(kinematics, pair_index, config);
    const double threshold = required + path_guard;
    const bool first_box = first.kind == PrimitiveKind::Box;
    const bool second_box = second.kind == PrimitiveKind::Box;
    const bool costly_pair = (first_box || second_box) &&
        first.kind != PrimitiveKind::Sphere &&
        second.kind != PrimitiveKind::Sphere;
    if (costly_pair &&
        ((first_box && box_axis_proves_clearance(first, second, first, threshold)) ||
         (second_box && box_axis_proves_clearance(first, second, second, threshold)))) {
      continue;
    }
    const DistanceResult distance = distance_between(first, second);
    if (!distance.valid || !std::isfinite(distance.signed_distance) ||
        distance.signed_distance < path_guard ||
        distance.signed_distance - required < path_guard) {
      return false;
    }
  }
  return true;
}

void insert_constraint(
    const LinearConstraint& constraint,
    std::array<LinearConstraint, kPrimitivePairCount>& constraints,
    std::size_t& count, std::size_t capacity) {
  capacity = std::min(capacity, constraints.size());
  if (capacity == 0) return;
  std::size_t position = count;
  if (count < capacity) {
    ++count;
  } else if (constraint.priority >= constraints[count - 1].priority) {
    return;
  } else {
    position = count - 1;
  }
  while (position > 0 &&
         constraint.priority < constraints[position - 1].priority) {
    if (position < capacity) constraints[position] = constraints[position - 1];
    --position;
  }
  constraints[position] = constraint;
}

KinematicsInput make_kinematics_input(
    const std::array<double, 14>& q_arm,
    const SafetyFilterInput& input) {
  KinematicsInput kinematics_input;
  kinematics_input.q_arm = q_arm;
  kinematics_input.q_thumb_urdf = input.q_thumb_urdf;
  kinematics_input.thumb_pose_valid = input.thumb_pose_valid;
  kinematics_input.hands_installed = input.hands_installed;
  return kinematics_input;
}

SafetyFilterResult hold_result(
    SafetyFilterResult result, const std::array<double, 14>& current,
    const SafetyFilterInput& input, const SafetyFilterConfig& config,
    FilterStatus status, Clock::time_point start) {
  result.q_safe = clamp_to_limits(current, input.command_joint_limits);  // Geometry still uses raw feedback below.
  const KinematicsOutput output_kinematics =
      forward_kinematics(make_kinematics_input(current, input));
  const PrimitiveDistanceReport output_distances =
      evaluate_proxy_distances(output_kinematics);
  result.output_min_distance_m = output_distances.minimum_distance;
  result.output_min_pair = output_distances.minimum_pair_index;
  result.output_min_safety_margin_m = minimum_safety_margin(
      output_kinematics, output_distances, config, result.output_min_pair);
  result.nonlinear_verified =
      result.output_min_distance_m >= 0.0 &&
      result.output_min_safety_margin_m >= -config.verification_tolerance_m;
  result.status = status;
  result.elapsed_us = elapsed_us(start);
  return result;
}

SafetyFilterResult hold_result_from_report(
    SafetyFilterResult result, const std::array<double, 14>& current,
    const DistanceGradientReport& current_report,
    const SafetyFilterConfig& config, FilterStatus status,
    Clock::time_point start, const ArmBounds& bounds = generated::kArmLimits) {
  result.q_safe = clamp_to_limits(current, bounds);
  result.output_min_distance_m = current_report.distances.minimum_distance;
  result.output_min_pair = current_report.distances.minimum_pair_index;
  result.output_min_safety_margin_m = minimum_safety_margin(
      current_report.kinematics, current_report.distances, config,
      result.output_min_pair);
  result.nonlinear_verified =
      result.output_min_distance_m >= 0.0 &&
      result.output_min_safety_margin_m >= -config.verification_tolerance_m;
  result.status = status;
  result.elapsed_us = elapsed_us(start);
  return result;
}

}  // namespace

static SafetyFilterResult filter_arm_target_fixed_thumb(
    const SafetyFilterInput& input, const SafetyFilterConfig& config,
    Clock::time_point start) {
  SafetyFilterResult result;

  std::array<double, 14> current = input.q_current;
  if (!finite_array(current)) current = generated::kNaturalStanding;
  if (!measured_arm_in_domain(input.q_current, input.current_arm_joint_limits, input.check_current_arm_goal_range)) {
    current = clamp_to_limits(current, input.command_joint_limits);
  }

  if (!finite_array(input.q_current) || !finite_array(input.q_nominal) ||
      !finite_array(input.q_thumb_urdf) || !input.thumb_pose_valid ||
      !valid_command_limits(input.command_joint_limits) ||
      config.projection_iterations < 0 || config.deadline_us <= 0.0) {
    diagnose_invalid_input(result, input, config, InvalidInputSite::FixedThumb);
    return hold_result(result, current, input, config,
                       FilterStatus::InvalidInputFallback, start);
  }

  const DistanceGradientReport current_report =
      evaluate_distance_gradients(make_kinematics_input(current, input));
  result.input_min_distance_m = current_report.distances.minimum_distance;
  result.input_min_pair = current_report.distances.minimum_pair_index;
  result.input_min_safety_margin_m = minimum_safety_margin(
      current_report.kinematics, current_report.distances, config,
      result.input_min_pair);
  if (!current_report.valid) {
    input_issue(result, "distance_gradient_invalid");
    for (std::size_t index = 0; index < current_report.pairs.size(); ++index) {
      if (!current_report.pairs[index].valid) {
        input_issue(result, "distance_gradient_invalid", static_cast<int>(index));
        break;
      }
    }
    return hold_result(result, current, input, config,
                       FilterStatus::InvalidInputFallback, start);
  }
  if (deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return hold_result_from_report(result, current, current_report, config,
                                   FilterStatus::DeadlineFallback, start);
  }

  std::array<double, 14> candidate =
      clamp_to_step(clamp_to_limits(input.q_nominal, input.command_joint_limits), current,
                    config.max_step_rad, input.command_joint_limits);
  std::array<LinearConstraint, kPrimitivePairCount> constraints{};
  std::size_t constraint_count = 0;
  bool has_degenerate_active_pair = false;

  for (std::size_t pair_index = 0; pair_index < kPrimitivePairCount;
       ++pair_index) {
    if (!proxy_pair_enabled(current_report.kinematics, pair_index)) continue;
    const PairDistanceGradient& pair = current_report.pairs[pair_index];
    const double required = pair_required_clearance(
        current_report.kinematics, pair_index, config);
    const double safety_margin = pair.distance.signed_distance - required;
    const double predicted_distance = pair.distance.signed_distance +
        dot_gradient_delta(pair.gradient, candidate, current);
    if (pair.distance.signed_distance >= config.activation_distance_m &&
        predicted_distance >= config.activation_distance_m) {
      continue;
    }
    if (!pair.valid ||
        pair.gradient_squared_norm <= config.gradient_squared_epsilon) {
      if (safety_margin < 0.0 ||
          predicted_distance < config.activation_distance_m) {
        has_degenerate_active_pair = true;
      }
      continue;
    }
    LinearConstraint constraint;
    constraint.gradient = pair.gradient;
    constraint.lower_bound = -std::clamp(config.cbf_gain, 0.0, 1.0) *
                             safety_margin;
    constraint.priority = std::min(safety_margin, predicted_distance - required);
    constraint.pair_index = pair_index;
    insert_constraint(constraint, constraints, constraint_count,
                      config.max_active_pairs);
  }
  result.active_pair_count = constraint_count;

  if (has_degenerate_active_pair && result.input_min_safety_margin_m < 0.0) {
    return hold_result(result, current, input, config,
                       FilterStatus::DegenerateGradientFallback, start);
  }

  bool changed = false;
  bool converged = constraint_count == 0;
  for (int iteration = 0;
       iteration < config.projection_iterations && !converged; ++iteration) {
    if (deadline_budget_exhausted(start, config)) {
      result.deadline_missed = true;
      return hold_result_from_report(result, current, current_report, config,
                                     FilterStatus::DeadlineFallback, start);
    }
    converged = true;
    double worst_violation = 0.0;
    std::size_t worst_constraint = 0;
    for (std::size_t index = 0; index < constraint_count; ++index) {
      const double achieved = dot_gradient_delta(
          constraints[index].gradient, candidate, current);
      const double violation = constraints[index].lower_bound - achieved;
      if (violation > worst_violation) {
        worst_violation = violation;
        worst_constraint = index;
        converged = false;
      }
    }
    if (!converged) {
      const LinearConstraint& constraint = constraints[worst_constraint];
      double denominator = 0.0;
      for (double value : constraint.gradient) denominator += value * value;
      if (denominator <= config.gradient_squared_epsilon) {
        return hold_result(result, current, input, config,
                           FilterStatus::DegenerateGradientFallback, start);
      }
      const double scale = worst_violation / denominator;
      for (std::size_t joint = 0; joint < candidate.size(); ++joint) {
        candidate[joint] += scale * constraint.gradient[joint];
      }
      candidate = clamp_to_step(candidate, current, config.max_step_rad, input.command_joint_limits);
      changed = true;
      result.projection_iterations = iteration + 1;
    }
  }

  if (deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return hold_result_from_report(result, current, current_report, config,
                                   FilterStatus::DeadlineFallback, start);
  }

  bool recovery_changed = false;
  std::optional<DistanceGradientReport> last_candidate_report;
  std::array<double, 14> last_evaluated_candidate{};
  for (int recovery = 0;
       recovery < config.nonlinear_recovery_iterations; ++recovery) {
    if (deadline_budget_exhausted(start, config)) {
      result.deadline_missed = true;
      return hold_result_from_report(result, current, current_report, config,
                                     FilterStatus::DeadlineFallback, start);
    }
    last_candidate_report =
        evaluate_distance_gradients(make_kinematics_input(candidate, input));
    last_evaluated_candidate = candidate;
    const DistanceGradientReport& candidate_report = *last_candidate_report;
    std::size_t worst_pair = 0;
    double worst_margin = std::numeric_limits<double>::infinity();
    for (std::size_t pair_index = 0;
         pair_index < kPrimitivePairCount; ++pair_index) {
      if (!proxy_pair_enabled(candidate_report.kinematics, pair_index)) continue;
      const double margin =
          candidate_report.pairs[pair_index].distance.signed_distance -
          pair_required_clearance(candidate_report.kinematics, pair_index,
                                  config);
      if (margin < worst_margin) {
        worst_margin = margin;
        worst_pair = pair_index;
      }
    }
    const double target_margin =
        std::max(config.verification_tolerance_m, 1e-6);
    if (worst_margin >= target_margin &&
        candidate_report.distances.minimum_distance >= target_margin) {
      break;
    }
    const PairDistanceGradient& pair = candidate_report.pairs[worst_pair];
    if (!pair.valid ||
        pair.gradient_squared_norm <= config.gradient_squared_epsilon) {
      break;
    }
    const double required_step =
        (target_margin - worst_margin) / pair.gradient_squared_norm;
    if (!(required_step > 0.0) || !std::isfinite(required_step)) break;
    const std::array<double, 14> previous = candidate;
    for (std::size_t joint = 0; joint < candidate.size(); ++joint) {
      candidate[joint] += required_step * pair.gradient[joint];
    }
    candidate = clamp_to_step(candidate, current, config.max_step_rad, input.command_joint_limits);
    if (maximum_array_difference(candidate, previous) <= 1e-12) break;
    recovery_changed = true;
  }

  if (deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return hold_result_from_report(result, current, current_report, config,
                                   FilterStatus::DeadlineFallback, start);
  }
  SafetyEvaluation output_evaluation;
  if (last_candidate_report && candidate == last_evaluated_candidate) {
    output_evaluation.kinematics = last_candidate_report->kinematics;
    output_evaluation.distances = last_candidate_report->distances;
    output_evaluation.minimum_safety_margin = minimum_safety_margin(
        output_evaluation.kinematics, output_evaluation.distances, config,
        output_evaluation.minimum_pair);
    output_evaluation.safe = output_evaluation.distances.minimum_distance >= 0.0 &&
        output_evaluation.minimum_safety_margin >= -config.verification_tolerance_m;
  } else {
    output_evaluation = evaluate_safety(candidate, input, config);
  }
  KinematicsOutput output_kinematics = output_evaluation.kinematics;
  PrimitiveDistanceReport output_distances = output_evaluation.distances;
  result.q_safe = candidate;
  result.output_min_distance_m = output_distances.minimum_distance;
  result.output_min_pair = output_distances.minimum_pair_index;
  result.output_min_safety_margin_m = output_evaluation.minimum_safety_margin;
  result.nonlinear_verified =
      result.output_min_distance_m >= 0.0 &&
      result.output_min_safety_margin_m >= -config.verification_tolerance_m;

  const bool current_safe = result.input_min_safety_margin_m >=
                            -config.verification_tolerance_m;

  if (result.nonlinear_verified && current_safe &&
      config.path_check_substeps > 0) {
    const double path_margin_target = std::max(
        2.0 * config.verification_tolerance_m, 1e-6);
    const int substeps = std::max(1, config.path_check_substeps);
    double safe_amount = 0.0;
    double unsafe_amount = 1.0;
    bool path_violation = false;
    for (int substep = 1; substep <= substeps; ++substep) {
      if (deadline_budget_exhausted(start, config)) {
        result.deadline_missed = true;
        return hold_result_from_report(result, current, current_report, config,
                                       FilterStatus::DeadlineFallback, start);
      }
      const double amount = static_cast<double>(substep) /
                            static_cast<double>(substeps);
      const auto path_arm = interpolate_arm(current, candidate, amount);
      const SafetyEvaluation path_evaluation =
          evaluate_safety(path_arm, input, config);
      const bool path_strictly_safe =
          path_evaluation.distances.minimum_distance >= path_margin_target &&
          path_evaluation.minimum_safety_margin >= path_margin_target;
      if (!path_strictly_safe) {
        unsafe_amount = amount;
        path_violation = true;
        break;
      }
      safe_amount = amount;
    }

    if (path_violation) {
      const int recovery_iterations = std::max(0, config.path_recovery_iterations);
      for (int iteration = 0; iteration < recovery_iterations; ++iteration) {
        if (deadline_budget_exhausted(start, config)) {
          result.deadline_missed = true;
          return hold_result_from_report(result, current, current_report, config,
                                         FilterStatus::DeadlineFallback, start);
        }
        const double midpoint = 0.5 * (safe_amount + unsafe_amount);
        const auto path_arm = interpolate_arm(current, candidate, midpoint);
        const SafetyEvaluation path_evaluation =
            evaluate_safety(path_arm, input, config);
        const bool path_strictly_safe =
            path_evaluation.distances.minimum_distance >= path_margin_target &&
            path_evaluation.minimum_safety_margin >= path_margin_target;
        if (path_strictly_safe) {
          safe_amount = midpoint;
        } else {
          unsafe_amount = midpoint;
        }
      }
      candidate = interpolate_arm(current, candidate, safe_amount);
      candidate = clamp_to_step(candidate, current, config.max_step_rad, input.command_joint_limits);
      if (deadline_budget_exhausted(start, config)) {
        result.deadline_missed = true;
        return hold_result_from_report(result, current, current_report, config,
                                       FilterStatus::DeadlineFallback, start);
      }
      output_evaluation = evaluate_safety(candidate, input, config);
      output_kinematics = output_evaluation.kinematics;
      output_distances = output_evaluation.distances;
      result.q_safe = candidate;
      result.output_min_distance_m = output_distances.minimum_distance;
      result.output_min_pair = output_distances.minimum_pair_index;
      result.output_min_safety_margin_m = output_evaluation.minimum_safety_margin;
      result.nonlinear_verified = output_evaluation.safe;
      changed = true;
    }
  }

  const bool improved = result.output_min_safety_margin_m >=
                        result.input_min_safety_margin_m +
                            config.improvement_epsilon_m;
  if (result.nonlinear_verified) {
    if (!converged && (changed || recovery_changed)) {
      result.status = FilterStatus::IterationLimitSafe;
    } else {
      result.status = (changed || recovery_changed)
                          ? FilterStatus::Projected
                          : FilterStatus::Normal;
    }
  } else if (!current_safe && improved) {
    result.status = result.input_min_distance_m < 0.0
                        ? FilterStatus::PenetrationRecovery
                        : FilterStatus::AirwallRecovery;
  } else if (config.allow_airwall_recovery && !current_safe) {
    // Provisional candidate only. The outer state-relative swept verifier
    // decides whether this bounded recovery candidate may actually be used.
    result.status = result.input_min_distance_m < 0.0
                        ? FilterStatus::PenetrationRecovery
                        : FilterStatus::AirwallRecovery;
  } else {
    return hold_result(result, current, input, config,
                       has_degenerate_active_pair
                           ? FilterStatus::DegenerateGradientFallback
                           : FilterStatus::DegradedHold,
                       start);
  }
  result.elapsed_us = elapsed_us(start);
  return result;
}

SafetyFilterResult filter_arm_target(const SafetyFilterInput& raw_input,
                                     const SafetyFilterConfig& config) {
  const Clock::time_point start = Clock::now();
  SafetyFilterInput input = raw_input;
  input.q_thumb_urdf = rigid_thumb_pose(input.q_thumb_urdf);
  input.q_thumb_current_urdf = rigid_thumb_pose(
      input.thumb_current_valid ? input.q_thumb_current_urdf
                                : input.q_thumb_urdf);
  std::array<double, 14> current = input.q_current;
  if (!finite_array(current)) current = generated::kNaturalStanding;
  const bool current_arm_valid = measured_arm_in_domain(
      input.q_current, input.current_arm_joint_limits, input.check_current_arm_goal_range);
  if (!current_arm_valid) current = clamp_to_limits(current, input.command_joint_limits);
  const bool current_thumb_valid = input.thumb_pose_valid &&
      valid_rigid_thumb_pose(input.q_thumb_current_urdf, current_thumb_limits(input));
  const bool target_thumb_valid = input.thumb_pose_valid &&
      valid_rigid_thumb_pose(input.q_thumb_urdf, input.thumb_joint_limits_urdf);
  const bool config_valid = std::isfinite(config.deadline_us) &&
      config.deadline_us > 0.0 && std::isfinite(config.deadline_guard_us) &&
      std::isfinite(config.max_step_rad) && config.max_step_rad >= 0.0 &&
      std::isfinite(config.verification_tolerance_m) &&
      config.verification_tolerance_m >= 0.0 &&
      config.projection_iterations >= 0 && valid_command_limits(input.command_joint_limits);
  bool thumb_changes = false;
  for (std::size_t axis : kRigidThumbIndices) {
    const double change = std::abs(input.q_thumb_urdf[axis] -
                                   input.q_thumb_current_urdf[axis]);
    thumb_changes = thumb_changes || change > 1e-12;
  }

  SafetyEvaluation actual_current;
  bool actual_current_evaluated = false;
  auto evaluate_current = [&]() {
    if (!actual_current_evaluated && current_arm_valid &&
        current_thumb_valid && config_valid) {
      SafetyFilterInput current_context = input;
      current_context.q_thumb_urdf = input.q_thumb_current_urdf;
      actual_current = evaluate_safety(current, current_context, config);
      actual_current.safe = actual_current.safe &&
                            actual_current.minimum_safety_margin >= 0.0;
      actual_current_evaluated = true;
    }
  };
  auto hold_current = [&](SafetyFilterResult result, FilterStatus status) {
    evaluate_current();
    result.q_safe = clamp_to_limits(current, input.command_joint_limits);
    result.status = status;
    result.thumb_target_accepted = false;
    result.nonlinear_verified = actual_current_evaluated && actual_current.safe;
    if (actual_current_evaluated) {
      result.input_min_distance_m = actual_current.distances.minimum_distance;
      result.output_min_distance_m = actual_current.distances.minimum_distance;
      result.input_min_safety_margin_m = actual_current.minimum_safety_margin;
      result.output_min_safety_margin_m = actual_current.minimum_safety_margin;
      result.input_min_pair = actual_current.minimum_pair;
      result.output_min_pair = actual_current.minimum_pair;
    }
    result.elapsed_us = elapsed_us(start);
    return result;
  };

  if (!current_arm_valid || !current_thumb_valid || !target_thumb_valid ||
      !finite_array(input.q_nominal) || !config_valid) {
    SafetyFilterResult invalid;
    diagnose_invalid_input(invalid, input, config, InvalidInputSite::Filter);
    return hold_current(invalid, FilterStatus::InvalidInputFallback);
  }
  evaluate_current();
  const bool recovering = !actual_current.safe && config.allow_airwall_recovery;
  if (!actual_current.safe && !recovering) {
    return hold_current({}, FilterStatus::DegradedHold);
  }
  for (std::size_t j = 0; j < current.size(); ++j) {
    if (std::max(input.command_joint_limits[j][0], current[j] - config.max_step_rad) >
        std::min(input.command_joint_limits[j][1], current[j] + config.max_step_rad)) {
      return hold_current({}, FilterStatus::DegradedHold);
    }
  }
  if (deadline_budget_exhausted(start, config)) {
    SafetyFilterResult result;
    result.deadline_missed = true;
    return hold_current(result, FilterStatus::DeadlineFallback);
  }

  SafetyFilterConfig fixed_config = config;
  if (thumb_changes) {
    fixed_config.path_check_substeps = 0;
    const double motion_guard =
        std::max(4.0 * config.verification_tolerance_m, 4e-5);
    fixed_config.extra_clearance_m += motion_guard;
    fixed_config.shoulder_torso_clearance_m += motion_guard;
  }
  SafetyFilterResult result =
      filter_arm_target_fixed_thumb(input, fixed_config, start);
  if (result.deadline_missed || deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return hold_current(result, FilterStatus::DeadlineFallback);
  }
  if (recovering) {
    // The legacy endpoint solver can recover only part of a violated margin
    // inside a small command box. Verify that path against the measured initial
    // barriers, never require an already-unsafe start to be safe before trying.
    auto candidate = input;
    candidate.q_nominal = result.q_safe;
    SafetyFilterResult verified;
    auto deadline_hold = [&]() {
      verified.deadline_missed = true;
      return hold_current(verified, FilterStatus::DeadlineFallback);
    };
    for (int repair = 0; repair <= std::max(0, config.path_recovery_iterations); ++repair) {
      if (deadline_budget_exhausted(start, config)) return deadline_hold();
      auto recovery_config = config;
      recovery_config.deadline_us = std::max(1.0, config.deadline_us - elapsed_us(start));
      verified = verify_arm_target(candidate, recovery_config);
      verified.active_pair_count = result.active_pair_count;
      verified.projection_iterations = result.projection_iterations + repair;
      verified.elapsed_us = elapsed_us(start);
      if (verified.deadline_missed || deadline_budget_exhausted(start, config)) return deadline_hold();
      if (verified.nonlinear_verified && verified.thumb_target_accepted) {
        return verified;
      }
      if (repair == config.path_recovery_iterations ||
          verified.path_violation_pair >= kPrimitivePairCount ||
          verified.path_violation_fraction <= 0.0) break;
      const double t = verified.path_violation_fraction;
      auto context = input;
      for (auto axis : kRigidThumbIndices) {
        context.q_thumb_urdf[axis] = input.q_thumb_current_urdf[axis] +
            t * (input.q_thumb_urdf[axis] - input.q_thumb_current_urdf[axis]);
      }
      if (deadline_budget_exhausted(start, config)) return deadline_hold();
      const auto report = evaluate_distance_gradients(make_kinematics_input(
          interpolate_arm(current, candidate.q_nominal, t), context));
      if (deadline_budget_exhausted(start, config)) return deadline_hold();
      const auto& pair = report.pairs[verified.path_violation_pair];
      if (!report.valid || !pair.valid || pair.gradient_squared_norm <= config.gradient_squared_epsilon) break;
      ArmGradient path_gradient{};
      for (std::size_t j = 0; j < current.size(); ++j) path_gradient[j] = t * pair.gradient[j];
      const double previous_t = verified.path_violation_previous_fraction;
      if (previous_t > 0.0) {
        auto previous_context = input;
        for (auto axis : kRigidThumbIndices) {
          previous_context.q_thumb_urdf[axis] = input.q_thumb_current_urdf[axis] +
              previous_t * (input.q_thumb_urdf[axis] - input.q_thumb_current_urdf[axis]);
        }
        if (deadline_budget_exhausted(start, config)) return deadline_hold();
        const auto previous_report = evaluate_distance_gradients(make_kinematics_input(
            interpolate_arm(current, candidate.q_nominal, previous_t), previous_context));
        if (deadline_budget_exhausted(start, config)) return deadline_hold();
        const auto& previous_pair = previous_report.pairs[verified.path_violation_pair];
        if (!previous_report.valid || !previous_pair.valid) break;
        // Differentiate d(t)-d(t_previous), not just d(t). Otherwise a later
        // monotonicity violation would be under-corrected by its sample index.
        for (std::size_t j = 0; j < current.size(); ++j) {
          path_gradient[j] -= previous_t * previous_pair.gradient[j];
        }
      }
      double denominator = 0.0;
      for (double value : path_gradient) denominator += value * value;
      if (denominator <= config.gradient_squared_epsilon) break;
      const auto before = candidate.q_nominal;
      const double guard = std::max(1e-8, config.verification_tolerance_m);
      const double scale = (verified.path_violation_deficit_m + guard) / denominator;
      for (std::size_t j = 0; j < current.size(); ++j) {
        candidate.q_nominal[j] += scale * path_gradient[j];
      }
      candidate.q_nominal = clamp_to_step(candidate.q_nominal, current,
          config.max_step_rad, input.command_joint_limits);
      if (maximum_array_difference(before, candidate.q_nominal) <= 1e-12) break;
    }
    if (verified.deadline_missed || deadline_budget_exhausted(start, config)) return deadline_hold();
    return hold_current(verified, verified.status);
  }
  if (!thumb_changes) {
    const bool inner_hold =
        result.status == FilterStatus::InvalidInputFallback ||
        result.status == FilterStatus::DegenerateGradientFallback ||
        result.status == FilterStatus::DegradedHold;
    if (!result.nonlinear_verified || result.output_min_safety_margin_m < 0.0 ||
        result.output_min_distance_m < 0.0 || inner_hold) {
      const bool recovery = result.status == FilterStatus::AirwallRecovery ||
                            result.status == FilterStatus::PenetrationRecovery;
      return hold_current(result, recovery ? FilterStatus::DegradedHold
                                          : result.status);
    }
    result.thumb_target_accepted = result.nonlinear_verified &&
        result.status != FilterStatus::InvalidInputFallback;
    result.elapsed_us = elapsed_us(start);
    return result;
  }
  if (!result.nonlinear_verified) {
    return hold_current(result, FilterStatus::ThumbMotionRejected);
  }

  const double left_thumb_travel =
      std::abs(input.q_thumb_urdf[0] - input.q_thumb_current_urdf[0]) +
      std::abs(input.q_thumb_urdf[1] - input.q_thumb_current_urdf[1]);
  const double right_thumb_travel =
      std::abs(input.q_thumb_urdf[3] - input.q_thumb_current_urdf[3]) +
      std::abs(input.q_thumb_urdf[4] - input.q_thumb_current_urdf[4]);
  const int angle_substeps = static_cast<int>(
      std::ceil(std::max(left_thumb_travel, right_thumb_travel) / 0.02));
  const int substeps = std::max({64, config.path_check_substeps, angle_substeps});
  const double path_guard =
      std::max(2.0 * config.verification_tolerance_m, 2e-5);
  for (int substep = 1; substep <= substeps; ++substep) {
    if (deadline_budget_exhausted(start, config)) {
      result.deadline_missed = true;
      return hold_current(result, FilterStatus::DeadlineFallback);
    }
    const double amount = static_cast<double>(substep) / substeps;
    SafetyFilterInput path_context = input;
    for (std::size_t axis : kRigidThumbIndices) {
      path_context.q_thumb_urdf[axis] = input.q_thumb_current_urdf[axis] +
          amount * (input.q_thumb_urdf[axis] - input.q_thumb_current_urdf[axis]);
    }
    const auto path_arm = interpolate_arm(current, result.q_safe, amount);
    if (config.use_path_separation_culling && substep < substeps) {
      if (!path_sample_is_safe(path_arm, path_context, config, path_guard)) {
        return hold_current(result, FilterStatus::ThumbMotionRejected);
      }
      continue;
    }
    const SafetyEvaluation path_evaluation =
        evaluate_safety(path_arm, path_context, config);
    if (!path_evaluation.safe ||
        path_evaluation.minimum_safety_margin < path_guard ||
        path_evaluation.distances.minimum_distance < path_guard) {
      return hold_current(result, FilterStatus::ThumbMotionRejected);
    }
    if (substep == substeps) {
      result.output_min_distance_m = path_evaluation.distances.minimum_distance;
      result.output_min_safety_margin_m = path_evaluation.minimum_safety_margin;
      result.output_min_pair = path_evaluation.minimum_pair;
    }
  }
  if (deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return hold_current(result, FilterStatus::DeadlineFallback);
  }
  result.thumb_target_accepted = true;
  result.input_min_distance_m = actual_current.distances.minimum_distance;
  result.input_min_safety_margin_m = actual_current.minimum_safety_margin;
  result.input_min_pair = actual_current.minimum_pair;
  result.elapsed_us = elapsed_us(start);
  return result;
}

static SafetyFilterResult verify_target_impl(const SafetyFilterInput& raw_input,
                                             const SafetyFilterConfig& config,
                                             bool enforce_command_box) {
  const auto start = Clock::now();
  SafetyFilterInput input = raw_input;
  input.q_thumb_urdf = rigid_thumb_pose(input.q_thumb_urdf);
  input.q_thumb_current_urdf = rigid_thumb_pose(
      input.thumb_current_valid ? input.q_thumb_current_urdf : input.q_thumb_urdf);
  SafetyFilterResult result;
  result.q_safe = clamp_to_limits(finite_array(input.q_current) ?
      input.q_current : generated::kNaturalStanding, input.command_joint_limits);
  auto finish = [&](FilterStatus status) {
    result.status = status;
    result.elapsed_us = elapsed_us(start);
    return result;
  };
  if (!measured_arm_in_domain(input.q_current, input.current_arm_joint_limits, input.check_current_arm_goal_range) ||
      !valid_command_limits(input.command_joint_limits) ||
      !finite_array(input.q_nominal) ||
      (enforce_command_box && clamp_to_limits(input.q_nominal, input.command_joint_limits) != input.q_nominal) ||
      !input.thumb_pose_valid ||
      !valid_rigid_thumb_pose(input.q_thumb_current_urdf, current_thumb_limits(input)) ||
      !valid_rigid_thumb_pose(input.q_thumb_urdf, input.thumb_joint_limits_urdf) ||
      !std::isfinite(config.deadline_us) || config.deadline_us <= 0.0 ||
      !std::isfinite(config.deadline_guard_us) || config.deadline_guard_us < 0.0 ||
      !std::isfinite(config.extra_clearance_m) || !std::isfinite(config.shoulder_torso_clearance_m)) {
    diagnose_invalid_input(result, input, config, InvalidInputSite::Verify);
    return finish(FilterStatus::InvalidInputFallback);
  }
  if (deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return finish(FilterStatus::DeadlineFallback);
  }
  auto current_context = input;
  current_context.q_thumb_urdf = input.q_thumb_current_urdf;
  const auto current = evaluate_safety(input.q_current, current_context, config);
  const auto valid_evaluation = [](const SafetyEvaluation& evaluation) {
    if (!evaluation.kinematics.valid || !std::isfinite(evaluation.minimum_safety_margin)) return false;
    for (std::size_t pair_index = 0; pair_index < kPrimitivePairCount; ++pair_index) {
      if (!proxy_pair_enabled(evaluation.kinematics, pair_index)) continue;
      const auto& distance = evaluation.distances.distances[pair_index];
      if (!distance.valid || !std::isfinite(distance.signed_distance)) return false;
    }
    return true;
  };
  const auto strictly_safe = [&](const SafetyEvaluation& evaluation) {
    return valid_evaluation(evaluation) && evaluation.minimum_safety_margin >= 0.0 &&
      evaluation.distances.minimum_distance >= 0.0;
  };
  result.input_min_distance_m = current.distances.minimum_distance;
  result.input_min_safety_margin_m = current.minimum_safety_margin;
  result.input_min_pair = current.minimum_pair;
  const bool recovering = !strictly_safe(current);
  if (!valid_evaluation(current) || (recovering && !config.allow_airwall_recovery)) {
    return finish(FilterStatus::DegradedHold);
  }
  const bool stationary = input.q_current == input.q_nominal &&
                          input.q_thumb_current_urdf == input.q_thumb_urdf;
  const auto target = stationary ? current : evaluate_safety(input.q_nominal, input, config);
  result.output_min_distance_m = target.distances.minimum_distance;
  result.output_min_safety_margin_m = target.minimum_safety_margin;
  result.output_min_pair = target.minimum_pair;
  if (!valid_evaluation(target) || (!recovering && !strictly_safe(target))) {
    return finish(FilterStatus::ThumbMotionRejected);
  }
  if (recovering) {
    // A violated barrier is not a reason to reject every withdrawal. Until
    // each affected pair has left its barrier, its distance may not decrease.
    // Pairs initially safe keep their full clearance; an initially penetrating
    // pair cannot penetrate further. Partial recovery and tangential motion are
    // permitted, so a rate-limited command need not clear the wall in one tick.
    std::array<double, kPrimitivePairCount> previous_margin{};
    for (std::size_t p = 0; p < kPrimitivePairCount; ++p) {
      if (!proxy_pair_enabled(current.kinematics, p)) continue;
      previous_margin[p] = current.distances.distances[p].signed_distance -
          pair_required_clearance(current.kinematics, p, config);
    }
    if (!stationary) {
      const double left = std::abs(input.q_thumb_urdf[0] - input.q_thumb_current_urdf[0]) +
                          std::abs(input.q_thumb_urdf[1] - input.q_thumb_current_urdf[1]);
      const double right = std::abs(input.q_thumb_urdf[3] - input.q_thumb_current_urdf[3]) +
                           std::abs(input.q_thumb_urdf[4] - input.q_thumb_current_urdf[4]);
      const int steps = std::max({64, config.path_check_substeps,
          static_cast<int>(std::ceil(std::max(left, right) / 0.02))});
      for (int step = 1; step <= steps; ++step) {
        if (deadline_budget_exhausted(start, config)) {
          result.deadline_missed = true;
          return finish(FilterStatus::DeadlineFallback);
        }
        const double t = static_cast<double>(step) / steps;
        auto context = input;
        for (auto axis : kRigidThumbIndices) {
          context.q_thumb_urdf[axis] = input.q_thumb_current_urdf[axis] +
              t * (input.q_thumb_urdf[axis] - input.q_thumb_current_urdf[axis]);
        }
        const auto kinematics = step == steps ? target.kinematics : forward_kinematics(
            make_kinematics_input(interpolate_arm(input.q_current, input.q_nominal, t), context));
        if (!kinematics.valid) return finish(FilterStatus::ThumbMotionRejected);
        for (std::size_t p = 0; p < kPrimitivePairCount; ++p) {
          if (!proxy_pair_enabled(kinematics, p)) continue;
          const double floor = std::min(0.0, previous_margin[p]);
          const auto pair = kPrimitivePairs[p];
          const auto& first = kinematics.primitives[static_cast<std::size_t>(pair.first)];
          const auto& second = kinematics.primitives[static_cast<std::size_t>(pair.second)];
          const double required = pair_required_clearance(kinematics, p, config);
          // Keep the existing fast separating-axis proof for pairs already
          // outside their barrier; only recovering pairs need actual distance
          // values at every sample to enforce monotonic withdrawal.
          if (step < steps && floor == 0.0 && config.use_path_separation_culling &&
              first.kind != PrimitiveKind::Sphere && second.kind != PrimitiveKind::Sphere &&
              ((first.kind == PrimitiveKind::Box && box_axis_proves_clearance(first, second, first, required)) ||
               (second.kind == PrimitiveKind::Box && box_axis_proves_clearance(first, second, second, required)))) {
            continue;
          }
          const auto distance = step == steps ? target.distances.distances[p] : distance_between(first, second);
          if (!distance.valid || !std::isfinite(distance.signed_distance)) {
            return finish(FilterStatus::ThumbMotionRejected);
          }
          const double margin = distance.signed_distance - required;
          if (margin < floor - (floor < 0.0 ? 1e-10 : 0.0)) {
            result.path_violation_pair = p;
            result.path_violation_fraction = t;
            result.path_violation_previous_fraction = floor < 0.0 ?
                static_cast<double>(step - 1) / steps : 0.0;
            result.path_violation_deficit_m = floor - margin;
            return finish(FilterStatus::ThumbMotionRejected);
          }
          previous_margin[p] = margin;
        }
      }
    }
    if (deadline_budget_exhausted(start, config)) {
      result.deadline_missed = true;
      return finish(FilterStatus::DeadlineFallback);
    }
    result.q_safe = input.q_nominal;
    result.nonlinear_verified = true;
    result.thumb_target_accepted = true;
    result.recovery_verified = true;
    return finish(current.distances.minimum_distance < 0.0 ?
        FilterStatus::PenetrationRecovery : FilterStatus::AirwallRecovery);
  }
  if (!stationary) {
    const double left = std::abs(input.q_thumb_urdf[0] - input.q_thumb_current_urdf[0]) +
                        std::abs(input.q_thumb_urdf[1] - input.q_thumb_current_urdf[1]);
    const double right = std::abs(input.q_thumb_urdf[3] - input.q_thumb_current_urdf[3]) +
                         std::abs(input.q_thumb_urdf[4] - input.q_thumb_current_urdf[4]);
    const int steps = std::max({64, config.path_check_substeps,
        static_cast<int>(std::ceil(std::max(left, right) / 0.02))});
    for (int step = 1; step < steps; ++step) {
      if (deadline_budget_exhausted(start, config)) {
        result.deadline_missed = true;
        return finish(FilterStatus::DeadlineFallback);
      }
      const double t = static_cast<double>(step) / steps;
      auto context = input;
      for (auto axis : kRigidThumbIndices) {
        context.q_thumb_urdf[axis] = input.q_thumb_current_urdf[axis] +
            t * (input.q_thumb_urdf[axis] - input.q_thumb_current_urdf[axis]);
      }
      // Strict nonnegative raw separation and configured airwall at every
      // sample. No negative numerical tolerance is used for this verifier.
      const auto arm = interpolate_arm(input.q_current, input.q_nominal, t);
      const bool safe = config.use_path_separation_culling ?
          path_sample_is_safe(arm, context, config, 0.0) :
          strictly_safe(evaluate_safety(arm, context, config));
      if (!safe) {
        return finish(FilterStatus::ThumbMotionRejected);
      }
    }
  }
  if (deadline_budget_exhausted(start, config)) {
    result.deadline_missed = true;
    return finish(FilterStatus::DeadlineFallback);
  }
  result.q_safe = input.q_nominal;
  result.nonlinear_verified = true;
  result.thumb_target_accepted = true;
  return finish(FilterStatus::Normal);
}

SafetyFilterResult verify_arm_target(const SafetyFilterInput& input,
                                     const SafetyFilterConfig& config) {
  return verify_target_impl(input, config, true);
}

SafetyFilterResult verify_thumb_target_with_arm_held(const SafetyFilterInput& input,
                                                     const SafetyFilterConfig& config) {
  auto held = input;
  held.q_nominal = input.q_current;
  return verify_target_impl(held, config, false);
}

const char* filter_status_name(FilterStatus status) {
  switch (status) {
    case FilterStatus::Normal:
      return "NORMAL";
    case FilterStatus::Projected:
      return "PROJECTED";
    case FilterStatus::IterationLimitSafe:
      return "ITERATION_LIMIT_SAFE";
    case FilterStatus::AirwallRecovery:
      return "AIRWALL_RECOVERY";
    case FilterStatus::PenetrationRecovery:
      return "PENETRATION_RECOVERY";
    case FilterStatus::DeadlineFallback:
      return "DEADLINE_FALLBACK";
    case FilterStatus::InvalidInputFallback:
      return "INVALID_INPUT_FALLBACK";
    case FilterStatus::DegenerateGradientFallback:
      return "DEGENERATE_GRADIENT_FALLBACK";
    case FilterStatus::DegradedHold:
      return "DEGRADED_HOLD";
    case FilterStatus::ThumbMotionRejected:
      return "THUMB_MOTION_REJECTED";
  }
  return "UNKNOWN";
}

}  // namespace x2::cbf
