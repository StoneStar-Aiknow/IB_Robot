// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0

#include "aimdk_dataplane/projection.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace aimdk_dataplane
{

const std::vector<std::string> kVendorArmOrder = {
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
};
const std::vector<std::string> kVendorHeadOrder = {"head_yaw_joint", "head_pitch_joint"};
const std::vector<std::string> kDexterousJointOrder = {
  "thumb_roll", "thumb_abad", "thumb_mcp", "index_abad", "index_pip",
  "middle_pip", "ring_abad", "ring_pip", "pinky_abad", "pinky_pip",
};
const char kUpperBodyHandsAction[] = "UPPERBODY_REMOTE_SPLIT";

namespace
{

std::string number(double value)
{
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%g", value);
  return buffer;
}

void finite(const std::vector<double> & values, const std::string & label)
{
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw CommandRejected("NON_FINITE_COMMAND", label + ": " + number(value));
    }
  }
}

std::string joined(const std::vector<std::string> & names)
{
  std::string out = "[";
  for (size_t index = 0; index < names.size(); ++index) {
    out += (index ? ", '" : "'") + names[index] + "'";
  }
  return out + "]";
}

// Values of ``channel_joints`` in ``vendor_order``; throws ``<prefix>_LENGTH`` / ``<prefix>_JOINTS``.
std::vector<double> reorder(
  const std::vector<double> & values, const std::vector<std::string> & channel_joints,
  const std::vector<std::string> & vendor_order, const std::string & prefix)
{
  if (values.size() != channel_joints.size()) {
    throw CommandRejected(
      prefix + "_LENGTH",
      "expected " + std::to_string(channel_joints.size()) + " values, got " + std::to_string(values.size()));
  }
  std::unordered_map<std::string, double> by_name;
  for (size_t index = 0; index < values.size(); ++index) {
    by_name[channel_joints[index]] = values[index];
  }
  std::vector<std::string> missing;
  std::vector<double> out;
  for (const auto & name : vendor_order) {
    const auto found = by_name.find(name);
    if (found == by_name.end()) {
      missing.push_back(name);
    } else {
      out.push_back(found->second);
    }
  }
  if (!missing.empty()) {
    throw CommandRejected(prefix + "_JOINTS", "missing " + joined(missing));
  }
  return out;
}

bool ends_with(const std::string & text, const std::string & suffix)
{
  return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

std::vector<double> arm_payload(const std::vector<double> & values, const std::vector<std::string> & channel_joints)
{
  finite(values, "arm command");
  return reorder(values, channel_joints, kVendorArmOrder, "ARM_COMMAND");
}

std::vector<double> head_payload(
  const std::vector<double> & values, const std::vector<std::string> & channel_joints,
  const std::map<std::string, double> & limits)
{
  finite(values, "head command");
  auto payload = reorder(values, channel_joints, kVendorHeadOrder, "HEAD_COMMAND");
  for (size_t index = 0; index < payload.size(); ++index) {
    const auto limit = limits.find(kVendorHeadOrder[index]);
    if (limit != limits.end() && std::abs(payload[index]) > std::abs(limit->second)) {
      char detail[160];
      std::snprintf(
        detail, sizeof(detail), "%s target %.3f outside the platform's +/-%.3f travel",
        kVendorHeadOrder[index].c_str(), payload[index], std::abs(limit->second));
      throw CommandRejected("HEAD_COMMAND_RANGE", detail);
    }
  }
  return payload;
}

namespace
{

std::vector<double> claw_payload(const std::vector<double> & values, double command_min, double command_max)
{
  finite(values, "gripper command");
  if (values.size() != 2) {
    throw CommandRejected(
      "GRIPPER_COMMAND_LENGTH", "expected 2 values [left, right], got " + std::to_string(values.size()));
  }
  for (size_t index = 0; index < values.size(); ++index) {
    if (!(command_min <= values[index] && values[index] <= command_max)) {
      throw CommandRejected(
        "GRIPPER_COMMAND_RANGE", std::string(index == 0 ? "left" : "right") + "=" + number(values[index]) +
        " outside [" + number(command_min) + ", " + number(command_max) + "]");
    }
  }
  return values;
}

std::vector<double> dexterous_joint_payload(const std::vector<double> & values)
{
  finite(values, "hand joint command");
  const size_t expected = 2 * kDexterousJointOrder.size();
  if (values.size() != expected) {
    throw CommandRejected(
      "HAND_COMMAND_LENGTH",
      "expected " + std::to_string(expected) + " values, got " + std::to_string(values.size()));
  }
  return values;
}

std::vector<double> gesture_payload(const std::vector<double> & values, const std::optional<std::vector<int>> & ids)
{
  finite(values, "hand gesture command");
  if (values.size() != 4) {
    throw CommandRejected(
      "HAND_GESTURE_LENGTH",
      "expected 4 values [left_gesture, left_open, right_gesture, right_open], got " + std::to_string(values.size()));
  }
  for (size_t index : {0u, 2u}) {
    const double gesture = values[index];
    if (gesture != std::trunc(gesture)) {
      throw CommandRejected("HAND_GESTURE_ID", "gesture id must be an integer, got " + number(gesture));
    }
    if (ids && std::find(ids->begin(), ids->end(), static_cast<int>(gesture)) == ids->end()) {
      throw CommandRejected("HAND_GESTURE_ID", "unknown gesture id " + std::to_string(static_cast<int>(gesture)));
    }
  }
  for (size_t index : {1u, 3u}) {
    if (!(0.0 <= values[index] && values[index] <= 1.0)) {
      throw CommandRejected("HAND_GESTURE_RANGE", "openness " + number(values[index]) + " outside [0, 1]");
    }
  }
  return values;
}

}  // namespace

std::pair<int, std::vector<double>> hand_payload(
  const std::vector<double> & values, const std::string & hand_type, const HandConfig & config)
{
  if (hand_type == "claw") {
    return {kHandSubModeClaw, claw_payload(values, config.command_min, config.command_max)};
  }
  if (hand_type == "dexterous_joint") {
    return {kHandSubModeDexterousJoint, dexterous_joint_payload(values)};
  }
  if (hand_type == "dexterous_gesture") {
    return {kHandSubModeDexterousGesture, gesture_payload(values, config.gesture_ids)};
  }
  throw std::invalid_argument("unsupported hand type '" + hand_type + "'");
}

std::optional<std::vector<double>> dexterous_hold_payload(
  const std::array<bool, 2> & installed, const std::array<std::map<std::string, double>, 2> & measured)
{
  if (!installed[0] && !installed[1]) {
    return std::nullopt;
  }
  std::vector<double> payload;
  for (size_t side = 0; side < 2; ++side) {
    if (!installed[side]) {
      payload.insert(payload.end(), kDexterousJointOrder.size(), 0.0);
      continue;
    }
    std::map<std::string, double> by_joint;
    for (const auto & [name, value] : measured[side]) {
      std::string lower = name;
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {return std::tolower(c);});
      by_joint[lower] = value;
    }
    for (const auto & joint : kDexterousJointOrder) {
      std::vector<double> match;
      for (const auto & [name, value] : by_joint) {
        if (ends_with(name, joint + "_joint")) {
          match.push_back(value);
        }
      }
      if (match.size() != 1) {
        return std::nullopt;
      }
      payload.push_back(match.front());
    }
  }
  return payload;
}

LocomotionLimits LocomotionLimits::with_platform_envelope(
  const std::optional<std::pair<double, double>> & envelope) const
{
  // All-zero bounds mean "no envelope reported", not "this robot may not move".
  if (!envelope || envelope->first <= 0.0 || envelope->second <= 0.0) {
    return *this;
  }
  LocomotionLimits out = *this;
  out.max_linear_mps = std::min(max_linear_mps, envelope->first);
  out.max_angular_radps = std::min(max_angular_radps, envelope->second);
  return out;
}

std::array<double, 3> twist_to_locomotion(
  double linear_x, double linear_y, double angular_z, const LocomotionLimits & limits)
{
  finite({linear_x, linear_y, angular_z}, "twist");
  const std::array<std::tuple<double, double, double, const char *>, 3> components = {{
    {linear_x, limits.min_linear_mps, limits.max_linear_mps, "linear.x"},
    {linear_y, limits.min_linear_mps, limits.max_linear_mps, "linear.y"},
    {angular_z, limits.min_angular_radps, limits.max_angular_radps, "angular.z"},
  }};
  std::array<double, 3> out{};
  for (size_t index = 0; index < components.size(); ++index) {
    const auto & [value, minimum, maximum, label] = components[index];
    const double magnitude = std::abs(value);
    if (magnitude < limits.deadband) {
      out[index] = 0.0;
      continue;
    }
    if (magnitude > maximum) {
      throw CommandRejected(
        "VELOCITY_ABOVE_MAX", std::string(label) + "=" + number(value) + " exceeds " + number(maximum));
    }
    if (magnitude < minimum) {
      throw CommandRejected(
        "VELOCITY_BELOW_MIN", std::string(label) + "=" + number(value) + " is below the platform minimum " +
        number(minimum) + " and would not move the robot");
    }
    out[index] = value;
  }
  return out;
}

std::optional<std::string> hand_type_family(int value)
{
  switch (value) {
    case 0x01:  // NIMBLE
    case 0x03:  // LEISAI_NIMBLE
    case 0x04:  // LITE_S
      return std::string("dexterous");
    case 0x02:  // CLAW
      return std::string("claw");
    default:
      return std::nullopt;
  }
}

Aggregate aggregate_joint_state(const std::vector<const JointEntries *> & groups, const std::vector<std::string> & order)
{
  std::unordered_map<std::string, std::tuple<double, double, double>> merged;
  for (const auto * entries : groups) {
    for (const auto & [name, position, velocity, effort] : *entries) {
      merged[name] = {position, velocity, effort};
    }
  }
  Aggregate out;
  for (const auto & name : order) {
    const auto found = merged.find(name);
    if (found == merged.end()) {
      continue;
    }
    out.names.push_back(name);
    out.position.push_back(std::get<0>(found->second));
    out.velocity.push_back(std::get<1>(found->second));
    out.effort.push_back(std::get<2>(found->second));
  }
  return out;
}

}  // namespace aimdk_dataplane
