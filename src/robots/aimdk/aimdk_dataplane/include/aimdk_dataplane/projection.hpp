// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The command and state projection rules of aimdk_robot/projection.py that the
// data plane applies at the platform's rate. Each function mirrors its Python
// namesake rule for rule, rejection reasons included: those reasons appear
// verbatim in RuntimeStatus, so an operator reads the same answer whichever
// side refused. No ROS and no vendor types here.

#ifndef AIMDK_DATAPLANE__PROJECTION_HPP_
#define AIMDK_DATAPLANE__PROJECTION_HPP_

#include <array>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace aimdk_dataplane
{

// UpperBodyCommandArray.arm_pos: left 7 then right 7.
extern const std::vector<std::string> kVendorArmOrder;
// UpperBodyCommandArray.head_pos: [yaw, pitch].
extern const std::vector<std::string> kVendorHeadOrder;
// Dexterous hand joints per hand, in the SDK example's documented order.
extern const std::vector<std::string> kDexterousJointOrder;

constexpr int kHandSubModeClaw = 1;
constexpr int kHandSubModeDexterousJoint = 2;
constexpr int kHandSubModeDexterousGesture = 3;

// The MC action whose upper-body commands carry the hands.
extern const char kUpperBodyHandsAction[];

// A well-formed command the runtime must refuse; ``reason`` is machine-readable.
class CommandRejected : public std::runtime_error
{
public:
  CommandRejected(std::string reason, const std::string & detail = "")
  : std::runtime_error(detail.empty() ? reason : reason + ": " + detail), reason(std::move(reason)) {}

  std::string reason;
};

std::vector<double> arm_payload(const std::vector<double> & values, const std::vector<std::string> & channel_joints);

std::vector<double> head_payload(
  const std::vector<double> & values, const std::vector<std::string> & channel_joints,
  const std::map<std::string, double> & limits);

struct HandConfig
{
  double command_min = 0.0;
  double command_max = 1.0;
  std::optional<std::vector<int>> gesture_ids;
};

// ``(hand_sub_mode, hand_pos)`` for a channel's hand type: claw, dexterous_joint or dexterous_gesture.
std::pair<int, std::vector<double>> hand_payload(
  const std::vector<double> & values, const std::string & hand_type, const HandConfig & config);

// Sub-mode 2 hand_pos holding the dexterous hands at ``measured`` (left, right),
// or nullopt when an installed hand is not fully measured or none is installed.
std::optional<std::vector<double>> dexterous_hold_payload(
  const std::array<bool, 2> & installed, const std::array<std::map<std::string, double>, 2> & measured);

struct LocomotionLimits
{
  double deadband = 0.005;
  double min_linear_mps = 0.2;
  double min_angular_radps = 0.1;
  double max_linear_mps = 1.0;
  double max_angular_radps = 1.0;

  // The maxima tightened to the envelope the platform reports, when it reports one.
  LocomotionLimits with_platform_envelope(const std::optional<std::pair<double, double>> & envelope) const;
};

std::array<double, 3> twist_to_locomotion(
  double linear_x, double linear_y, double angular_z, const LocomotionLimits & limits);

// "dexterous" or "claw" for a vendor HandType value, nullopt for NONE, ERROR or unknown.
std::optional<std::string> hand_type_family(int value);

// One joint group's latest sample: (name, position, velocity, effort) per joint.
using JointEntries = std::vector<std::tuple<std::string, double, double, double>>;

struct Aggregate
{
  std::vector<std::string> names;
  std::vector<double> position;
  std::vector<double> velocity;
  std::vector<double> effort;
};

// Merge groups into ``order``; joints the groups do not carry are omitted, not zero-filled.
Aggregate aggregate_joint_state(const std::vector<const JointEntries *> & groups, const std::vector<std::string> & order);

}  // namespace aimdk_dataplane

#endif  // AIMDK_DATAPLANE__PROJECTION_HPP_
