// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// What the data plane knows and decides, without ROS: the platform's latest
// state per joint group, the commands the public channels set, and the fixed-rate
// command each tick sends. Times are seconds on one monotonic clock supplied by
// the caller, so every rule here is testable without a robot or a DDS participant.
//
// The rules are aimdk_robot's runtime_node rules, moved here from Python where
// converting every ~500 Hz vendor sample under one GIL held the X2's 310P
// robot-side host to /joint_states at 0.5 Hz and delayed commands by 100-200 ms.
// The runtime keeps the control plane (modes, arbitration, stop, services) and
// tells this core what is admitted through Control.

#ifndef AIMDK_DATAPLANE__CORE_HPP_
#define AIMDK_DATAPLANE__CORE_HPP_

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "aimdk_dataplane/projection.hpp"

namespace aimdk_dataplane
{

struct Channel
{
  std::string name;
  std::string topic;
  bool twist = false;
  std::vector<std::string> joints;
  std::set<std::string> modes;
  // claw, dexterous_joint or dexterous_gesture for a hand channel, empty otherwise.
  std::string hand_type;
};

struct Endpoint
{
  std::string topic;
  std::string reliability = "best_effort";
  std::string durability = "volatile";
  int depth = 10;
};

struct View
{
  std::string topic;
  std::vector<std::string> order;
};

// Everything static, resolved by aimdk_robot.dataplane from the profile; see plan_from_json.
struct Plan
{
  double joint_state_rate_hz = 50.0;
  double command_rate_hz = 50.0;
  double status_rate_hz = 20.0;
  double feedback_window_s = 0.5;
  double stream_staleness_s = 0.5;
  double cmd_vel_staleness_s = 0.5;
  double control_timeout_s = 0.5;
  std::string source;
  std::vector<std::pair<std::string, Endpoint>> joint_state;  // (group, vendor endpoint)
  std::optional<Endpoint> hand_state;
  std::optional<Endpoint> arm_command;
  Endpoint upper_body_command;
  Endpoint locomotion_command;
  std::vector<View> views;  // /joint_states first
  std::optional<Endpoint> touch;
  double touch_rate_hz = 50.0;
  std::optional<View> arm_command_state;
  double arm_command_state_rate_hz = 50.0;
  std::vector<Channel> channels;
  std::map<std::string, double> head_limits;
  HandConfig hand;
  int default_hand_sub_mode = kHandSubModeClaw;
  LocomotionLimits locomotion;
  std::map<std::string, std::string> mc_actions;
  bool navigation_gate = false;
  std::string control_topic;
  std::string status_topic;
  std::string named_arm_topic;
};

Plan plan_from_json(const std::string & text);

// What the runtime's control plane admits, as it last said. Counters move to ask
// for a one-off effect: ``clear`` drops every cached command, ``safe`` sends an
// explicit zero velocity. Closing ``admitted`` after it was open does both: no
// fail-safe exists on this tier, so the runtime's own faults (a faulted joint
// group, a stop) must not leave the last non-zero velocity standing.
struct Control
{
  uint64_t seq = 0;
  uint64_t clear = 0;
  uint64_t safe = 0;
  std::string mode;
  bool allows_stream = false;
  bool admitted = false;
  bool navigation_enabled = false;
  std::optional<int64_t> arm_owner;
  std::optional<std::pair<double, double>> envelope;  // platform (max linear, max angular)
};

Control control_from_json(const std::string & text);

struct Output
{
  enum class Kind { kNone, kUpperBody, kLocomotion };
  Kind kind = Kind::kNone;
  uint32_t sequence = 0;
  int hand_sub_mode = 0;
  std::vector<double> head;
  std::vector<double> arm;
  std::vector<double> hand;
  std::array<double, 3> twist{};
};

struct JointView
{
  Aggregate joints;
  int64_t stamp_ns = 0;
};

// A merged measurement of ``order`` and how old its stalest contributing group was.
struct Measured
{
  std::vector<double> position;
  double age_s = 0.0;
};

class Core
{
public:
  explicit Core(Plan plan);

  const Plan & plan() const {return plan_;}

  // --- platform state; ``stamp_ns`` is the platform's stamp, or the host's time when it sent none
  void on_joint_state(
    const std::string & group, JointEntries entries, const std::vector<std::pair<std::string, int>> & codes,
    int64_t stamp_ns, double now);
  void on_hand_state(
    const std::array<int, 2> & types, JointEntries entries, std::array<std::map<std::string, double>, 2> measured,
    const std::vector<std::pair<std::string, int>> & codes, int64_t stamp_ns, double now);
  // The motion-control unit's arm targets by joint, from a usable command message.
  void on_arm_board(std::map<std::string, double> board, double now);

  // ``order`` from the fresh groups that carry any of it, stamped with the oldest of them; nullopt if none.
  std::optional<JointView> view(const std::vector<std::string> & order, double now) const;

  // --- control and commands
  // Applies ``control``; true when an explicit zero velocity is due: the word
  // asked for one, or it closed an admission that was open. A claim of the arm
  // (no owner -> owner) is refused while a public arm target is live: judged
  // here, where that target was accepted, it cannot slip past between the
  // status reports the runtime reads. A refused claim keeps the previous owner.
  bool on_control(const Control & control, double now);
  uint64_t control_seq() const {return control_.seq;}
  void on_stream(const Channel & channel, const std::vector<double> & values, double now);
  void on_twist(const Channel & channel, double linear_x, double linear_y, double angular_z, double now);
  void on_named_arm(int64_t owner, const std::vector<double> & positions, double now);
  // The command to send this tick, if any (runtime_node._publish_commands).
  Output tick(double now, bool remember_arm = true);
  // Optional mirror filter: prepare without committing an unverified target.
  void remember_arm_output(const Output & output);
  bool mirror_candidate(const std::string & mode, double now) const;
  bool mirror_context(const std::string & mode, double now) const;
  bool candidate_arm_is_named() const {return arm_cmd_.has_value() && arm_cmd_named_;}
  uint64_t safety_epoch() const {return safety_epoch_;}
  // An explicit zero velocity, whatever is admitted.
  Output safe_command();
  // True once, when the runtime's control went quiet: its commands are dropped and a zero is due.
  bool control_lapsed(double now);

  // --- what the control plane reads back
  // Rejection notes ("<channel>[:<reason>]") and log lines since the last take.
  std::vector<std::string> take_rejections();
  std::vector<std::string> take_warnings();
  // Fault codes per group since the last take: the last non-zero code of each joint, for
  // groups that reported at all, so a code present in one sample between takes is kept.
  std::map<std::string, std::vector<std::pair<std::string, int>>> take_faults();
  std::map<std::string, double> group_ages(double now) const;
  const std::vector<int> & hand_types() const {return hand_types_;}
  std::optional<std::vector<double>> measured(const std::vector<std::string> & order, double now) const;
  std::optional<Measured> measured_with_age(const std::vector<std::string> & order, double now) const;
  // The arm owner as applied: a claim refused above stays unowned.
  std::optional<int64_t> arm_owner() const {return control_.arm_owner;}
  uint64_t public_arm_count() const {return public_arm_count_;}
  std::optional<double> public_arm_age(double now) const;

private:
  struct Group
  {
    JointEntries entries;
    double received = 0.0;
    int64_t stamp_ns = 0;
  };

  bool fresh(const Group & group, double now) const;
  bool admitted(double now) const;
  bool channel_active(const Channel & channel, double now) const;
  void clear();
  void reject(const std::string & channel, const CommandRejected & rejected);
  void note_faults(const std::string & group, const std::vector<std::pair<std::string, int>> & codes);
  std::optional<std::vector<double>> arm_hold(double now) const;
  std::optional<std::pair<int, std::vector<double>>> hand_hold(double now) const;

  Plan plan_;
  std::map<std::string, Group> groups_;
  std::vector<int> hand_types_;
  std::array<std::map<std::string, double>, 2> hand_measured_;
  std::optional<std::map<std::string, double>> arm_board_;
  double arm_board_received_ = 0.0;
  std::map<std::string, std::map<std::string, int>> faults_;

  Control control_;
  double control_received_ = -1.0;
  bool control_lapsed_ = false;
  uint32_t sequence_ = 0;

  std::optional<std::vector<double>> arm_cmd_;
  std::optional<std::vector<double>> head_cmd_;
  std::optional<std::pair<int, std::vector<double>>> hand_cmd_;
  // The last hand command accepted in this mode, held once its channel is quiet.
  std::optional<std::pair<int, std::vector<double>>> hand_held_;
  std::optional<std::vector<double>> arm_sent_;
  std::optional<std::array<double, 3>> twist_;
  double arm_stamp_ = 0.0;
  double head_stamp_ = 0.0;
  double hand_stamp_ = 0.0;
  double twist_stamp_ = 0.0;
  std::optional<double> public_arm_stamp_;
  uint64_t public_arm_count_ = 0;
  bool hold_target_missing_ = false;
  bool hand_hold_missing_ = false;

  // Internal provenance only; no command-format or arbitration changes.
  uint64_t safety_epoch_ = 1;
  uint64_t public_arm_epoch_ = 0;
  uint64_t public_hand_epoch_ = 0;
  bool arm_cmd_named_ = false;

  std::vector<std::string> rejections_;
  std::vector<std::string> warnings_;
};

}  // namespace aimdk_dataplane

#endif  // AIMDK_DATAPLANE__CORE_HPP_
