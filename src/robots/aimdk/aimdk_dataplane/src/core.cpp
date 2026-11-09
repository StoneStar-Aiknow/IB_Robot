// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0

#include "aimdk_dataplane/core.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace aimdk_dataplane
{

using nlohmann::json;

namespace
{

Endpoint endpoint_from(const json & value)
{
  Endpoint endpoint;
  endpoint.topic = value.at("topic").get<std::string>();
  endpoint.reliability = value.value("reliability", endpoint.reliability);
  endpoint.durability = value.value("durability", endpoint.durability);
  endpoint.depth = value.value("depth", endpoint.depth);
  return endpoint;
}

std::optional<Endpoint> optional_endpoint(const json & plan, const char * key)
{
  if (!plan.contains(key) || plan.at(key).is_null()) {
    return std::nullopt;
  }
  return endpoint_from(plan.at(key));
}

View view_from(const json & value)
{
  return View{value.at("topic").get<std::string>(), value.at("order").get<std::vector<std::string>>()};
}

}  // namespace

Plan plan_from_json(const std::string & text)
{
  const json plan = json::parse(text);
  Plan out;
  out.joint_state_rate_hz = plan.at("joint_state_rate_hz").get<double>();
  out.command_rate_hz = plan.at("command_rate_hz").get<double>();
  out.status_rate_hz = plan.at("status_rate_hz").get<double>();
  out.feedback_window_s = plan.at("feedback_window_s").get<double>();
  out.stream_staleness_s = plan.at("stream_staleness_s").get<double>();
  out.cmd_vel_staleness_s = plan.at("cmd_vel_staleness_s").get<double>();
  out.control_timeout_s = plan.at("control_timeout_s").get<double>();
  out.source = plan.at("source").get<std::string>();
  for (const auto & entry : plan.at("joint_state")) {
    out.joint_state.emplace_back(entry.at("group").get<std::string>(), endpoint_from(entry));
  }
  out.hand_state = optional_endpoint(plan, "hand_state");
  out.arm_command = optional_endpoint(plan, "arm_command");
  out.upper_body_command = endpoint_from(plan.at("upper_body_command"));
  out.locomotion_command = endpoint_from(plan.at("locomotion_command"));
  for (const auto & entry : plan.at("views")) {
    out.views.push_back(view_from(entry));
  }
  out.touch = optional_endpoint(plan, "touch");
  out.touch_rate_hz = plan.value("touch_rate_hz", out.joint_state_rate_hz);
  if (plan.contains("arm_command_state") && !plan.at("arm_command_state").is_null()) {
    out.arm_command_state = view_from(plan.at("arm_command_state"));
  }
  out.arm_command_state_rate_hz = plan.value("arm_command_state_rate_hz", out.joint_state_rate_hz);
  for (const auto & entry : plan.at("channels")) {
    Channel channel;
    channel.name = entry.at("name").get<std::string>();
    channel.topic = entry.at("topic").get<std::string>();
    channel.twist = entry.value("twist", false);
    channel.joints = entry.value("joints", std::vector<std::string>{});
    const auto modes = entry.value("modes", std::vector<std::string>{});
    channel.modes = std::set<std::string>(modes.begin(), modes.end());
    channel.hand_type = entry.value("hand_type", std::string());
    out.channels.push_back(std::move(channel));
  }
  out.head_limits = plan.value("head_limits", std::map<std::string, double>{});
  const json & hand = plan.at("hand");
  out.hand.command_min = hand.value("command_min", 0.0);
  out.hand.command_max = hand.value("command_max", 1.0);
  if (hand.contains("gesture_ids") && !hand.at("gesture_ids").is_null()) {
    out.hand.gesture_ids = hand.at("gesture_ids").get<std::vector<int>>();
  }
  out.default_hand_sub_mode = hand.value("default_sub_mode", kHandSubModeClaw);
  const json & locomotion = plan.at("locomotion");
  out.locomotion.deadband = locomotion.at("deadband").get<double>();
  out.locomotion.min_linear_mps = locomotion.at("min_linear_mps").get<double>();
  out.locomotion.min_angular_radps = locomotion.at("min_angular_radps").get<double>();
  out.locomotion.max_linear_mps = locomotion.at("max_linear_mps").get<double>();
  out.locomotion.max_angular_radps = locomotion.at("max_angular_radps").get<double>();
  out.mc_actions = plan.value("mc_actions", std::map<std::string, std::string>{});
  out.navigation_gate = plan.value("navigation_gate", false);
  out.control_topic = plan.at("control_topic").get<std::string>();
  out.status_topic = plan.at("status_topic").get<std::string>();
  out.named_arm_topic = plan.at("named_arm_topic").get<std::string>();
  return out;
}

Control control_from_json(const std::string & text)
{
  const json control = json::parse(text);
  Control out;
  out.seq = control.at("seq").get<uint64_t>();
  out.clear = control.at("clear").get<uint64_t>();
  out.safe = control.at("safe").get<uint64_t>();
  out.mode = control.at("mode").get<std::string>();
  out.allows_stream = control.at("allows_stream").get<bool>();
  out.admitted = control.at("admitted").get<bool>();
  out.navigation_enabled = control.at("navigation_enabled").get<bool>();
  if (!control.at("arm_owner").is_null()) {
    out.arm_owner = control.at("arm_owner").get<int64_t>();
  }
  if (!control.at("envelope").is_null()) {
    const auto envelope = control.at("envelope").get<std::vector<double>>();
    if (envelope.size() == 2) {
      out.envelope = std::make_pair(envelope[0], envelope[1]);
    }
  }
  return out;
}

Core::Core(Plan plan)
: plan_(std::move(plan))
{
}

// --- platform state ----------------------------------------------------------

bool Core::fresh(const Group & group, double now) const
{
  return now - group.received <= plan_.feedback_window_s;
}

void Core::note_faults(const std::string & group, const std::vector<std::pair<std::string, int>> & codes)
{
  auto & seen = faults_[group];  // present, possibly empty: the group reported
  for (const auto & [name, code] : codes) {
    if (code != 0) {
      seen[name] = code;
    }
  }
}

void Core::on_joint_state(
  const std::string & group, JointEntries entries, const std::vector<std::pair<std::string, int>> & codes,
  int64_t stamp_ns, double now)
{
  groups_[group] = Group{std::move(entries), now, stamp_ns};
  note_faults(group, codes);
}

void Core::on_hand_state(
  const std::array<int, 2> & types, JointEntries entries, std::array<std::map<std::string, double>, 2> measured,
  const std::vector<std::pair<std::string, int>> & codes, int64_t stamp_ns, double now)
{
  hand_types_ = {types[0], types[1]};
  hand_measured_ = std::move(measured);
  groups_["hand"] = Group{std::move(entries), now, stamp_ns};
  note_faults("hand", codes);
}

void Core::on_arm_board(std::map<std::string, double> board, double now)
{
  arm_board_ = std::move(board);
  arm_board_received_ = now;
}

std::optional<JointView> Core::view(const std::vector<std::string> & order, double now) const
{
  const std::set<std::string> wanted(order.begin(), order.end());
  std::vector<const JointEntries *> sources;
  int64_t stamp_ns = std::numeric_limits<int64_t>::max();
  for (const auto & [name, group] : groups_) {
    if (!fresh(group, now)) {
      continue;
    }
    const bool carries = std::any_of(
      group.entries.begin(), group.entries.end(),
      [&wanted](const auto & entry) {return wanted.count(std::get<0>(entry)) > 0;});
    if (carries) {
      sources.push_back(&group.entries);
      stamp_ns = std::min(stamp_ns, group.stamp_ns);
    }
  }
  if (sources.empty()) {
    return std::nullopt;
  }
  return JointView{aggregate_joint_state(sources, order), stamp_ns};
}

std::optional<std::vector<double>> Core::measured(const std::vector<std::string> & order, double now) const
{
  std::vector<const JointEntries *> sources;
  for (const auto & [name, group] : groups_) {
    if (fresh(group, now)) {
      sources.push_back(&group.entries);
    }
  }
  auto merged = aggregate_joint_state(sources, order);
  if (merged.names.size() != order.size()) {
    return std::nullopt;  // a partial measurement is no hold target
  }
  return merged.position;
}

std::optional<Measured> Core::measured_with_age(const std::vector<std::string> & order, double now) const
{
  // The same merge as ``measured``, plus how old the stalest contributing group
  // was: the runtime adds that to the time this report spent in flight, so a
  // measurement is judged by its end-to-end age, not by when it was read.
  const std::set<std::string> wanted(order.begin(), order.end());
  std::vector<const JointEntries *> sources;
  double age_s = 0.0;
  for (const auto & [name, group] : groups_) {
    if (!fresh(group, now)) {
      continue;
    }
    const bool carries = std::any_of(
      group.entries.begin(), group.entries.end(),
      [&wanted](const auto & entry) {return wanted.count(std::get<0>(entry)) > 0;});
    if (!carries) {
      continue;
    }
    sources.push_back(&group.entries);
    age_s = std::max(age_s, now - group.received);
  }
  auto merged = aggregate_joint_state(sources, order);
  if (merged.names.size() != order.size()) {
    return std::nullopt;  // a partial measurement is no hold target
  }
  return Measured{std::move(merged.position), age_s};
}

std::map<std::string, double> Core::group_ages(double now) const
{
  std::map<std::string, double> ages;
  for (const auto & [name, group] : groups_) {
    ages[name] = now - group.received;
  }
  return ages;
}

// --- control -------------------------------------------------------------------

bool Core::on_control(const Control & control, double now)
{
  const bool first = control_received_ < 0.0;
  const bool clear_asked = !first && control.clear != control_.clear;
  const bool safe_asked = !first && control.safe != control_.safe;
  // An admission that was open and closed is a revoke of everything output: the
  // runtime closes it for its own faults (a faulted joint group) and for stops,
  // without counting a safe, and its heartbeat keeps this control fresh so the
  // lapse below never fires. Without this the last non-zero velocity would
  // stand until the fault cleared, and then resume on the stale cache.
  const bool revoked = !first && control_.admitted && !control.admitted;
  // A named motion may take the arm only from a public stream that has gone
  // quiet. Judged here, where the public target was accepted, it cannot slip
  // past between the status reports the runtime decides on.
  const bool claiming = control.arm_owner.has_value() && !control_.arm_owner.has_value();
  const bool public_stream_live =
    public_arm_stamp_ && now - *public_arm_stamp_ <= plan_.stream_staleness_s;
  const bool claim_refused = !first && claiming && public_stream_live;
  Control applied = control;
  if (claim_refused) {
    applied.arm_owner = control_.arm_owner;
    rejections_.push_back("arm_claim:PUBLIC_ARM_STREAM_ACTIVE");
    warnings_.push_back("refused a named motion's arm claim: the public arm stream is live");
  }
  control_ = applied;
  control_received_ = now;
  control_lapsed_ = false;
  if (clear_asked || first || revoked) {
    clear();
  }
  return safe_asked || revoked || (first && control.safe > 0);
}

bool Core::control_lapsed(double now)
{
  if (control_received_ < 0.0 || control_lapsed_ || now - control_received_ <= plan_.control_timeout_s) {
    return false;
  }
  control_lapsed_ = true;
  clear();
  warnings_.push_back("the runtime's control went quiet: commands dropped and a zero velocity sent");
  return true;
}

bool Core::admitted(double now) const
{
  return control_received_ >= 0.0 && now - control_received_ <= plan_.control_timeout_s && control_.admitted;
}

bool Core::channel_active(const Channel & channel, double now) const
{
  if (!admitted(now) || !channel.modes.count(control_.mode)) {
    return false;
  }
  if (channel.twist && plan_.navigation_gate) {
    return control_.navigation_enabled;
  }
  return true;
}

void Core::clear()
{
  arm_cmd_.reset();
  hand_cmd_.reset();
  hand_held_.reset();
  arm_sent_.reset();
  head_cmd_.reset();
  twist_.reset();
}

void Core::reject(const std::string & channel, const CommandRejected & rejected)
{
  rejections_.push_back(channel + ":" + rejected.reason);
  warnings_.push_back("rejected " + channel + " command: " + rejected.what());
}

// --- command channels ------------------------------------------------------------

void Core::on_stream(const Channel & channel, const std::vector<double> & values, double now)
{
  if (!channel_active(channel, now)) {
    rejections_.push_back(channel.name);
    return;
  }
  try {
    if (channel.name == "arm_stream") {
      auto payload = arm_payload(values, channel.joints);
      if (control_.arm_owner) {
        rejections_.push_back(channel.name + ":NAMED_MOTION_ACTIVE");
        return;
      }
      arm_cmd_ = std::move(payload);
      arm_stamp_ = now;
      public_arm_stamp_ = now;
      ++public_arm_count_;
    } else if (channel.name == "head_stream") {
      head_cmd_ = head_payload(values, channel.joints, plan_.head_limits);
      head_stamp_ = now;
    } else if (!channel.hand_type.empty()) {
      hand_cmd_ = hand_payload(values, channel.hand_type, plan_.hand);
      hand_held_ = hand_cmd_;
      hand_stamp_ = now;
    } else {
      rejections_.push_back(channel.name);
    }
  } catch (const CommandRejected & rejected) {
    reject(channel.name, rejected);
  }
}

void Core::on_twist(const Channel & channel, double linear_x, double linear_y, double angular_z, double now)
{
  if (!channel_active(channel, now)) {
    rejections_.push_back(channel.name);
    return;
  }
  try {
    // The platform narrows its own speed envelope with battery, load and terrain.
    twist_ = twist_to_locomotion(
      linear_x, linear_y, angular_z, plan_.locomotion.with_platform_envelope(control_.envelope));
    twist_stamp_ = now;
  } catch (const CommandRejected & rejected) {
    reject(channel.name, rejected);
  }
}

void Core::on_named_arm(int64_t owner, const std::vector<double> & positions, double now)
{
  if (!admitted(now) || control_.arm_owner != owner || positions.size() != kVendorArmOrder.size()) {
    return;
  }
  arm_cmd_ = positions;
  arm_stamp_ = now;
}

// --- fixed-rate publication --------------------------------------------------------

std::optional<std::vector<double>> Core::arm_hold(double now) const
{
  // The controller's own targets, not the measured arm: loaded, it settles below
  // its target, so a measured pose sent back as the target sinks it every tick.
  if (!plan_.arm_command) {
    return measured(kVendorArmOrder, now);
  }
  if (arm_board_ && now - arm_board_received_ <= plan_.feedback_window_s) {
    std::vector<double> out;
    for (const auto & name : kVendorArmOrder) {
      const auto found = arm_board_->find(name);
      if (found == arm_board_->end()) {
        out.clear();
        break;
      }
      out.push_back(found->second);
    }
    if (out.size() == kVendorArmOrder.size()) {
      return out;
    }
  }
  return arm_sent_;
}

std::optional<std::pair<int, std::vector<double>>> Core::hand_hold(double now) const
{
  if (hand_held_) {
    return hand_held_;
  }
  const auto action = plan_.mc_actions.find(control_.mode);
  if (plan_.default_hand_sub_mode != kHandSubModeDexterousJoint || action == plan_.mc_actions.end() ||
    action->second != kUpperBodyHandsAction)
  {
    return std::make_pair(plan_.default_hand_sub_mode, std::vector<double>{});
  }
  std::array<bool, 2> installed{false, false};
  for (size_t side = 0; side < std::min<size_t>(2, hand_types_.size()); ++side) {
    installed[side] = hand_type_family(hand_types_[side]).has_value();
  }
  if (!installed[0] && !installed[1]) {
    return std::make_pair(plan_.default_hand_sub_mode, std::vector<double>{});
  }
  const auto hand = groups_.find("hand");
  if (hand == groups_.end() || !fresh(hand->second, now)) {
    return std::nullopt;
  }
  auto payload = dexterous_hold_payload(installed, hand_measured_);
  if (!payload) {
    return std::nullopt;
  }
  return std::make_pair(plan_.default_hand_sub_mode, std::move(*payload));
}

Output Core::tick(double now)
{
  Output out;
  if (!control_.allows_stream || !admitted(now)) {
    return out;
  }
  out.sequence = ++sequence_;
  const auto action = plan_.mc_actions.find(control_.mode);
  const bool locomotion = action != plan_.mc_actions.end() &&
    (action->second == "STAND_DEFAULT" || action->second == "LOCOMOTION_DEFAULT" ||
    action->second == "LOCOMOTION_STEP");
  if (locomotion) {
    if (!twist_) {
      return out;
    }
    const bool gated = plan_.navigation_gate && !control_.navigation_enabled;
    out.kind = Output::Kind::kLocomotion;
    out.twist = *twist_;
    if (gated || now - twist_stamp_ > plan_.cmd_vel_staleness_s) {
      out.twist = {0.0, 0.0, 0.0};
      twist_.reset();
    }
    return out;
  }
  // Each channel expires on its own clock.
  if (arm_cmd_ && now - arm_stamp_ > plan_.stream_staleness_s) {
    arm_cmd_.reset();
  }
  if (head_cmd_ && now - head_stamp_ > plan_.stream_staleness_s) {
    head_cmd_.reset();
  }
  if (hand_cmd_ && now - hand_stamp_ > plan_.stream_staleness_s) {
    hand_cmd_.reset();
  }
  if (!arm_cmd_ && !head_cmd_ && !hand_cmd_) {
    return out;
  }
  // Every element of arm_pos/head_pos is a target: an uncommanded joint is told to
  // stay where it is, and without that measurement nothing is sent.
  const auto arm = arm_cmd_ ? arm_cmd_ : arm_hold(now);
  const auto head = head_cmd_ ? head_cmd_ : measured(kVendorHeadOrder, now);
  if (!arm || !head) {
    if (!hold_target_missing_) {
      hold_target_missing_ = true;
      rejections_.push_back("upper_body:HOLD_TARGET_UNAVAILABLE");
      warnings_.push_back("upper-body command withheld: no fresh measurement to hold the uncommanded joints");
    }
    return out;
  }
  hold_target_missing_ = false;
  const auto hand = hand_cmd_ ? hand_cmd_ : hand_hold(now);
  if (!hand) {
    if (!hand_hold_missing_) {
      hand_hold_missing_ = true;
      rejections_.push_back("upper_body:HAND_HOLD_UNAVAILABLE");
      warnings_.push_back("upper-body command withheld: the installed hands cannot be held at a measured pose");
    }
    return out;
  }
  hand_hold_missing_ = false;
  arm_sent_ = arm;
  out.kind = Output::Kind::kUpperBody;
  out.arm = *arm;
  out.head = *head;
  out.hand_sub_mode = hand->first;
  out.hand = hand->second;
  return out;
}

Output Core::safe_command()
{
  Output out;
  out.kind = Output::Kind::kLocomotion;
  out.sequence = ++sequence_;
  return out;
}

// --- read back ---------------------------------------------------------------------

std::vector<std::string> Core::take_rejections()
{
  return std::exchange(rejections_, {});
}

std::vector<std::string> Core::take_warnings()
{
  return std::exchange(warnings_, {});
}

std::map<std::string, std::vector<std::pair<std::string, int>>> Core::take_faults()
{
  std::map<std::string, std::vector<std::pair<std::string, int>>> out;
  for (const auto & [group, codes] : faults_) {
    auto & listed = out[group];
    for (const auto & [name, code] : codes) {
      listed.emplace_back(name, code);
    }
  }
  faults_.clear();
  return out;
}

std::optional<double> Core::public_arm_age(double now) const
{
  if (!public_arm_stamp_) {
    return std::nullopt;
  }
  return now - *public_arm_stamp_;
}

}  // namespace aimdk_dataplane
