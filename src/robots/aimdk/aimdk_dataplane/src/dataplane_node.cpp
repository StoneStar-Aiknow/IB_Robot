// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// aimdk_dataplane: the X2 runtime's data plane, in its own native process.
//
// Takes every joint, hand and arm command sample the HAL publishes (~2800 per
// second across the topics), publishes /joint_states and the profile's other state
// views, the hand touch arrays and the arm command echo, and owns the fixed-rate
// command stream to the motion-control unit: the public command channels in, the
// upper-body and locomotion commands out. aimdk_runtime keeps the control plane and
// says what is admitted on the control topic; this process reports back on the
// status topic (feedback ages, faults, rejections, the measured arm).
//
// See aimdk_dataplane/core.hpp for the rules and why they moved here.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "aimdk_dataplane/core.hpp"
#include "aimdk_dataplane/cbf_safety.hpp"
#include "aimdk_dataplane/template_play.hpp"
#include "aimdk_dataplane/tick_timing.hpp"
#include "cbf/proxy_model.hpp"
#include "aimdk_msgs/msg/hand_state_array.hpp"
#include "aimdk_msgs/msg/joint_command_array.hpp"
#include "aimdk_msgs/msg/joint_state_array.hpp"
#include "aimdk_msgs/msg/mc_locomotion_velocity.hpp"
#include "aimdk_msgs/msg/message_header.hpp"
#include "aimdk_msgs/msg/upper_body_command_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "ibrobot_msgs/msg/joint_command_state.hpp"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"

namespace aimdk_dataplane
{

namespace
{

using aimdk_msgs::msg::HandStateArray;
using aimdk_msgs::msg::JointCommandArray;
using aimdk_msgs::msg::JointStateArray;

// Queue each vendor input deep enough to ride out a late wakeup of this process;
// the callbacks keep up with the platform otherwise.
constexpr size_t kVendorInputDepth = 64;

std::atomic<bool> g_stop{false};

void on_signal(int)
{
  g_stop = true;
}

double monotonic_s()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

rclcpp::QoS vendor_qos(const Endpoint & endpoint, size_t depth)
{
  rclcpp::QoS qos{rclcpp::KeepLast(depth)};
  qos.reliability(
    endpoint.reliability == "reliable" ? rclcpp::ReliabilityPolicy::Reliable :
    rclcpp::ReliabilityPolicy::BestEffort);
  qos.durability(
    endpoint.durability == "transient_local" ? rclcpp::DurabilityPolicy::TransientLocal :
    rclcpp::DurabilityPolicy::Volatile);
  return qos;
}

// The runtime's QoS for what it publishes on the public surface (runtime_node).
rclcpp::QoS reliable_qos() {return rclcpp::QoS(10).reliable().durability_volatile();}
rclcpp::QoS sensor_qos() {return rclcpp::QoS(10).best_effort().durability_volatile();}
// A command channel carries a target, so only the newest one is ever wanted.
rclcpp::QoS command_qos() {return rclcpp::QoS(1).best_effort().durability_volatile();}

int64_t to_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

// Times a callback from its entry to its return, its wait for the lock included.
// Nothing at all when timing is off.
class Timed
{
public:
  Timed(std::optional<TickTiming> & timing, TimingSegment segment)
  : timing_(timing), segment_(segment), begun_(timing ? timing->now() : TickTiming::Clock::time_point{}) {}
  ~Timed()
  {
    if (timing_) {timing_->add(segment_, TickTiming::ms(timing_->now() - begun_));}
  }
  Timed(const Timed &) = delete;
  Timed & operator=(const Timed &) = delete;

private:
  std::optional<TickTiming> & timing_;
  TimingSegment segment_;
  TickTiming::Clock::time_point begun_;
};

}  // namespace

class Dataplane : public rclcpp::Node
{
public:
  Dataplane()
  : Node("aimdk_dataplane"),
    core_(plan_from_json(declare_parameter<std::string>("plan_json", "")))
  {
    const Plan & plan = core_.plan();
    const auto cbf_text = declare_parameter<std::string>("cbf_safety_json", "");
    if (!cbf_text.empty()) {
      auto config = CbfSafetyConfig::from_json(cbf_text, plan.command_rate_hz);
      if (config.enabled) {
        cbf_.emplace(std::move(config));
        RCLCPP_INFO(
          get_logger(), "CBF enabled: mode=%s mirror_mode=%s mapping_verified=%s budget_us=%.0f",
          cbf_->config().enforce ? "enforce" : "shadow", cbf_->config().mirror_mode.c_str(),
          cbf_->config().mapping_verified ? "true" : "false", cbf_->config().deadline_us);
        if (cbf_->config().enforce && cbf_->config().mapping_available && !cbf_->config().mapping_verified)
        {
          RCLCPP_WARN(get_logger(),
            "CBF enforcing with the configured candidate thumb mapping; mapping_verified=false");
        }
      }
    }
    setup_template_play(declare_parameter<std::string>("template_play_json", ""));
    // Observation only (tick_timing.hpp): off, the status and the INFO line are as before.
    if (declare_parameter<bool>("tick_timing", true)) {
      timing_.emplace(1.0 / std::max(plan.command_rate_hz, 1.0));
    }
    // Each vendor input is taken on a thread of its own (see intake_groups): the
    // 500 Hz groups alone are more than one rclcpp thread keeps up with on the
    // loaded 310P.
    for (const auto & [group, endpoint] : plan.joint_state) {
      subscriptions_.push_back(
        create_subscription<JointStateArray>(
          endpoint.topic, vendor_qos(endpoint, kVendorInputDepth),
          [this, group = group](JointStateArray::ConstSharedPtr message) {on_joint_state(group, *message);},
          intake_options()));
    }
    if (plan.hand_state) {
      subscriptions_.push_back(
        create_subscription<HandStateArray>(
          plan.hand_state->topic, vendor_qos(*plan.hand_state, kVendorInputDepth),
          [this](HandStateArray::ConstSharedPtr message) {on_hand_state(message);}, intake_options()));
    }
    if (plan.arm_command && plan.arm_command_state) {
      subscriptions_.push_back(
        create_subscription<JointCommandArray>(
          plan.arm_command->topic, vendor_qos(*plan.arm_command, kVendorInputDepth),
          [this](JointCommandArray::ConstSharedPtr message) {on_arm_command(*message);}, intake_options()));
      echo_pub_ = create_publisher<ibrobot_msgs::msg::JointCommandState>(plan.arm_command_state->topic, sensor_qos());
      echo_timer_ = timer(plan.arm_command_state_rate_hz, [this]() {publish_echo();});
    }
    for (const auto & view : plan.views) {
      views_.emplace_back(create_publisher<sensor_msgs::msg::JointState>(view.topic, reliable_qos()), view.order);
    }
    if (plan.touch && plan.hand_state) {
      touch_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(plan.touch->topic, sensor_qos());
      touch_timer_ = timer(plan.touch_rate_hz, [this]() {publish_touch();});
    }
    upper_body_pub_ = create_publisher<aimdk_msgs::msg::UpperBodyCommandArray>(
      plan.upper_body_command.topic, vendor_qos(plan.upper_body_command, plan.upper_body_command.depth));
    locomotion_pub_ = create_publisher<aimdk_msgs::msg::McLocomotionVelocity>(
      plan.locomotion_command.topic, vendor_qos(plan.locomotion_command, plan.locomotion_command.depth));
    for (const auto & channel : plan.channels) {
      if (channel.twist) {
        subscriptions_.push_back(
          create_subscription<geometry_msgs::msg::Twist>(
            channel.topic, command_qos(), [this, &channel](geometry_msgs::msg::Twist::ConstSharedPtr message) {
              const Timed timed(timing_, TimingSegment::kStream);
              std::lock_guard<std::recursive_mutex> lock(mutex_);
              core_.on_twist(channel, message->linear.x, message->linear.y, message->angular.z, monotonic_s());
              flush_notes();
            }));
      } else {
        subscriptions_.push_back(
          create_subscription<std_msgs::msg::Float64MultiArray>(
            channel.topic, command_qos(),
            [this, &channel](std_msgs::msg::Float64MultiArray::ConstSharedPtr message) {
              const Timed timed(timing_, TimingSegment::kStream);
              std::lock_guard<std::recursive_mutex> lock(mutex_);
              core_.on_stream(channel, message->data, monotonic_s());
              flush_notes();
            }));
      }
    }
    // Control and status are the runtime's own wiring to this process: reliable, and
    // the control latched so a restart of either side picks up the current word.
    subscriptions_.push_back(
      create_subscription<std_msgs::msg::String>(
        plan.control_topic, rclcpp::QoS(1).reliable().transient_local(),
        [this](std_msgs::msg::String::ConstSharedPtr message) {on_control(message->data);}));
    subscriptions_.push_back(
      create_subscription<std_msgs::msg::Float64MultiArray>(
        plan.named_arm_topic, rclcpp::QoS(1).reliable(),
        [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr message) {
          if (!message->data.empty()) {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            core_.on_named_arm(
              static_cast<int64_t>(message->data.front()),
              std::vector<double>(message->data.begin() + 1, message->data.end()), monotonic_s());
          }
        }));
    status_pub_ = create_publisher<std_msgs::msg::String>(plan.status_topic, rclcpp::QoS(20).reliable());
    joint_timer_ = timer(plan.joint_state_rate_hz, [this]() {publish_joint_states();});
    command_timer_ = timer(plan.command_rate_hz, [this]() {publish_commands();});
    status_timer_ = timer(plan.status_rate_hz, [this]() {publish_status();});
    RCLCPP_INFO(
      get_logger(), "data plane up: %zu joint group(s)%s%s, %zu view(s), %zu command channel(s)",
      plan.joint_state.size(), plan.hand_state ? " + hands" : "", plan.arm_command_state ? " + arm command echo" : "",
      plan.views.size(), plan.channels.size());
  }

  // An explicit zero velocity before this process stops publishing: the vendor's
  // command tier has no fail-safe.
  void publish_safe_command()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    publish(core_.safe_command());
  }

  // The vendor inputs' callback groups, each to be spun on a thread of its own.
  const std::vector<rclcpp::CallbackGroup::SharedPtr> & intake_groups() const {return intake_groups_;}

private:
  // The template playback port (template_play.hpp), off unless the parameter
  // enables it: off, there is no player, no subscription and no status topic.
  void setup_template_play(const std::string & text)
  {
    if (text.empty()) {
      return;
    }
    nlohmann::json settings;
    try {
      settings = nlohmann::json::parse(text);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "template playback stays off, template_play_json is malformed: %s", error.what());
      return;
    }
    if (!settings.is_object()) {
      RCLCPP_ERROR(get_logger(), "template playback stays off, template_play_json is not an object");
      return;
    }
    const auto enabled = settings.find("enabled");
    if (enabled != settings.end() && !enabled->is_boolean()) {
      RCLCPP_ERROR(get_logger(), "template playback stays off, enabled is not true or false");
      return;
    }
    if (enabled == settings.end() || !enabled->get<bool>()) {
      return;
    }
    // Any setting of the wrong type leaves the port off; it never stops the node.
    try {
      const auto path = settings.value("segments_file", std::string());
      TemplateWhitelist whitelist;
      bool loaded = false;
      if (cbf_) {
        try {
          std::ifstream file(path);
          if (!file) {
            throw std::runtime_error("cannot read " + path);
          }
          std::stringstream contents;
          contents << file.rdbuf();
          whitelist = TemplateWhitelist::from_json(contents.str(), cbf_->config());
          loaded = true;
        } catch (const std::exception & error) {
          RCLCPP_ERROR(get_logger(), "template whitelist not loaded, every start is refused: %s", error.what());
        }
        for (const auto & rejected : whitelist.rejected) {
          RCLCPP_WARN(get_logger(), "template segment left out: %s", rejected.c_str());
        }
      } else {
        RCLCPP_WARN(get_logger(), "template playback enabled without the CBF: every start is refused as disabled");
      }
      const auto status_topic = settings.value("status_topic", std::string("/aimdk_dataplane/template_status"));
      const auto command_topic = settings.value("command_topic", std::string("/mirror_session/template_play"));
      player_.emplace(
        cbf_ ? std::optional<CbfSafetyConfig>(cbf_->config()) : std::nullopt, std::move(whitelist), loaded);
      template_status_pub_ = create_publisher<std_msgs::msg::String>(status_topic, rclcpp::QoS(10).reliable());
      subscriptions_.push_back(
        create_subscription<std_msgs::msg::String>(
          command_topic, rclcpp::QoS(10).reliable(),
          [this](std_msgs::msg::String::ConstSharedPtr message) {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            try {
              player_->on_message(message->data, monotonic_s());
            } catch (const std::exception & error) {
              RCLCPP_ERROR(get_logger(), "template command dropped: %s", error.what());
            }
          }));
      RCLCPP_INFO(
        get_logger(), "template playback enabled: %zu segment(s) from %s", player_->whitelist().segments.size(),
        path.c_str());
    } catch (const std::exception & error) {
      player_.reset();
      template_status_pub_.reset();
      RCLCPP_ERROR(get_logger(), "template playback stays off: %s", error.what());
    }
  }

  rclcpp::SubscriptionOptions intake_options()
  {
    intake_groups_.push_back(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false));
    rclcpp::SubscriptionOptions options;
    options.callback_group = intake_groups_.back();
    return options;
  }

  template<typename Callback>
  rclcpp::TimerBase::SharedPtr timer(double rate_hz, Callback callback)
  {
    return create_wall_timer(std::chrono::duration<double>(1.0 / std::max(rate_hz, 1.0)), callback);
  }

  // The platform's stamp (meas_stamp if set, else stamp), or this host's time when it sent neither.
  int64_t stamp_ns(const aimdk_msgs::msg::MessageHeader & header)
  {
    for (const auto * stamp : {&header.meas_stamp, &header.stamp}) {
      if (stamp->sec || stamp->nanosec) {
        return to_ns(*stamp);
      }
    }
    return now().nanoseconds();
  }

  void on_joint_state(const std::string & group, const JointStateArray & message)
  {
    JointEntries entries;
    std::vector<std::pair<std::string, int>> codes;
    entries.reserve(message.joints.size());
    for (const auto & joint : message.joints) {
      entries.emplace_back(joint.name, joint.position, joint.velocity, joint.effort);
      codes.emplace_back(joint.name, joint.error_code);
    }
    const int64_t stamp = stamp_ns(message.header);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    note_intake(group, stamp);
    core_.on_joint_state(group, std::move(entries), codes, stamp, monotonic_s());
  }

  // How old each group's samples were when this process took them, by the platform's
  // stamp (PTP clock): the most since the previous status. A queue that backs up shows
  // here, apart from what the platform or the network added.
  void note_intake(const std::string & group, int64_t stamp)
  {
    const double lag_ms = static_cast<double>(now().nanoseconds() - stamp) * 1e-6;
    auto & worst = intake_lag_ms_[group];
    worst = std::max(worst, lag_ms);
  }

  void on_hand_state(const HandStateArray::ConstSharedPtr & message)
  {
    JointEntries entries;
    std::array<std::map<std::string, double>, 2> measured;
    std::vector<std::pair<std::string, int>> codes;
    const std::array<std::pair<const char *, const decltype(message->left_hands) *>, 2> sides = {{
      {"left", &message->left_hands}, {"right", &message->right_hands},
    }};
    for (size_t side = 0; side < sides.size(); ++side) {
      for (const auto & hand : *sides[side].second) {
        const std::string name = hand.name.empty() ? std::string(sides[side].first) + "_hand" : hand.name;
        entries.emplace_back(name, hand.position, hand.velocity, hand.effort);
        measured[side][name] = hand.position;
        codes.emplace_back(name, hand.faultcode);
      }
    }
    const int64_t stamp = stamp_ns(message->header);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::string faulted;
    for (const auto & [name, code] : codes) {
      if (code != 0) {faulted += " " + name + "=" + std::to_string(code);}
    }
    if (!faulted.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "HAND_FAULT%s", faulted.c_str());
    }
    note_intake("hand", stamp);
    core_.on_hand_state(
      {message->left_hand_type.value, message->right_hand_type.value}, std::move(entries), std::move(measured),
      codes, stamp, monotonic_s());
    hand_ = message;
    hand_fresh_ = true;
    note_phantom_transitions();
  }

  // An installed side reported as no hand is checked as a phantom hand. That is
  // right for an empty flange but also hides a real hand that dropped out, so
  // it is logged to be visible in the field: the start at once, the end only
  // once the hand has been reported for kPhantomEndAfterS without a break, so
  // a type flickering between 0 and 1 logs one episode, not every flip.
  void note_phantom_transitions()
  {
    if (!cbf_) {return;}
    static constexpr double kPhantomEndAfterS = 0.5;
    static const std::array<const char *, 2> side_names{"left", "right"};
    const auto phantom = phantom_hands();
    const double now = monotonic_s();
    for (size_t side = 0; side < phantom.size(); ++side) {
      if (phantom[side]) {
        present_since_[side].reset();
        if (!phantom_logged_[side]) {
          phantom_logged_[side] = true;
          RCLCPP_WARN(get_logger(), "CBF_HAND_ABSENT side=%s start: installed hand reported as type 0, "
            "checked as a phantom hand", side_names[side]);
        }
      } else if (phantom_logged_[side]) {
        if (!present_since_[side]) {present_since_[side] = now;}
        if (now - *present_since_[side] >= kPhantomEndAfterS) {
          phantom_logged_[side] = false;
          present_since_[side].reset();
          RCLCPP_INFO(get_logger(), "CBF_HAND_ABSENT side=%s end", side_names[side]);
        }
      }
    }
  }

  // What the motion-control unit is driving the arm to. A message with a
  // non-finite value, a joint named twice or a declared joint missing is no
  // target at all: half a target is not the controller's target.
  void on_arm_command(const JointCommandArray & message)
  {
    std::map<std::string, std::array<double, 3>> entries;
    for (const auto & joint : message.joints) {
      const std::array<double, 3> values = {joint.position, joint.stiffness, joint.damping};
      if (joint.name.empty() || entries.count(joint.name) ||
        !std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);}))
      {
        return;
      }
      entries[joint.name] = values;
    }
    const auto & order = core_.plan().arm_command_state->order;
    auto echo = std::make_unique<ibrobot_msgs::msg::JointCommandState>();
    std::map<std::string, double> board;
    for (const auto & name : order) {
      const auto found = entries.find(name);
      if (found == entries.end()) {
        return;
      }
      echo->name.push_back(name);
      echo->position.push_back(found->second[0]);
      echo->stiffness.push_back(found->second[1]);
      echo->damping.push_back(found->second[2]);
      board[name] = found->second[0];
    }
    echo->header.stamp = rclcpp::Time(stamp_ns(message.header));
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    core_.on_arm_board(std::move(board), monotonic_s());
    echo_ = std::move(echo);
  }

  void on_control(const std::string & text)
  {
    const Timed timed(timing_, TimingSegment::kControl);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    Control control;
    try {
      control = control_from_json(text);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "ignoring a malformed control word: %s", error.what());
      return;
    }
    const auto old_epoch = core_.safety_epoch();
    const bool safe_due = core_.on_control(control, monotonic_s());
    if (cbf_ && old_epoch != core_.safety_epoch()) {
      cbf_->inactive(core_.safety_epoch());
    }
    if (player_ && old_epoch != core_.safety_epoch()) {
      player_->stop("epoch_changed");
      player_->reject_pending("not_admitted");
    }
    if (safe_due) {
      publish_safe_command();
    }
    publish_status();  // the acknowledgement the runtime waits for
  }

  void publish_joint_states()
  {
    const Timed timed(timing_, TimingSegment::kJointStates);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const double now_s = monotonic_s();
    for (const auto & [publisher, order] : views_) {
      auto view = core_.view(order, now_s);
      if (!view) {
        continue;
      }
      sensor_msgs::msg::JointState message;
      message.header.stamp = rclcpp::Time(view->stamp_ns);
      message.name = std::move(view->joints.names);
      message.position = std::move(view->joints.position);
      message.velocity = std::move(view->joints.velocity);
      message.effort = std::move(view->joints.effort);
      publisher->publish(message);
    }
  }

  // The touch arrays ride on the hand state message. Only a hand of the dexterous
  // family has them; a gripper's all-zero fields are no sensor, not "no contact".
  void publish_touch()
  {
    const Timed timed(timing_, TimingSegment::kTouch);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!hand_fresh_ || !hand_) {
      return;
    }
    hand_fresh_ = false;
    std::vector<std::pair<std::string, const uint8_t *>> pads;
    std::vector<std::pair<std::string, size_t>> sizes;
    const auto add_side = [&](const std::string & side, int type, const aimdk_msgs::msg::HandTouchSensorData & data) {
        if (hand_type_family(type) != std::optional<std::string>("dexterous")) {
          return;
        }
        pads.emplace_back(side + "_palm", data.palm_touch_data.data());
        sizes.emplace_back(side + "_palm", data.palm_touch_data.size());
        pads.emplace_back(side + "_back_of_hand", data.back_of_hand_touch_data.data());
        sizes.emplace_back(side + "_back_of_hand", data.back_of_hand_touch_data.size());
        const std::array<std::pair<const char *, const std::array<uint8_t, 16> *>, 5> tips = {{
          {"thumb", &data.thumb_touch_data}, {"index_finger", &data.index_finger_touch_data},
          {"middle_finger", &data.middle_finger_touch_data}, {"ring_finger", &data.ring_finger_touch_data},
          {"little_finger", &data.little_finger_touch_data},
        }};
        for (const auto & [pad, cells] : tips) {
          pads.emplace_back(side + "_" + pad, cells->data());
          sizes.emplace_back(side + "_" + pad, cells->size());
        }
      };
    add_side("left", hand_->left_hand_type.value, hand_->left_touch_sensors);
    add_side("right", hand_->right_hand_type.value, hand_->right_touch_sensors);
    if (pads.empty()) {
      return;
    }
    // Pads sorted by name, each a layout dimension carrying its size (runtime_node._publish_hand_touch).
    std::vector<size_t> index(pads.size());
    for (size_t i = 0; i < index.size(); ++i) {
      index[i] = i;
    }
    std::sort(index.begin(), index.end(), [&pads](size_t a, size_t b) {return pads[a].first < pads[b].first;});
    std_msgs::msg::Float64MultiArray array;
    size_t offset = 0;
    for (const auto & [name, size] : sizes) {
      offset += size;
    }
    for (size_t i : index) {
      std_msgs::msg::MultiArrayDimension dimension;
      dimension.label = pads[i].first;
      dimension.size = static_cast<uint32_t>(sizes[i].second);
      dimension.stride = static_cast<uint32_t>(offset);
      offset -= sizes[i].second;
      array.layout.dim.push_back(dimension);
      array.data.insert(array.data.end(), pads[i].second, pads[i].second + sizes[i].second);
    }
    touch_pub_->publish(array);
  }

  void publish_echo()
  {
    const Timed timed(timing_, TimingSegment::kEcho);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (echo_) {
      echo_pub_->publish(std::move(echo_));
    }
  }

  void publish_commands()
  {
    command_tick();
    if (player_) {
      const Timed timed(timing_, TimingSegment::kTemplateStatus);
      std::lock_guard<std::recursive_mutex> lock(mutex_);
      publish_template_status(monotonic_s());
    }
  }

  void command_tick()
  {
    if (timing_) {
      trace_ = TickTrace{};
      trace_.entered = timing_->now();
      trace_.at_entered = timing_->thread_times();
    }
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    const auto tick_started = std::chrono::steady_clock::now();
    run_tick(lock, tick_started);
    if (timing_) {
      trace_.locked = tick_started;
      trace_.at_ended = timing_->thread_times();
      trace_.ended = timing_->now();
      timing_->tick(trace_, cbf_ ? &cbf_->diagnostics() : nullptr);
    }
  }

  // A clock reading for the tick's trace, when timing is on.
  void mark(TickTrace::Clock::time_point & at)
  {
    if (timing_) {at = timing_->now();}
  }

  void mark_cpu(std::optional<double> & cpu_ms)
  {
    if (timing_) {cpu_ms = timing_->thread_cpu_ms();}
  }

  void run_tick(
    std::unique_lock<std::recursive_mutex> & lock, std::chrono::steady_clock::time_point tick_started)
  {
    const double now_s = std::chrono::duration<double>(tick_started.time_since_epoch()).count();
    // Read once: the check latches the lapse and clears, so a second read sees no lapse.
    const bool lapsed = core_.control_lapsed(now_s);
    if (lapsed) {
      publish_safe_command();
    }
    const bool mirror_context = cbf_ && core_.mirror_context(cbf_->config().mirror_mode, now_s);
    const bool source_ready = cbf_ && core_.mirror_candidate(cbf_->config().mirror_mode, now_s);
    const bool named_only = core_.candidate_arm_is_named() && !source_ready;
    if (player_ && player_->playing() && play_template(now_s, lapsed, mirror_context, named_only)) {
      return;
    }
    if (!cbf_ || !mirror_context || named_only) {
      if (player_) {player_->reject_pending(cbf_ ? "not_admitted" : "disabled");}
      if (cbf_) {cbf_->inactive(core_.safety_epoch());}
      trace_.path = TickPath::kPlain;
      publish(core_.tick(now_s));
      flush_notes();
      return;
    }
    CbfSnapshot snapshot;
    snapshot.publication_deadline = tick_started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::micro>(cbf_->config().deadline_us));
    snapshot.requested = core_.tick(now_s, false);
    // A missing uncommanded-channel measurement can suppress the Core
    // candidate. Enforce may still retransmit its committed complete pair;
    // context/hardware exits remain checked before publication below; source
    // freshness loss can use that same committed-pair fallback.
    if (snapshot.requested.kind != Output::Kind::kUpperBody &&
      (!cbf_->config().enforce || snapshot.requested.kind != Output::Kind::kNone))
    {
      if (player_) {player_->reject_pending("not_upper_body");}
      flush_notes();
      return;
    }
    snapshot.epoch = core_.safety_epoch();
    snapshot.named_arm_cache = core_.candidate_arm_is_named();
    snapshot.source_ready = source_ready;
    snapshot.hand_configuration_matches = cbf_hands_match();
    static const std::array<std::string, 4> all_thumb_names{
      "L_thumb_roll_joint", "L_thumb_abad_joint", "R_thumb_roll_joint", "R_thumb_abad_joint"};
    const auto state = [&](const std::vector<std::string> & order, double at) {
        auto measured = core_.measured_with_age(order, at);
        const auto view = core_.view(order, at);
        if (measured && view && view->stamp_ns > 0) {
          const double stamp_age = static_cast<double>(now().nanoseconds() - view->stamp_ns) * 1e-9;
          measured->age_s = stamp_age < -0.005 ? std::numeric_limits<double>::infinity() :
            std::max(measured->age_s, stamp_age);
        }
        return measured;
      };
    snapshot.arm = state(kVendorArmOrder, now_s);
    snapshot.hand_reported_absent = phantom_hands();
    {
      // Each installed side is read on its own, so a side without readings (an
      // empty flange, a dropout) leaves the other side's readings in use. Its
      // entries are NaN; no readings on any installed side is no thumb at all.
      Measured thumb;
      bool any = false;
      for (size_t side = 0; side < 2; ++side) {
        if (!cbf_->config().hands_installed[side]) {continue;}
        const auto reading = snapshot.hand_reported_absent[side] ? std::nullopt :
          state({all_thumb_names[2 * side], all_thumb_names[2 * side + 1]}, now_s);
        for (size_t axis = 0; axis < 2; ++axis) {
          thumb.position.push_back(reading ? reading->position[axis] : std::numeric_limits<double>::quiet_NaN());
        }
        if (reading) {
          thumb.age_s = any ? std::max(thumb.age_s, reading->age_s) : reading->age_s;
          any = true;
        }
      }
      if (any) {snapshot.thumb = std::move(thumb);}
    }
    if (hand_) {
      // A faulted hand's positions are not measurements; the filter substitutes
      // that side's thumb only.
      for (const auto * hands : {&hand_->left_hands, &hand_->right_hands}) {
        for (const auto & hand : *hands) {
          if (hand.faultcode == 0) {continue;}
          for (size_t axis = 0; axis < all_thumb_names.size(); ++axis) {
            if (cbf_->config().hands_installed[axis / 2] && hand.name == all_thumb_names[axis]) {
              snapshot.thumb_faulted[axis / 2] = true;
            }
          }
        }
      }
    }
    mark(trace_.unlocked);
    mark_cpu(trace_.unlocked_cpu_ms);
    lock.unlock();  // Vendor feedback intake continues on its existing threads.
    CbfDecision decision;
    try {
      decision = cbf_->filter(snapshot, now_s);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "CBF rejected frame: %s", error.what());
      decision.epoch = snapshot.epoch;
      if (!cbf_->config().enforce) {decision.output = snapshot.requested;}
    }
    mark_cpu(trace_.filtered_cpu_ms);
    mark(trace_.filtered);
    lock.lock();
    mark(trace_.relocked);
    double finished = monotonic_s();
    std::string latest_feedback_problem;
    if (cbf_->config().enforce) {
      // Recheck arm data validity after callbacks ran during the solve. An ordinary
      // new sample is allowed; a new NaN, missing joint or stale stamp is not.
      CbfSnapshot latest;
      latest.arm = state(kVendorArmOrder, finished);
      latest_feedback_problem = cbf_->config().feedback_error(latest);
      finished = monotonic_s();
    }
    const double oldest_age = snapshot.arm ? snapshot.arm->age_s + finished - now_s :
      std::numeric_limits<double>::infinity();
    if (core_.safety_epoch() != snapshot.epoch ||
      !core_.mirror_context(cbf_->config().mirror_mode, finished))
    {
      cbf_->discarded();
      trace_.path = TickPath::kDiscarded;
      if (player_) {player_->reject_pending("not_admitted");}
      cbf_->tick_finished((finished - now_s) * 1e6);
      flush_notes();
      return;
    }
    if (cbf_->config().enforce && !cbf_hands_match()) {
      decision = cbf_->feedback_fallback(decision, "hand_configuration_mismatch");
    } else if (cbf_->config().enforce && !core_.mirror_candidate(cbf_->config().mirror_mode, finished)) {
      decision = cbf_->feedback_fallback(decision, "mirror_input_stale");
    } else if (cbf_->config().enforce &&
      (!latest_feedback_problem.empty() || oldest_age > cbf_->config().feedback_max_age_s))
    {
      decision = cbf_->feedback_fallback(decision, latest_feedback_problem.empty() ?
        "snapshot_feedback_expired" : latest_feedback_problem);
    } else if (cbf_->config().enforce &&
      (finished - now_s) * 1e6 >= cbf_->config().deadline_us - cbf_->config().guard_us)
    {
      decision = cbf_->deadline_fallback(decision);
    }
    const bool was_releasing = cbf_->diagnostics().releasing;
    const double previous_stuck_s = cbf_->diagnostics().stuck_s;
    const bool was_collision = cbf_->diagnostics().stuck_collision;
    const double release_at = monotonic_s();
    decision = cbf_->release_if_stuck(decision, snapshot, release_at);
    const auto & d = cbf_->diagnostics();
    // For the INFO line only: the closest verified output since it last printed.
    // A held or released frame has no distance of its own.
    if (d.distance_valid) {
      log_min_distance_m_ = std::min(log_min_distance_m_, d.minimum_distance_m);
      ++log_distance_frames_;
    } else {
      ++log_no_distance_frames_;
    }
    publish(decision.output);
    trace_.path = TickPath::kCbf;
    trace_.decided(decision, cbf_->config().enforce);
    // Start, end and every second of a period published without verification.
    // Observation only: a long release is a CBF criterion to fix, not to act on.
    // Logged after the command is out, so the WARN never delays it.
    if (d.releasing != was_releasing || (d.releasing && release_at - last_release_log_s_ >= 1.0)) {
      const auto warn_begun = timing_ ? timing_->now() : TickTiming::Clock::time_point{};
      RCLCPP_WARN(
        get_logger(), "CBF_RELEASE %s episode=%llu held_s=%.3f released_frames=%llu collision=%d reason=%s",
        d.releasing != was_releasing ? (d.releasing ? "start" : "end") : "ongoing",
        static_cast<unsigned long long>(d.release_episodes),
        d.releasing ? d.stuck_s : previous_stuck_s, static_cast<unsigned long long>(d.released),
        d.releasing ? d.stuck_collision : was_collision, d.reason.c_str());
      if (timing_) {
        trace_.warn_ms += TickTiming::ms(timing_->now() - warn_begun);
        trace_.warned = true;
      }
      last_release_log_s_ = release_at;
    }
    core_.remember_arm_output(decision.output);
    cbf_->committed(decision, monotonic_s());
    if (player_ && player_->start_pending()) {
      // Judged on what was just published; admitted, it plays from the next tick.
      try {
        player_->evaluate_start(decision, core_.safety_epoch(), cbf_->config().enforce, now_s);
      } catch (const std::exception & error) {
        RCLCPP_ERROR(get_logger(), "template start not judged: %s", error.what());
        player_->reject_pending("internal_error");
      }
    }
    const double ended = monotonic_s();
    cbf_->tick_finished((ended - now_s) * 1e6, (ended - finished) * 1e6);
    flush_notes();
  }

  // One tick of a playing template segment, in place of the CBF path. False when
  // the player stopped instead (a runtime-level exit here, or an internal one in
  // step): the tick then goes on down the ordinary path, so the stream never pauses.
  bool play_template(double now_s, bool lapsed, bool mirror_context, bool named_only)
  {
    const char * stop = nullptr;
    if (lapsed) {
      stop = "control_lapsed";
    } else if (!cbf_ || !mirror_context) {
      stop = "revoked";
    } else if (named_only) {
      // This and the epoch check are normally met first by on_control, which
      // stops the player before the tick; kept so the tick never plays past them.
      stop = "named_arm";
    } else if (core_.safety_epoch() != player_->epoch()) {
      stop = "epoch_changed";
    }
    Output output;
    if (!stop) {
      try {
        output = player_->step(core_.tick(now_s, false), now_s);
      } catch (const std::exception & error) {
        RCLCPP_ERROR(get_logger(), "template playback failed: %s", error.what());
        stop = "internal_error";
      }
    }
    if (stop) {
      player_->stop(stop);
    }
    if (output.kind != Output::Kind::kUpperBody) {
      return false;
    }
    publish(output);
    trace_.path = TickPath::kPlayed;
    cbf_->played(output, player_->epoch(), now_s);
    core_.remember_arm_output(output);
    cbf_->tick_finished((monotonic_s() - now_s) * 1e6);
    flush_notes();
    return true;
  }

  void publish_template_status(double now_s)
  {
    const auto messages = player_->take_status(now_s);
    for (const auto & text : messages) {
      std_msgs::msg::String message;
      message.data = text;
      template_status_pub_->publish(message);
    }
    // Every transition and refusal once in the log; the periodic repeats are not.
    const auto & counters = player_->counters();
    if (!messages.empty() &&
      (player_->state() != logged_template_state_ || counters.rejected != logged_template_rejected_))
    {
      RCLCPP_INFO(get_logger(), "TEMPLATE %s", messages.back().c_str());
      logged_template_state_ = player_->state();
      logged_template_rejected_ = counters.rejected;
    }
  }

  // The hands the platform reports are the ones the CBF geometry was built for.
  // The runtime reads this from the status and refuses to start mirroring otherwise.
  // An installed side may report no hand (an empty flange, checked as a phantom
  // hand); a hand on a side configured without one never matches.
  bool cbf_hands_match() const
  {
    const auto & installed = cbf_->config().hands_installed;
    return hand_ && cbf_->config().matches_hand_types(core_.hand_types()) &&
           (installed[0] || hand_->left_hands.empty()) && (installed[1] || hand_->right_hands.empty());
  }

  // Installed sides the platform reports as type 0, checked as phantom hands.
  std::array<bool, 2> phantom_hands() const
  {
    std::array<bool, 2> phantom{};
    const auto & types = core_.hand_types();
    for (size_t side = 0; side < phantom.size() && side < types.size(); ++side) {
      phantom[side] = cbf_->config().hands_installed[side] && types[side] == 0;
    }
    return phantom;
  }

  void publish(const Output & output)
  {
    const bool timed = timing_ && output.kind != Output::Kind::kNone;
    const auto begun = timed ? timing_->now() : TickTiming::Clock::time_point{};
    const auto header = [this, &output](const char * frame_id) {
        aimdk_msgs::msg::MessageHeader header;
        header.stamp = now();  // the platform drops commands stamped outside its window
        header.frame_id = frame_id;
        header.sequence = output.sequence;
        return header;
      };
    if (output.kind == Output::Kind::kUpperBody) {
      aimdk_msgs::msg::UpperBodyCommandArray message;
      message.header = header("mc_upper_body");
      message.source = core_.plan().source;
      message.hand_sub_mode = static_cast<uint8_t>(output.hand_sub_mode);
      std::copy_n(output.head.begin(), std::min(output.head.size(), message.head_pos.size()), message.head_pos.begin());
      std::copy_n(output.arm.begin(), std::min(output.arm.size(), message.arm_pos.size()), message.arm_pos.begin());
      message.hand_pos = output.hand;
      upper_body_pub_->publish(message);
    } else if (output.kind == Output::Kind::kLocomotion) {
      aimdk_msgs::msg::McLocomotionVelocity message;
      message.header = header("mc_locomotion");
      message.source = core_.plan().source;
      message.forward_velocity = output.twist[0];
      message.lateral_velocity = output.twist[1];
      message.angular_velocity = output.twist[2];
      locomotion_pub_->publish(message);
    }
    if (timed) {
      trace_.publish_ms += TickTiming::ms(timing_->now() - begun);
      ++trace_.publishes;
      timing_->published();
    }
  }

  void flush_notes()
  {
    for (const auto & warning : core_.take_warnings()) {
      RCLCPP_WARN(get_logger(), "%s", warning.c_str());
    }
    auto rejected = core_.take_rejections();
    if (!rejected.empty()) {
      pending_rejections_.insert(pending_rejections_.end(), rejected.begin(), rejected.end());
    }
  }

  void publish_status()
  {
    const auto entered = timing_ ? timing_->now() : TickTiming::Clock::time_point{};
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (timing_) {timing_->add(TimingSegment::kStatusLockWait, TickTiming::ms(timing_->now() - entered));}
    flush_notes();
    const double now_s = monotonic_s();
    nlohmann::json status;
    // ``at`` is when this report was generated, on the monotonic clock both
    // processes share: the runtime ages the report end to end (its queue time
    // included), not from the moment its own callback happened to run.
    status["at"] = now_s;
    status["control_seq"] = core_.control_seq();
    // Observation only. Unlike control_seq, which the runtime's heartbeat moves,
    // this moves only when the mode, owner, admission, clear or safe state does,
    // or the data plane clears: equal before and after means nothing was revoked.
    status["safety_epoch"] = core_.safety_epoch();
    status["groups"] = core_.group_ages(now_s);
    status["hand_types"] = core_.hand_types();
    const auto hand = core_.measured_hand(now_s);
    status["hand_measured"] = hand ? nlohmann::json(hand->position) : nlohmann::json(nullptr);
    status["hand_measured_age_s"] = hand ? nlohmann::json(hand->age_s) : nlohmann::json(nullptr);
    const auto arm = core_.measured_with_age(kVendorArmOrder, now_s);
    status["arm_measured"] = arm ? nlohmann::json(arm->position) : nlohmann::json(nullptr);
    status["arm_measured_age_s"] = arm ? nlohmann::json(arm->age_s) : nlohmann::json(nullptr);
    // The owner as applied: a claim the data plane refused stays unowned here,
    // which is how the runtime learns its claim did not take.
    status["arm_owner"] = core_.arm_owner() ? nlohmann::json(*core_.arm_owner()) : nlohmann::json(nullptr);
    nlohmann::json faults = nlohmann::json::object();
    for (const auto & [group, codes] : core_.take_faults()) {
      nlohmann::json listed = nlohmann::json::array();
      for (const auto & [name, code] : codes) {
        listed.push_back({name, code});
      }
      faults[group] = listed;
    }
    status["faults"] = faults;
    status["rejected"] = std::exchange(pending_rejections_, {});
    status["public_arm_count"] = core_.public_arm_count();
    status["intake_lag_ms"] = std::exchange(intake_lag_ms_, {});
    if (cbf_) {
      const auto & d = cbf_->diagnostics();
      status["cbf"] = {
        {"enabled", true}, {"mode", cbf_->config().enforce ? "enforce" : "shadow"},
        {"mirror_mode", cbf_->config().mirror_mode},
        {"mapping_verified", cbf_->config().mapping_verified}, {"mirror_active", d.active},
        {"mapping_available", cbf_->config().mapping_available},
        {"hands_installed", cbf_->config().hands_installed},
        {"hand_configuration_matches", cbf_hands_match()},
        {"phantom_hands", phantom_hands()},
        {"check_arm_feedback_goal_range", cbf_->config().check_arm_feedback_goal_range},
        {"arm_age_s", std::isfinite(d.arm_age_s) && d.arm_age_s >= 0.0 ? nlohmann::json(d.arm_age_s) : nlohmann::json(nullptr)},
        {"thumb_age_s", std::isfinite(d.thumb_age_s) && d.thumb_age_s >= 0.0 ? nlohmann::json(d.thumb_age_s) : nlohmann::json(nullptr)},
        {"calls", d.calls}, {"accepted", d.accepted}, {"published", d.published},
        {"modified", d.modified}, {"held", d.held}, {"withheld", d.withheld},
        {"geometry_modified", d.geometry_modified},
        {"recovery_verified", d.recovery_verified}, {"hand_held", d.hand_held},
        {"deadlines", d.deadlines}, {"reason", d.reason},
        {"node_deadlines", d.node_deadlines}, {"tick_us", d.tick_us}, {"max_tick_us", d.max_tick_us},
        {"max_after_check_us", d.max_after_check_us},
        {"publish_overruns", d.publish_overruns},
        {"available_budget_us", d.available_budget_us},
        {"hand_check_us", d.hand_check_us}, {"verification_us", d.verification_us},
        {"projection_us", d.projection_us},
        {"max_hand_check_us", d.max_hand_check_us}, {"max_verification_us", d.max_verification_us},
        {"max_projection_us", d.max_projection_us},
        {"verification_reuses", d.verification_reuses}, {"hand_budget_holds", d.hand_budget_holds},
        {"elapsed_us", d.elapsed_us}, {"max_elapsed_us", d.max_elapsed_us},
        {"minimum_distance_m", d.distance_valid ? nlohmann::json(d.minimum_distance_m) : nlohmann::json(nullptr)}
      };
      status["cbf"]["invalid_input_frames"] = d.invalid_input_frames;
      status["cbf"]["thumb_feedback_frames"] = d.thumb_feedback_frames;
      status["cbf"]["thumb_feedback_issue"] = d.thumb_feedback_issue;
      status["cbf"]["released"] = d.released;
      if (player_) {status["cbf"]["played"] = d.played;}
      status["cbf"]["release_episodes"] = d.release_episodes;
      status["cbf"]["collision_release_episodes"] = d.collision_release_episodes;
      status["cbf"]["releasing"] = d.releasing;
      status["cbf"]["stuck_collision"] = d.stuck_collision;
      status["cbf"]["stuck_s"] = d.stuck_s;
      status["cbf"]["hold_release_after_s"] = cbf_->config().hold_release_after_s;
      status["cbf"]["collision_release_after_s"] = cbf_->config().collision_release_after_s;
      status["cbf"]["collision_release_without_history_s"] = cbf_->config().collision_release_without_history_s;
      status["cbf"]["collision_release_speed_scale"] = cbf_->config().collision_release_speed_scale;
      if (d.last_input_fault) {
        status["cbf"]["last_input_fault_event"] = d.last_input_fault->event;
        status["cbf"]["last_input_fault_issue"] = d.last_input_fault->issue;
      }
      if (d.thumb_angles_valid) {
        status["cbf"]["thumb_current_urdf"] = d.thumb_current_urdf;
        status["cbf"]["thumb_target_urdf"] = d.thumb_target_urdf;
        for (size_t axis = 0; axis < 4; ++axis) {
          if (!cbf_->config().hands_installed[axis / 2]) {
            status["cbf"]["thumb_current_urdf"][axis] = nullptr;
            status["cbf"]["thumb_target_urdf"][axis] = nullptr;
          }
        }
      }
      // Every 2 s; the distance figures cover the frames since the previous line.
      if (now_s - last_status_log_s_ >= 2.0) {
        char min_distance[32] = "na";
        if (log_distance_frames_ > 0) {
          std::snprintf(min_distance, sizeof(min_distance), "%.4f", log_min_distance_m_);
        }
        const auto missed_ticks = timing_ ? std::to_string(timing_->missed_ticks()) : std::string("na");
        RCLCPP_INFO(
          get_logger(),
          "CBF status: mode=%s mirror_active=%s calls=%llu published=%llu geometry_modified=%llu "
          "held=%llu withheld=%llu released=%llu thumb_feedback_frames=%llu "
          "recovery_verified=%s hand_held=%s "
          "budget_us=%.0f hand_us=%.0f verify_us=%.0f project_us=%.0f reuses=%llu "
          "max_tick_us=%.0f max_filter_us=%.0f max_after_check_us=%.0f "
          "min_distance_m=%s distance_frames=%llu no_distance_frames=%llu "
          "missed_ticks=%s node_deadlines=%llu deadlines=%llu publish_overruns=%llu reason=%s",
          cbf_->config().enforce ? "enforce" : "shadow", d.active ? "true" : "false",
          static_cast<unsigned long long>(d.calls), static_cast<unsigned long long>(d.published),
          static_cast<unsigned long long>(d.geometry_modified), static_cast<unsigned long long>(d.held),
          static_cast<unsigned long long>(d.withheld), static_cast<unsigned long long>(d.released),
          static_cast<unsigned long long>(d.thumb_feedback_frames), d.recovery_verified ? "true" : "false",
          d.hand_held ? "true" : "false", d.available_budget_us, d.hand_check_us,
          d.verification_us, d.projection_us, static_cast<unsigned long long>(d.verification_reuses),
          d.max_tick_us, d.max_elapsed_us, d.max_after_check_us, min_distance,
          static_cast<unsigned long long>(log_distance_frames_),
          static_cast<unsigned long long>(log_no_distance_frames_),
          missed_ticks.c_str(),
          static_cast<unsigned long long>(d.node_deadlines), static_cast<unsigned long long>(d.deadlines),
          static_cast<unsigned long long>(d.publish_overruns), d.reason.c_str());
        last_status_log_s_ = now_s;
        log_min_distance_m_ = std::numeric_limits<double>::infinity();
        log_distance_frames_ = 0;
        log_no_distance_frames_ = 0;
      }
      if (d.last_thumb_feedback_event && d.last_thumb_feedback_event->event != logged_thumb_event_ &&
        (logged_thumb_event_ == 0 || now_s - last_thumb_event_log_s_ >= 2.0))
      {
        // Advisory: the frame went ahead with a clamped or substituted reading.
        static constexpr std::array<const char *, 4> names{
          "L_thumb_roll", "L_thumb_abad", "R_thumb_roll", "R_thumb_abad"};
        const auto & event = *d.last_thumb_feedback_event;
        const auto number = [](double value) {
            return std::isfinite(value) ? nlohmann::json(value) : nlohmann::json(nullptr);
          };
        const nlohmann::json detail = {
          {"event", event.event}, {"at_monotonic_s", number(event.at_s)}, {"issue", event.issue},
          {"joint", event.axis >= 0 && event.axis < 4 ? nlohmann::json(names[event.axis]) : nlohmann::json(nullptr)},
          {"value", number(event.value)}, {"lower", number(event.lower)}, {"upper", number(event.upper)},
          {"frames", d.thumb_feedback_frames}, {"angle_units", "native rad"}
        };
        RCLCPP_WARN(get_logger(), "CBF_THUMB_FEEDBACK %s", detail.dump().c_str());
        logged_thumb_event_ = event.event;
        last_thumb_event_log_s_ = now_s;
      }
      if (d.last_input_fault && d.last_input_fault->event != logged_input_fault_event_ &&
        (logged_input_fault_event_ == 0 || now_s - last_input_fault_log_s_ >= 2.0))
      {
        const auto & fault = *d.last_input_fault;
        const auto number = [](double value) {
            return std::isfinite(value) ? nlohmann::json(value) : nlohmann::json(nullptr);
          };
        const auto numbers = [&](const auto & values) {
            auto array = nlohmann::json::array();
            for (double value : values) {array.push_back(number(value));}
            return array;
          };
        nlohmann::json detail = {
          {"event", fault.event}, {"call", fault.call}, {"epoch", fault.epoch},
          {"command_sequence", fault.command_sequence}, {"stage", fault.stage},
          {"at_monotonic_s", number(fault.at_s)},
          {"issue", fault.issue}, {"index", fault.index}, {"joint", nullptr},
          {"arm_current14", numbers(fault.arm_current)}, {"arm_target14", numbers(fault.arm_target)},
          {"arm_requested14", numbers(fault.arm_requested)},
          {"kernel_thumb_current6", numbers(fault.thumb_current_urdf)},
          {"kernel_thumb_target6", numbers(fault.thumb_target_urdf)},
          {"thumb_native_feedback_compact", numbers(fault.thumb_native_current)},
          {"thumb_native_target_slots_0_1_10_11", numbers(fault.thumb_native_target)},
          {"hands_installed", cbf_->config().hands_installed},
          {"mapping_verified", cbf_->config().mapping_verified},
          {"feedback_limit_tolerance_rad", cbf_->config().feedback_limit_tolerance_rad},
          {"check_arm_feedback_goal_range", cbf_->config().check_arm_feedback_goal_range},
          {"arm_age_s", number(fault.arm_age_s)}, {"thumb_age_s", number(fault.thumb_age_s)},
          {"angle_units", "rad"}
        };
        auto bounds = nlohmann::json::array();
        for (const auto & band : fault.thumb_limits_urdf) {
          bounds.push_back({number(band[0]), number(band[1])});
        }
        detail["kernel_thumb_limits4"] = std::move(bounds);
        auto current_thumb_bounds = nlohmann::json::array();
        for (const auto & band : fault.thumb_current_limits_urdf) {
          current_thumb_bounds.push_back({number(band[0]), number(band[1])});
        }
        detail["kernel_thumb_feedback_limits4"] = std::move(current_thumb_bounds);
        auto arm_bounds = nlohmann::json::array();
        for (const auto & band : fault.arm_feedback_limits) {
          arm_bounds.push_back({number(band[0]), number(band[1])});
        }
        detail["kernel_arm_feedback_limits14"] = std::move(arm_bounds);
        const bool arm_issue = fault.issue.find("_arm_") != std::string::npos;
        const bool thumb_pose_issue = fault.issue.find("current_thumb_") == 0 ||
          fault.issue.find("target_thumb_") == 0;
        if (arm_issue && fault.index >= 0 && static_cast<size_t>(fault.index) < kVendorArmOrder.size()) {
          detail["joint"] = kVendorArmOrder[static_cast<size_t>(fault.index)];
        } else if (thumb_pose_issue && fault.index >= 0 && fault.index < 6) {
          static constexpr std::array<const char *, 6> names{
            "L_thumb_roll", "L_thumb_abad", "ignored_L_thumb_mcp",
            "R_thumb_roll", "R_thumb_abad", "ignored_R_thumb_mcp"};
          detail["joint"] = names[static_cast<size_t>(fault.index)];
        }
        if (fault.index >= 0 && (arm_issue || thumb_pose_issue || fault.issue == "thumb_limits_invalid")) {
          detail["value"] = number(fault.value);
          detail["lower"] = number(fault.lower);
          detail["upper"] = number(fault.upper);
        }
        if (fault.issue == "distance_gradient_invalid") {
          detail["pair_index"] = fault.index;
          if (fault.index >= 0 && static_cast<size_t>(fault.index) < x2::cbf::kPrimitivePairs.size()) {
            const auto pair = x2::cbf::kPrimitivePairs[static_cast<size_t>(fault.index)];
            detail["pair_first"] = x2::cbf::primitive_name(pair.first);
            detail["pair_second"] = x2::cbf::primitive_name(pair.second);
          }
        }
        RCLCPP_WARN(get_logger(), "CBF_INPUT_FAULT %s", detail.dump().c_str());
        logged_input_fault_event_ = fault.event;
        last_input_fault_log_s_ = now_s;
      }
    }
    const auto age = core_.public_arm_age(now_s);
    status["public_arm_age_s"] = age ? nlohmann::json(*age) : nlohmann::json(nullptr);
    if (timing_) {
      const auto begun = timing_->now();
      status["tick_timing"] = timing_->report(now_s);
      timing_->add(TimingSegment::kTimingJson, TickTiming::ms(timing_->now() - begun));
    }
    std_msgs::msg::String message;
    message.data = status.dump();
    status_pub_->publish(message);
    if (timing_) {timing_->add(TimingSegment::kStatus, TickTiming::ms(timing_->now() - entered));}
  }

  // The vendor input threads and the main thread share the core and the latest
  // hand and echo messages; every touch of them holds this (recursive: control
  // publishes the safe command and the status itself).
  std::recursive_mutex mutex_;
  Core core_;
  std::optional<CbfSafety> cbf_;
  std::optional<TemplatePlayer> player_;
  std::optional<TickTiming> timing_;
  TickTrace trace_;  // the command tick in progress, when timing is on
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr template_status_pub_;
  TemplateState logged_template_state_ = TemplateState::kIdle;
  uint64_t logged_template_rejected_ = 0;
  uint64_t logged_input_fault_event_ = 0;
  double last_input_fault_log_s_ = 0.0;
  uint64_t logged_thumb_event_ = 0;
  double last_thumb_event_log_s_ = 0.0;
  double last_release_log_s_ = 0.0;
  double last_status_log_s_ = -std::numeric_limits<double>::infinity();
  double log_min_distance_m_ = std::numeric_limits<double>::infinity();
  uint64_t log_distance_frames_ = 0;
  uint64_t log_no_distance_frames_ = 0;
  std::vector<rclcpp::CallbackGroup::SharedPtr> intake_groups_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
  std::vector<std::pair<rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr, std::vector<std::string>>> views_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr touch_pub_;
  rclcpp::Publisher<ibrobot_msgs::msg::JointCommandState>::SharedPtr echo_pub_;
  rclcpp::Publisher<aimdk_msgs::msg::UpperBodyCommandArray>::SharedPtr upper_body_pub_;
  rclcpp::Publisher<aimdk_msgs::msg::McLocomotionVelocity>::SharedPtr locomotion_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr joint_timer_;
  rclcpp::TimerBase::SharedPtr command_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr touch_timer_;
  rclcpp::TimerBase::SharedPtr echo_timer_;
  HandStateArray::ConstSharedPtr hand_;
  std::array<bool, 2> phantom_logged_{};
  std::array<std::optional<double>, 2> present_since_{};
  bool hand_fresh_ = false;
  std::unique_ptr<ibrobot_msgs::msg::JointCommandState> echo_;
  std::vector<std::string> pending_rejections_;
  std::map<std::string, double> intake_lag_ms_;
};

}  // namespace aimdk_dataplane

int main(int argc, char ** argv)
{
  // Our own signal handling: the safe command must go out while the context is still up.
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, aimdk_dataplane::on_signal);
  std::signal(SIGTERM, aimdk_dataplane::on_signal);
  int code = 0;
  try {
    auto node = std::make_shared<aimdk_dataplane::Dataplane>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::vector<std::unique_ptr<rclcpp::executors::SingleThreadedExecutor>> intake;
    std::vector<std::thread> intake_threads;
    for (const auto & group : node->intake_groups()) {
      intake.push_back(std::make_unique<rclcpp::executors::SingleThreadedExecutor>());
      intake.back()->add_callback_group(group, node->get_node_base_interface());
      intake_threads.emplace_back([spinner = intake.back().get()]() {spinner->spin();});
    }
    // spin(), not a spin_once() loop: spin_once runs one callback per wait, and at
    // ~2800 vendor samples a second plus the timers that rebuild-the-wait-set-per-
    // callback rate fell behind on the loaded 310P, so the reader queues filled to
    // their depth and samples were taken up to depth x period late (166 ms for the
    // 500 Hz groups, 342 ms for the hands, measured 2026-10-28). The signal handler
    // only sets a flag; this thread turns it into a cancel.
    std::thread watcher([&executor, &intake]() {
        while (!aimdk_dataplane::g_stop && rclcpp::ok()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        executor.cancel();
        for (auto & spinner : intake) {
          spinner->cancel();
        }
      });
    executor.spin();
    aimdk_dataplane::g_stop = true;
    watcher.join();
    for (auto & thread : intake_threads) {
      thread.join();
    }
    node->publish_safe_command();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  } catch (const std::exception & error) {
    std::fprintf(stderr, "aimdk_dataplane failed: %s\n", error.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
