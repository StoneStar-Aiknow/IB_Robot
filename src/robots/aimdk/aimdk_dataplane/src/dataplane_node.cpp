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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "aimdk_dataplane/core.hpp"
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

}  // namespace

class Dataplane : public rclcpp::Node
{
public:
  Dataplane()
  : Node("aimdk_dataplane"),
    core_(plan_from_json(declare_parameter<std::string>("plan_json", "")))
  {
    const Plan & plan = core_.plan();
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
    if (!plan.upper_body_command.topic.empty()) {
      upper_body_pub_ = create_publisher<aimdk_msgs::msg::UpperBodyCommandArray>(
        plan.upper_body_command.topic, vendor_qos(plan.upper_body_command, plan.upper_body_command.depth));
    }
    locomotion_pub_ = create_publisher<aimdk_msgs::msg::McLocomotionVelocity>(
      plan.locomotion_command.topic, vendor_qos(plan.locomotion_command, plan.locomotion_command.depth));
    for (const auto & channel : plan.channels) {
      if (channel.twist) {
        subscriptions_.push_back(
          create_subscription<geometry_msgs::msg::Twist>(
            channel.topic, command_qos(), [this, &channel](geometry_msgs::msg::Twist::ConstSharedPtr message) {
              std::lock_guard<std::recursive_mutex> lock(mutex_);
              core_.on_twist(channel, message->linear.x, message->linear.y, message->angular.z, monotonic_s());
              flush_notes();
            }));
      } else {
        subscriptions_.push_back(
          create_subscription<std_msgs::msg::Float64MultiArray>(
            channel.topic, command_qos(),
            [this, &channel](std_msgs::msg::Float64MultiArray::ConstSharedPtr message) {
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
    note_intake("hand", stamp);
    core_.on_hand_state(
      {message->left_hand_type.value, message->right_hand_type.value}, std::move(entries), std::move(measured),
      codes, stamp, monotonic_s());
    hand_ = message;
    hand_fresh_ = true;
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    Control control;
    try {
      control = control_from_json(text);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "ignoring a malformed control word: %s", error.what());
      return;
    }
    if (core_.on_control(control, monotonic_s())) {
      publish_safe_command();
    }
    publish_status();  // the acknowledgement the runtime waits for
  }

  void publish_joint_states()
  {
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (echo_) {
      echo_pub_->publish(std::move(echo_));
    }
  }

  void publish_commands()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const double now_s = monotonic_s();
    if (core_.control_lapsed(now_s)) {
      publish_safe_command();
    }
    publish(core_.tick(now_s));
    flush_notes();
  }

  void publish(const Output & output)
  {
    const auto header = [this, &output](const char * frame_id) {
        aimdk_msgs::msg::MessageHeader header;
        header.stamp = now();  // the platform drops commands stamped outside its window
        header.frame_id = frame_id;
        header.sequence = output.sequence;
        return header;
      };
    if (output.kind == Output::Kind::kUpperBody && upper_body_pub_) {
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    flush_notes();
    const double now_s = monotonic_s();
    nlohmann::json status;
    // ``at`` is when this report was generated, on the monotonic clock both
    // processes share: the runtime ages the report end to end (its queue time
    // included), not from the moment its own callback happened to run.
    status["at"] = now_s;
    status["control_seq"] = core_.control_seq();
    status["groups"] = core_.group_ages(now_s);
    status["hand_types"] = core_.hand_types();
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
    const auto age = core_.public_arm_age(now_s);
    status["public_arm_age_s"] = age ? nlohmann::json(*age) : nlohmann::json(nullptr);
    std_msgs::msg::String message;
    message.data = status.dump();
    status_pub_->publish(message);
  }

  // The vendor input threads and the main thread share the core and the latest
  // hand and echo messages; every touch of them holds this (recursive: control
  // publishes the safe command and the status itself).
  std::recursive_mutex mutex_;
  Core core_;
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
