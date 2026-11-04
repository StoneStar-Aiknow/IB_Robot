// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The data plane's command and state rules over time, without ROS: what each tick
// sends, what it holds, what it refuses and what it reports back.

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "aimdk_dataplane/core.hpp"

using aimdk_dataplane::Channel;
using aimdk_dataplane::Control;
using aimdk_dataplane::Core;
using aimdk_dataplane::JointEntries;
using aimdk_dataplane::kVendorArmOrder;
using aimdk_dataplane::kVendorHeadOrder;
using aimdk_dataplane::Output;
using aimdk_dataplane::Plan;

namespace
{

Channel channel(const std::string & name, std::vector<std::string> joints, std::string hand_type = "")
{
  Channel out;
  out.name = name;
  out.topic = "/" + name;
  out.joints = std::move(joints);
  out.modes = {"stream"};
  out.hand_type = std::move(hand_type);
  return out;
}

Plan plan(bool arm_board = false)
{
  Plan out;
  out.source = "test";
  out.control_timeout_s = 100.0;  // the lapse has a test of its own
  out.channels = {
    channel("arm_stream", kVendorArmOrder), channel("head_stream", kVendorHeadOrder),
    channel("gripper_stream", {}, "claw"), channel("hand_stream", {}, "dexterous_joint"),
  };
  Channel twist = channel("cmd_vel", {});
  twist.twist = true;
  twist.modes = {"walk"};
  out.channels.push_back(twist);
  out.mc_actions = {{"stream", "UPPERBODY_REMOTE_SPLIT"}, {"walk", "LOCOMOTION_DEFAULT"}};
  if (arm_board) {
    out.arm_command = aimdk_dataplane::Endpoint{"/aima/hal/joint/arm/command"};
    out.arm_command_state = aimdk_dataplane::View{"/aimdk/arm/command_state", kVendorArmOrder};
  }
  return out;
}

Control control(const std::string & mode = "stream", bool admitted = true)
{
  Control out;
  out.seq = 1;
  out.mode = mode;
  out.allows_stream = true;
  out.admitted = admitted;
  return out;
}

JointEntries entries(const std::vector<std::string> & names, double base)
{
  JointEntries out;
  for (size_t index = 0; index < names.size(); ++index) {
    out.emplace_back(names[index], base + static_cast<double>(index), 0.0, 0.0);
  }
  return out;
}

void measure(Core & core, double now, double arm = 0.0, double head = 100.0)
{
  core.on_joint_state("arm", entries(kVendorArmOrder, arm), {}, 1000, now);
  core.on_joint_state("head", entries(kVendorHeadOrder, head), {}, 2000, now);
}

std::vector<double> values(size_t count, double value)
{
  return std::vector<double>(count, value);
}

const Channel & find(const Core & core, const std::string & name)
{
  for (const auto & candidate : core.plan().channels) {
    if (candidate.name == name) {
      return candidate;
    }
  }
  throw std::out_of_range(name);
}

}  // namespace

TEST(Core, SendsNothingUntilTheRuntimeAdmitsAStreamMode)
{
  Core core(plan());
  measure(core, 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  EXPECT_EQ(core.tick(0.01).kind, Output::Kind::kNone);
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"arm_stream"}));
  core.on_control(control("stream", false), 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  EXPECT_EQ(core.tick(0.01).kind, Output::Kind::kNone);
}

TEST(Core, AnArmTargetIsSentWithTheHeadHeldWhereItIsMeasured)
{
  Core core(plan());
  core.on_control(control(), 0.0);
  measure(core, 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  const auto out = core.tick(0.02);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.arm, values(14, 0.5));
  EXPECT_EQ(out.head, (std::vector<double>{100.0, 101.0}));
  EXPECT_EQ(out.hand_sub_mode, aimdk_dataplane::kHandSubModeClaw);
  EXPECT_TRUE(out.hand.empty());
}

TEST(Core, WithoutAFreshMeasurementNothingIsSentAndTheReasonIsNotedOnce)
{
  Core core(plan());
  core.on_control(control(), 0.0);
  core.on_stream(find(core, "head_stream"), {0.1, 0.1}, 0.0);
  EXPECT_EQ(core.tick(0.02).kind, Output::Kind::kNone);
  EXPECT_EQ(core.tick(0.04).kind, Output::Kind::kNone);
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"upper_body:HOLD_TARGET_UNAVAILABLE"}));
  measure(core, 0.0);
  core.on_stream(find(core, "head_stream"), {0.1, 0.1}, 1.0);
  measure(core, 1.0);
  EXPECT_EQ(core.tick(1.02).kind, Output::Kind::kUpperBody);
}

TEST(Core, EachChannelExpiresOnItsOwnClock)
{
  Core core(plan());
  core.on_control(control(), 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  core.on_stream(find(core, "head_stream"), {0.2, 0.1}, 0.4);
  measure(core, 0.6);
  // The arm target is 0.6 s old: the arm falls back to its measurement, the head target stays.
  const auto out = core.tick(0.6);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.arm[0], 0.0);
  EXPECT_EQ(out.head, (std::vector<double>{0.2, 0.1}));
  // Both quiet: nothing to send.
  EXPECT_EQ(core.tick(1.0).kind, Output::Kind::kNone);
}

TEST(Core, AQuietArmHoldsTheControllersTargetNotTheMeasuredArm)
{
  Core core(plan(true));
  core.on_control(control(), 0.0);
  measure(core, 0.0, -1.0);
  core.on_stream(find(core, "head_stream"), {0.1, 0.1}, 0.0);
  // No board yet and nothing sent in this mode: nothing to hold the arm at.
  EXPECT_EQ(core.tick(0.02).kind, Output::Kind::kNone);
  std::map<std::string, double> board;
  for (const auto & name : kVendorArmOrder) {
    board[name] = 0.7;
  }
  core.on_arm_board(board, 0.03);
  auto out = core.tick(0.04);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.arm, values(14, 0.7));
  // The board goes quiet: what was last sent in this mode is the target the platform holds.
  core.on_stream(find(core, "head_stream"), {0.1, 0.1}, 1.0);
  measure(core, 1.0, -1.0);
  out = core.tick(1.0);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.arm, values(14, 0.7));
}

TEST(Core, AQuietHandChannelRepeatsItsLastCommand)
{
  Core core(plan());
  core.on_control(control(), 0.0);
  measure(core, 0.0);
  core.on_stream(find(core, "gripper_stream"), {0.2, 0.8}, 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.6);
  measure(core, 0.6);
  const auto out = core.tick(0.6);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.hand, (std::vector<double>{0.2, 0.8}));
}

TEST(Core, DexterousHandsAreHeldAtTheirMeasuredPoseInTheHandsAction)
{
  Plan dexterous = plan();
  dexterous.default_hand_sub_mode = aimdk_dataplane::kHandSubModeDexterousJoint;
  Core core(dexterous);
  core.on_control(control(), 0.0);
  measure(core, 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  // A left hand installed and not yet measured: the whole command is withheld.
  core.on_hand_state({0x01, 0x00}, {}, {}, {}, 0, 0.0);
  EXPECT_EQ(core.tick(0.02).kind, Output::Kind::kNone);
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"upper_body:HAND_HOLD_UNAVAILABLE"}));
  std::map<std::string, double> left;
  JointEntries hand;
  for (const auto & joint : aimdk_dataplane::kDexterousJointOrder) {
    left["L_" + joint + "_joint"] = 0.3;
    hand.emplace_back("L_" + joint + "_joint", 0.3, 0.0, 0.0);
  }
  core.on_hand_state({0x01, 0x00}, hand, {{left, {}}}, {}, 0, 0.03);
  const auto out = core.tick(0.04);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.hand_sub_mode, aimdk_dataplane::kHandSubModeDexterousJoint);
  std::vector<double> expected(20, 0.0);
  std::fill(expected.begin(), expected.begin() + 10, 0.3);
  EXPECT_EQ(out.hand, expected);
}

TEST(Core, ClearDropsEveryCommandAndSafeAsksForAZero)
{
  Core core(plan());
  auto word = control();
  core.on_control(word, 0.0);
  measure(core, 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  word.seq = 2;
  word.clear = 1;
  EXPECT_FALSE(core.on_control(word, 0.01));
  EXPECT_EQ(core.tick(0.02).kind, Output::Kind::kNone);
  word.seq = 3;
  word.safe = 1;
  EXPECT_TRUE(core.on_control(word, 0.03));
  EXPECT_EQ(core.control_seq(), 3u);
  const auto zero = core.safe_command();
  EXPECT_EQ(zero.kind, Output::Kind::kLocomotion);
  EXPECT_EQ(zero.twist, (std::array<double, 3>{0.0, 0.0, 0.0}));
}

TEST(Core, ClosingAnOpenAdmissionDropsTheCacheAndAsksForAZero)
{
  Core core(plan());
  auto word = control("walk");
  core.on_control(word, 0.0);
  core.on_twist(find(core, "cmd_vel"), 0.5, 0.0, 0.0, 0.0);
  auto out = core.tick(0.02);
  ASSERT_EQ(out.kind, Output::Kind::kLocomotion);
  EXPECT_EQ(out.twist[0], 0.5);
  // A fault that degrades the runtime closes admission; its heartbeat keeps the
  // control fresh, so the lapse never fires. The drop itself owes the zero.
  word.seq = 2;
  word.admitted = false;
  EXPECT_TRUE(core.on_control(word, 0.1));
  const auto zero = core.safe_command();
  EXPECT_EQ(zero.kind, Output::Kind::kLocomotion);
  EXPECT_EQ(zero.twist, (std::array<double, 3>{0.0, 0.0, 0.0}));
  // Recovery admits again, but the dropped velocity is not parked: nothing
  // streams until a fresh command arrives.
  word.seq = 3;
  word.admitted = true;
  EXPECT_FALSE(core.on_control(word, 0.2));
  EXPECT_EQ(core.tick(0.22).kind, Output::Kind::kNone);
  core.on_twist(find(core, "cmd_vel"), 0.3, 0.0, 0.0, 0.3);
  EXPECT_EQ(core.tick(0.24).twist[0], 0.3);
}

TEST(Core, AQuietControlPlaneClosesTheOutputOnce)
{
  Plan lapsing = plan();
  lapsing.control_timeout_s = 0.5;
  Core core(lapsing);
  core.on_control(control(), 0.0);
  measure(core, 0.4);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.4);
  EXPECT_EQ(core.tick(0.45).kind, Output::Kind::kUpperBody);
  EXPECT_FALSE(core.control_lapsed(0.5));
  EXPECT_TRUE(core.control_lapsed(0.6));
  EXPECT_FALSE(core.control_lapsed(0.7));
  EXPECT_EQ(core.tick(0.6).kind, Output::Kind::kNone);
  // A word from the runtime reopens it, and the dropped target stays dropped.
  core.on_control(control(), 0.7);
  measure(core, 0.7);
  EXPECT_EQ(core.tick(0.72).kind, Output::Kind::kNone);
}

TEST(Core, VelocityGoesToZeroOnceWhenItsPublisherGoesQuiet)
{
  Core core(plan());
  core.on_control(control("walk"), 0.0);
  core.on_twist(find(core, "cmd_vel"), 0.5, 0.0, 0.0, 0.0);
  auto out = core.tick(0.02);
  ASSERT_EQ(out.kind, Output::Kind::kLocomotion);
  EXPECT_EQ(out.twist[0], 0.5);
  out = core.tick(0.6);
  ASSERT_EQ(out.kind, Output::Kind::kLocomotion);
  EXPECT_EQ(out.twist[0], 0.0);
  EXPECT_EQ(core.tick(0.62).kind, Output::Kind::kNone);
  core.on_twist(find(core, "cmd_vel"), 0.1, 0.0, 0.0, 1.0);
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"cmd_vel:VELOCITY_BELOW_MIN"}));
}

TEST(Core, TheNavigationGateHoldsVelocityAtZero)
{
  Plan gated = plan();
  gated.navigation_gate = true;
  Core core(gated);
  core.on_control(control("walk"), 0.0);
  core.on_twist(find(core, "cmd_vel"), 0.5, 0.0, 0.0, 0.0);
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"cmd_vel"}));
  auto word = control("walk");
  word.navigation_enabled = true;
  core.on_control(word, 0.0);
  core.on_twist(find(core, "cmd_vel"), 0.5, 0.0, 0.0, 0.0);
  EXPECT_EQ(core.tick(0.02).twist[0], 0.5);
}

TEST(Core, ANamedMotionOwnsTheArmChannelWhileItRuns)
{
  Core core(plan());
  auto word = control();
  word.arm_owner = 7;
  core.on_control(word, 0.0);
  measure(core, 0.0);
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.0);
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"arm_stream:NAMED_MOTION_ACTIVE"}));
  EXPECT_EQ(core.public_arm_count(), 0u);
  core.on_named_arm(8, values(14, 0.2), 0.0);  // not the owner
  EXPECT_EQ(core.tick(0.01).kind, Output::Kind::kNone);
  core.on_named_arm(7, values(14, 0.2), 0.0);
  const auto out = core.tick(0.02);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.arm, values(14, 0.2));
}

TEST(Core, AClaimIsRefusedWhileThePublicArmStreamIsLive)
{
  Core core(plan());
  core.on_control(control(), 0.0);
  // A public target the runtime's last status report has not shown yet.
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 0.01);
  auto word = control();
  word.seq = 2;
  word.arm_owner = 5;
  EXPECT_FALSE(core.on_control(word, 0.02));
  EXPECT_FALSE(core.arm_owner().has_value());
  EXPECT_EQ(core.take_rejections(), (std::vector<std::string>{"arm_claim:PUBLIC_ARM_STREAM_ACTIVE"}));
  // The public stream keeps the channel: its targets are still applied.
  core.on_stream(find(core, "arm_stream"), values(14, 0.6), 0.03);
  measure(core, 0.03);
  const auto out = core.tick(0.04);
  ASSERT_EQ(out.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(out.arm, values(14, 0.6));
  // The stream went quiet past its staleness: the claim now takes the arm.
  core.on_control(word, 0.6);
  EXPECT_TRUE(core.arm_owner().has_value());
  EXPECT_EQ(*core.arm_owner(), 5);
  measure(core, 0.6);
  core.on_named_arm(5, values(14, 0.2), 0.61);
  EXPECT_EQ(core.tick(0.62).arm, values(14, 0.2));
  // A motion replacing another one takes the channel over outright.
  word.seq = 3;
  word.arm_owner = 6;
  core.on_control(word, 0.63);
  EXPECT_EQ(*core.arm_owner(), 6);
  // Releasing it hands the arm back to the public stream.
  word.seq = 4;
  word.arm_owner.reset();
  core.on_control(word, 0.64);
  EXPECT_FALSE(core.arm_owner().has_value());
  measure(core, 0.65);
  core.on_stream(find(core, "arm_stream"), values(14, 0.7), 0.65);
  const auto again = core.tick(0.66);
  ASSERT_EQ(again.kind, Output::Kind::kUpperBody);
  EXPECT_EQ(again.arm, values(14, 0.7));
}

TEST(Core, APublicArmTargetIsCountedAndAged)
{
  Core core(plan());
  core.on_control(control(), 0.0);
  EXPECT_FALSE(core.public_arm_age(0.0).has_value());
  core.on_stream(find(core, "arm_stream"), values(14, 0.5), 1.0);
  EXPECT_EQ(core.public_arm_count(), 1u);
  EXPECT_DOUBLE_EQ(*core.public_arm_age(1.25), 0.25);
}

TEST(Core, AMeasurementCarriesTheAgeOfItsStalestContributingGroup)
{
  Core core(plan());
  core.on_joint_state("arm", entries(kVendorArmOrder, 0.0), {}, 1000, 0.0);
  core.on_joint_state("leg", entries({"left_knee_joint"}, 5.0), {}, 500, 0.1);
  const auto measured = core.measured_with_age(kVendorArmOrder, 0.2);
  ASSERT_TRUE(measured.has_value());
  std::vector<double> expected;
  for (size_t index = 0; index < kVendorArmOrder.size(); ++index) {
    expected.push_back(static_cast<double>(index));  // entries() steps by one
  }
  EXPECT_EQ(measured->position, expected);
  // The arm group only: a fresh group that carries none of the order does not age it.
  EXPECT_DOUBLE_EQ(measured->age_s, 0.2);
  // A contributing group that went stale means no full measurement, and a fresh
  // unrelated group cannot rescue it.
  core.on_joint_state("leg", entries({"left_knee_joint"}, 5.0), {}, 500, 1.2);
  EXPECT_FALSE(core.measured_with_age(kVendorArmOrder, 1.3).has_value());
}

TEST(Core, AFaultCodeInOneSampleBetweenTakesIsKept)
{
  Core core(plan());
  core.on_joint_state("arm", entries({"a", "b"}, 0.0), {{"a", 0}, {"b", 32}}, 0, 0.0);
  core.on_joint_state("arm", entries({"a", "b"}, 0.0), {{"a", 0}, {"b", 0}}, 0, 0.002);
  auto faults = core.take_faults();
  ASSERT_EQ(faults.count("arm"), 1u);
  EXPECT_EQ(faults["arm"], (std::vector<std::pair<std::string, int>>{{"b", 32}}));
  core.on_joint_state("arm", entries({"a", "b"}, 0.0), {{"a", 0}, {"b", 0}}, 0, 0.004);
  faults = core.take_faults();
  ASSERT_EQ(faults.count("arm"), 1u);
  EXPECT_TRUE(faults["arm"].empty());  // reported, and healthy
  EXPECT_TRUE(core.take_faults().empty());  // nothing reported since: no verdict
}

TEST(Core, AViewIsStampedWithItsStalestFreshSource)
{
  Core core(plan());
  core.on_joint_state("arm", entries({"a"}, 1.0), {}, 5000, 0.0);
  core.on_joint_state("head", entries({"h"}, 2.0), {}, 3000, 0.0);
  core.on_joint_state("leg", entries({"l"}, 3.0), {}, 1000, 0.0);
  auto view = core.view({"a", "h"}, 0.1);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->joints.names, (std::vector<std::string>{"a", "h"}));
  EXPECT_EQ(view->stamp_ns, 3000);  // the leg is not in this view and does not hold its stamp back
  core.on_joint_state("arm", entries({"a"}, 1.0), {}, 6000, 0.6);
  view = core.view({"a", "h"}, 0.7);  // the head went quiet: it drops out instead of being re-stamped
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->joints.names, (std::vector<std::string>{"a"}));
  EXPECT_EQ(view->stamp_ns, 6000);
  EXPECT_FALSE(core.view({"h"}, 0.7).has_value());
}

TEST(Plan, ParsesWhatTheRuntimeResolves)
{
  const auto parsed = aimdk_dataplane::plan_from_json(R"({
    "joint_state_rate_hz": 50, "command_rate_hz": 50, "status_rate_hz": 20,
    "feedback_window_s": 0.5, "stream_staleness_s": 0.5, "cmd_vel_staleness_s": 0.5, "control_timeout_s": 0.5,
    "source": "ibrobot",
    "joint_state": [{"group": "arm", "topic": "/arm", "reliability": "best_effort",
                     "durability": "transient_local", "depth": 10}],
    "hand_state": null, "arm_command": {"topic": "/cmd"},
    "upper_body_command": {"topic": "/mc/upper_body_command"},
    "locomotion_command": {"topic": "/loco"},
    "views": [{"topic": "/joint_states", "order": ["a"]}],
    "touch": null, "arm_command_state": {"topic": "/echo", "order": ["a"]},
    "channels": [{"name": "arm_stream", "topic": "/aimdk/arm/commands", "twist": false,
                  "joints": ["a"], "modes": ["stream"], "hand_type": ""}],
    "head_limits": {"head_yaw_joint": 0.38},
    "hand": {"command_min": 0.0, "command_max": 1.0, "gesture_ids": null, "default_sub_mode": 2},
    "locomotion": {"deadband": 0.005, "min_linear_mps": 0.2, "min_angular_radps": 0.1,
                   "max_linear_mps": 1.0, "max_angular_radps": 1.0},
    "mc_actions": {"stream": "UPPERBODY_REMOTE_SPLIT"}, "navigation_gate": false,
    "control_topic": "/c", "status_topic": "/s", "named_arm_topic": "/n"
  })");
  EXPECT_EQ(parsed.joint_state.at(0).second.durability, "transient_local");
  EXPECT_FALSE(parsed.hand_state.has_value());
  EXPECT_TRUE(parsed.arm_command.has_value());
  EXPECT_EQ(parsed.channels.at(0).modes.count("stream"), 1u);
  EXPECT_EQ(parsed.default_hand_sub_mode, 2);
  EXPECT_EQ(parsed.head_limits.at("head_yaw_joint"), 0.38);
  const auto word = aimdk_dataplane::control_from_json(
    R"({"seq": 4, "clear": 1, "safe": 0, "mode": "stream", "allows_stream": true, "admitted": true,
        "navigation_enabled": false, "arm_owner": 9, "envelope": [0.4, 0.3]})");
  EXPECT_EQ(word.arm_owner, 9);
  EXPECT_EQ(word.envelope->first, 0.4);
}
