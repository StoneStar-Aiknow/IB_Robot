// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The template playback port, without ROS: whitelist, messages, the start gate,
// playing and every exit. The identity thumb mapping is a test fixture only.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "aimdk_dataplane/template_play.hpp"
#include "cbf/generated_constants.hpp"
#include "cbf/rigid_thumb.hpp"
#include "nlohmann/json.hpp"

namespace
{

using aimdk_dataplane::CbfDecision;
using aimdk_dataplane::CbfSafetyConfig;
using aimdk_dataplane::Output;
using aimdk_dataplane::TemplateCommand;
using aimdk_dataplane::TemplatePlayer;
using aimdk_dataplane::TemplateState;
using aimdk_dataplane::TemplateWhitelist;
using aimdk_dataplane::parse_template_command;
using nlohmann::json;

constexpr double kTick = 0.02;
constexpr std::array<std::size_t, 4> kThumbSlots{0, 1, 10, 11};
constexpr std::array<std::size_t, 4> kMovedJoints{0, 3, 7, 10};
constexpr double kMove = 0.25;
const std::string kSha(64, 'a');

CbfSafetyConfig limits(const std::string & mode = "enforce")
{
  const json value = {
    {"enabled", true}, {"mode", mode}, {"protected_mirror_mode", "policy_stream"},
    {"thumb_mapping", {
      {"verified", true},
      {"command_scale", {1.0, 1.0, 1.0, 1.0}}, {"command_offset", {0.0, 0.0, 0.0, 0.0}},
      {"feedback_scale", {1.0, 1.0, 1.0, 1.0}}, {"feedback_offset", {0.0, 0.0, 0.0, 0.0}},
    }},
  };
  return CbfSafetyConfig::from_json(value.dump(), 50.0);
}

std::vector<double> start_arm()
{
  const auto & standing = x2::cbf::generated::kNaturalStanding;
  return {standing.begin(), standing.end()};
}

std::vector<double> target_arm()
{
  auto arm = start_arm();
  for (const auto j : kMovedJoints) {arm[j] += kMove;}
  return arm;
}

std::vector<double> start_hand()
{
  std::vector<double> hand(20, 0.0);
  for (std::size_t axis = 0; axis < kThumbSlots.size(); ++axis) {
    hand[kThumbSlots[axis]] = x2::cbf::kRigidThumbBaselineUrdf[x2::cbf::kRigidThumbIndices[axis]];
  }
  return hand;
}

std::vector<double> target_hand()
{
  auto hand = start_hand();
  for (std::size_t i = 0; i < 20; ++i) {
    if (std::find(kThumbSlots.begin(), kThumbSlots.end(), i) == kThumbSlots.end()) {hand[i] = 0.5;}
  }
  return hand;
}

json segment_json()
{
  return {
    {"P", start_arm()}, {"T", target_arm()}, {"hand_P", start_hand()}, {"hand_T", target_hand()},
    {"t_in", 1.5}, {"t_out", 1.5}, {"t_out_stop", 0.6}, {"hold_max_s", 4.0},
    {"start_tol_rad", 0.03}, {"hand_start_tol", 0.05}, {"report_sha256", kSha},
  };
}

json whitelist_json(const json & segment = segment_json())
{
  return {{"schema", "x2_template_segments/v1"}, {"segments", {{"holy", segment}}}};
}

json message(const std::string & op, uint64_t seg_id)
{
  return {{"schema", "x2_template_play/v1"}, {"op", op}, {"seg_id", seg_id}};
}

CbfDecision published(std::vector<double> arm = start_arm(), std::vector<double> hand = start_hand())
{
  CbfDecision decision;
  decision.epoch = 1;
  decision.accept_new = true;
  decision.output.kind = Output::Kind::kUpperBody;
  decision.output.sequence = 3;
  decision.output.hand_sub_mode = aimdk_dataplane::kHandSubModeDexterousJoint;
  decision.output.head = {0.1, -0.1};
  decision.output.arm = std::move(arm);
  decision.output.hand = std::move(hand);
  return decision;
}

// One mirror session against one player: messages, ticks, statuses.
struct Rig
{
  explicit Rig(const json & file = whitelist_json(), const CbfSafetyConfig & config = limits())
  : config(config), player(config, TemplateWhitelist::from_json(file.dump(), config), true) {}

  void send(const json & value) {player.on_message(value.dump(), now);}
  void start(uint64_t id, const std::string & segment = "holy")
  {
    auto value = message("start", id);
    value["seg"] = segment;
    send(value);
  }
  void keepalive(uint64_t id) {send(message("keepalive", id));}
  void release(uint64_t id, bool fast)
  {
    auto value = message("release", id);
    value["fast"] = fast;
    send(value);
  }
  bool admit(uint64_t id, const CbfDecision & decision = published())
  {
    start(id);
    return player.evaluate_start(decision, 1, config.enforce, now);
  }
  Output requested() const
  {
    Output out;
    out.kind = Output::Kind::kUpperBody;
    out.sequence = 40;
    out.head = {0.2, 0.3};
    return out;
  }
  // One command tick: the time advances, a keepalive is sent every 50 ms when
  // asked, then the player steps. Every published arm step is recorded.
  Output tick(std::optional<uint64_t> keep = std::nullopt)
  {
    now += kTick;
    if (keep && now - last_keepalive >= 0.05 - 1e-9) {
      keepalive(*keep);
      last_keepalive = now;
    }
    const auto out = player.step(requested(), now);
    if (out.kind == Output::Kind::kUpperBody) {
      if (!outputs.empty()) {
        for (std::size_t j = 0; j < 14; ++j) {
          max_step = std::max(max_step, std::abs(out.arm[j] - outputs.back().arm[j]));
        }
      }
      outputs.push_back(out);
    }
    return out;
  }
  // Ticks until the player leaves ``state`` (or a limit); returns the ticks taken.
  int run_while(TemplateState state, std::optional<uint64_t> keep, int limit = 1000)
  {
    int ticks = 0;
    while (player.state() == state && ticks < limit) {
      tick(keep);
      ++ticks;
    }
    return ticks;
  }
  json last_status()
  {
    const auto messages = player.take_status(now);
    if (messages.empty()) {return json();}
    return json::parse(messages.back());
  }

  CbfSafetyConfig config;
  TemplatePlayer player;
  double now = 100.0;
  double last_keepalive = 100.0;
  std::vector<Output> outputs;
  double max_step = 0.0;
};

std::string only_rejection(const json & file, const CbfSafetyConfig & config = limits())
{
  const auto whitelist = TemplateWhitelist::from_json(file.dump(), config);
  EXPECT_EQ(whitelist.segments.count("holy"), 0u);
  EXPECT_EQ(whitelist.segments.count("good"), 1u);
  return whitelist.rejected.size() == 1 ? whitelist.rejected.front() : std::string("<none>");
}

json with_good(const json & bad)
{
  return {{"schema", "x2_template_segments/v1"}, {"segments", {{"holy", bad}, {"good", segment_json()}}}};
}

}  // namespace

TEST(TemplateWhitelist, AValidSegmentLoadsWithEveryField)
{
  const auto config = limits();
  const auto whitelist = TemplateWhitelist::from_json(whitelist_json().dump(), config);
  ASSERT_EQ(whitelist.segments.count("holy"), 1u);
  EXPECT_TRUE(whitelist.rejected.empty());
  const auto & segment = whitelist.segments.at("holy");
  EXPECT_EQ(std::vector<double>(segment.P.begin(), segment.P.end()), start_arm());
  EXPECT_EQ(std::vector<double>(segment.hand_T.begin(), segment.hand_T.end()), target_hand());
  EXPECT_DOUBLE_EQ(segment.t_out_stop, 0.6);
  EXPECT_EQ(segment.report_sha256, kSha);
}

TEST(TemplateWhitelist, EachRuleRejectsOnlyItsSegment)
{
  const auto config = limits();
  struct Case
  {
    const char * key;
    json value;
    const char * expected;
  };
  auto above = start_arm();
  above[0] = config.upper[0] + 1e-3;
  auto below = target_arm();
  below[3] = config.lower[3] - 1e-3;
  auto thumb = target_hand();
  thumb[10] = aimdk_dataplane::cbf_thumb_command_band(config, 2)[1] + 1e-3;
  auto right_thumb = start_hand();
  right_thumb[11] = aimdk_dataplane::cbf_thumb_command_band(config, 3)[0] - 1e-3;
  const std::vector<Case> cases{
    {"P", std::vector<double>(13, 0.0), "P must have 14 values"},
    {"hand_T", std::vector<double>(19, 0.0), "hand_T must have 20 values"},
    {"T", json::array({nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), "T[0] is not a finite number"},
    {"t_in", 0.29, "t_in = 0.29 outside [0.3, 2]"},
    {"t_in", 2.01, "t_in = 2.01 outside [0.3, 2]"},
    {"t_out", 0.29, "t_out = 0.29 outside"},
    {"t_out_stop", 0.0, "t_out_stop = 0 outside (0, 0.6]"},
    {"t_out_stop", 0.61, "t_out_stop = 0.61 outside (0, 0.6]"},
    {"hold_max_s", 0.0, "hold_max_s = 0 outside (0, 10]"},
    {"hold_max_s", 10.01, "hold_max_s = 10.01 outside (0, 10]"},
    {"start_tol_rad", 0.0, "start_tol_rad must be positive"},
    {"hand_start_tol", -1.0, "hand_start_tol must be positive"},
    {"t_out", "1.5", "t_out must be a finite number"},
    {"report_sha256", std::string(63, 'a'), "report_sha256 must be 64 hexadecimal digits"},
    {"report_sha256", std::string(63, 'a') + "g", "report_sha256 must be 64 hexadecimal digits"},
    {"P", above, "P[0] = "},
    {"T", below, "T[3] = "},
    {"hand_T", thumb, "hand_T[10] = "},
    {"hand_P", right_thumb, "hand_P[11] = "},
  };
  for (const auto & test : cases) {
    auto bad = segment_json();
    bad[test.key] = test.value;
    const auto reason = only_rejection(with_good(bad));
    EXPECT_NE(reason.find(std::string("holy: ") + test.expected), std::string::npos)
      << test.key << " -> " << reason;
  }
  auto missing = segment_json();
  missing.erase("t_in");
  EXPECT_NE(only_rejection(with_good(missing)).find("t_in must be a finite number"), std::string::npos);
  missing = segment_json();
  missing.erase("report_sha256");
  EXPECT_NE(only_rejection(with_good(missing)).find("report_sha256 missing"), std::string::npos);
  EXPECT_NE(only_rejection(with_good(json::array())).find("not an object"), std::string::npos);
  // The joint and thumb limits name the bound they broke.
  auto bad = segment_json();
  bad["P"] = above;
  EXPECT_NE(only_rejection(with_good(bad)).find("outside the CBF joint limits"), std::string::npos);
  bad = segment_json();
  bad["hand_T"] = thumb;
  EXPECT_NE(only_rejection(with_good(bad)).find("outside the thumb command band"), std::string::npos);
}

TEST(TemplateWhitelist, EachOfTheThreeDurationsMustKeepTheQuinticWithinTheSpeedLimit)
{
  const auto config = limits();
  ASSERT_DOUBLE_EQ(config.velocity[3], 2.0);
  // 1.875 * 0.25 / t: 0.234 s is the shortest time this move allows.
  for (const char * key : {"t_in", "t_out", "t_out_stop"}) {
    auto bad = segment_json();
    bad[key] = std::string(key) == "t_out_stop" ? 0.2 : 0.3;
    if (std::string(key) != "t_out_stop") {
      auto far = target_arm();
      far[3] = start_arm()[3] + 0.33;  // 1.875 * 0.33 / 0.3 = 2.06 rad/s
      bad["T"] = far;
      bad["t_in"] = 2.0;
      bad["t_out"] = 2.0;
      bad[key] = 0.3;
    }
    const auto reason = only_rejection(with_good(bad));
    EXPECT_NE(reason.find(std::string("over ") + key), std::string::npos) << reason;
    EXPECT_NE(reason.find("above the CBF velocity limit 2"), std::string::npos) << reason;
  }
  // The bounds themselves are allowed.
  auto edge = segment_json();
  edge["t_in"] = 2.0;
  edge["t_out"] = 0.3;
  edge["t_out_stop"] = 0.6;
  edge["hold_max_s"] = 10.0;
  const auto whitelist = TemplateWhitelist::from_json(whitelist_json(edge).dump(), config);
  EXPECT_EQ(whitelist.segments.count("holy"), 1u) << (whitelist.rejected.empty() ? "" : whitelist.rejected[0]);
}

TEST(TemplateWhitelist, AMalformedFileThrows)
{
  const auto config = limits();
  EXPECT_THROW(TemplateWhitelist::from_json("{not json", config), std::exception);
  EXPECT_THROW(TemplateWhitelist::from_json(json::array().dump(), config), std::invalid_argument);
  auto file = whitelist_json();
  file["schema"] = "x2_template_segments/v2";
  EXPECT_THROW(TemplateWhitelist::from_json(file.dump(), config), std::invalid_argument);
  file = whitelist_json();
  file["segments"] = json::array();
  EXPECT_THROW(TemplateWhitelist::from_json(file.dump(), config), std::invalid_argument);
  file.erase("segments");
  EXPECT_THROW(TemplateWhitelist::from_json(file.dump(), config), std::invalid_argument);
}

TEST(TemplateCommand, ParsesTheThreeOpsAndRefusesAnythingElse)
{
  auto start = message("start", 17);
  start["seg"] = "holy";
  const auto parsed = parse_template_command(start.dump());
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->op, TemplateCommand::Op::kStart);
  EXPECT_EQ(parsed->segment, "holy");
  EXPECT_EQ(parsed->seg_id, 17u);
  EXPECT_EQ(parse_template_command(message("keepalive", 0).dump())->op, TemplateCommand::Op::kKeepalive);
  auto release = message("release", 17);
  EXPECT_FALSE(parse_template_command(release.dump())->fast);
  release["fast"] = true;
  EXPECT_TRUE(parse_template_command(release.dump())->fast);

  std::vector<json> bad;
  bad.push_back(message("start", 17));  // no seg
  auto empty = message("start", 17);
  empty["seg"] = "";
  bad.push_back(empty);
  auto fast = message("release", 17);
  fast["fast"] = 1;
  bad.push_back(fast);
  bad.push_back(message("stop", 17));
  auto schema = message("keepalive", 17);
  schema["schema"] = "x2_template_play/v2";
  bad.push_back(schema);
  auto no_schema = message("keepalive", 17);
  no_schema.erase("schema");
  bad.push_back(no_schema);
  auto negative = message("keepalive", 0);
  negative["seg_id"] = -1;
  bad.push_back(negative);
  auto fractional = message("keepalive", 0);
  fractional["seg_id"] = 1.5;
  bad.push_back(fractional);
  auto text_id = message("keepalive", 0);
  text_id["seg_id"] = "1";
  bad.push_back(text_id);
  auto no_id = message("keepalive", 0);
  no_id.erase("seg_id");
  bad.push_back(no_id);
  bad.push_back(json::array());
  for (const auto & value : bad) {EXPECT_FALSE(parse_template_command(value.dump())) << value.dump();}
  EXPECT_FALSE(parse_template_command("{"));
  EXPECT_FALSE(parse_template_command(""));
}

TEST(TemplatePlayer, AFullRoundPlaysFromThePublishedPointToTheTargetAndBackExactly)
{
  Rig rig;
  auto status = rig.last_status();
  EXPECT_EQ(status["state"], "IDLE");
  EXPECT_TRUE(status["seg_id"].is_null());
  EXPECT_EQ(status["segments"], json::array({"holy"}));
  EXPECT_FALSE(status.contains("P_actual"));

  // The published point is inside the tolerance but not P: it is where playing starts.
  auto arm = start_arm();
  arm[5] += 0.02;
  auto hand = start_hand();
  hand[4] -= 0.04;
  ASSERT_TRUE(rig.admit(7, published(arm, hand)));
  EXPECT_EQ(rig.player.state(), TemplateState::kEnterB);
  EXPECT_EQ(rig.player.epoch(), 1u);
  status = rig.last_status();
  EXPECT_EQ(status["state"], "ENTER_B");
  EXPECT_EQ(status["seg_id"], 7);
  EXPECT_EQ(status["seg"], "holy");
  EXPECT_EQ(status["P_actual"].get<std::vector<double>>(), arm);
  EXPECT_EQ(status["hand_P"].get<std::vector<double>>(), hand);
  EXPECT_EQ(status["counters"]["started"], 1);

  const int enter = rig.run_while(TemplateState::kEnterB, 7);
  EXPECT_NEAR(enter * kTick, 1.5, kTick + 1e-9);
  ASSERT_EQ(rig.player.state(), TemplateState::kHold);
  EXPECT_EQ(rig.last_status()["state"], "HOLD");
  for (int i = 0; i < 10; ++i) {
    const auto out = rig.tick(7);
    EXPECT_EQ(out.arm, target_arm());  // HOLD is T itself, not the start offset.
    EXPECT_EQ(out.hand, target_hand());
  }

  rig.release(7, false);
  const int exit = rig.run_while(TemplateState::kHold, 7) + rig.run_while(TemplateState::kExitB, 7);
  EXPECT_NEAR(exit * kTick, 1.5, 2 * kTick + 1e-9);
  ASSERT_EQ(rig.player.state(), TemplateState::kDone);
  EXPECT_EQ(rig.outputs.back().arm, arm);  // Exactly the published start point.
  EXPECT_EQ(rig.outputs.back().hand, hand);
  status = rig.last_status();
  EXPECT_EQ(status["state"], "DONE");
  EXPECT_EQ(status["reason"], "release");
  EXPECT_EQ(status["counters"]["done"], 1);
  EXPECT_EQ(status["counters"]["clamped"], 0);
  EXPECT_EQ(status["counters"]["mismatched"], 0);
  EXPECT_LE(rig.max_step, rig.config.velocity[0] * kTick + 1e-12);
  for (const auto & out : rig.outputs) {
    EXPECT_EQ(out.kind, Output::Kind::kUpperBody);
    EXPECT_EQ(out.hand_sub_mode, aimdk_dataplane::kHandSubModeDexterousJoint);
    EXPECT_EQ(out.sequence, 40u);
    EXPECT_EQ(out.head, (std::vector<double>{0.2, 0.3}));
  }

  // Afterwards nothing plays, a late keepalive is not a mismatch, and a new start is not busy.
  EXPECT_EQ(rig.tick(7).kind, Output::Kind::kNone);
  EXPECT_EQ(rig.player.counters().mismatched, 0u);
  EXPECT_TRUE(rig.admit(8));
  EXPECT_EQ(rig.last_status()["P_actual"].get<std::vector<double>>(), start_arm());
}

TEST(TemplatePlayer, EnterBStartsAtRestAndArrivesAtRest)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  rig.run_while(TemplateState::kEnterB, 1);
  ASSERT_GE(rig.outputs.size(), 3u);
  const auto & first = rig.outputs.front().arm;
  // Min-jerk: the first and last steps are tiny compared with the peak step.
  const double peak = 1.875 * kMove * kTick / 1.5;
  EXPECT_LT(std::abs(first[0] - start_arm()[0]), 0.01 * peak);
  const auto & before_last = rig.outputs[rig.outputs.size() - 2].arm;
  EXPECT_LT(std::abs(rig.outputs.back().arm[0] - before_last[0]), 0.01 * peak);
  EXPECT_NEAR(rig.max_step, peak, 0.01 * peak);
}

TEST(TemplatePlayer, AReleaseDuringEnterBReturnsInProportionWithoutAJump)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  for (int i = 0; i < 37; ++i) {rig.tick(1);}  // 0.74 s of 1.5 s: about half way.
  const std::size_t turn = rig.outputs.size();
  rig.release(1, false);
  const int exit = rig.run_while(TemplateState::kEnterB, 1) + rig.run_while(TemplateState::kExitB, 1);
  ASSERT_EQ(rig.player.state(), TemplateState::kDone);
  // Half of t_out, give or take the velocity-limit lengthening (none needed here).
  EXPECT_NEAR(exit * kTick, 0.5 * 1.5, 3 * kTick);
  EXPECT_EQ(rig.outputs.back().arm, start_arm());
  EXPECT_EQ(rig.player.counters().clamped, 0u);
  // Position and velocity carry through the turn: steps change only a little.
  for (std::size_t k = turn - 2; k < turn + 2; ++k) {
    const double before = rig.outputs[k].arm[0] - rig.outputs[k - 1].arm[0];
    const double after = rig.outputs[k + 1].arm[0] - rig.outputs[k].arm[0];
    EXPECT_LT(std::abs(after - before), 2e-3) << k;
  }
}

TEST(TemplatePlayer, AnEarlyReleaseStillTakesTheMinimumExitTime)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  rig.tick(1);
  rig.tick(1);
  rig.release(1, false);
  const int exit = rig.run_while(TemplateState::kEnterB, 1) + rig.run_while(TemplateState::kExitB, 1);
  ASSERT_EQ(rig.player.state(), TemplateState::kDone);
  EXPECT_GE(exit * kTick, TemplatePlayer::kMinExitS - 1e-9);
  EXPECT_LE(exit * kTick, TemplatePlayer::kMinExitS + 2 * kTick);
  EXPECT_EQ(rig.outputs.back().arm, start_arm());
}

TEST(TemplatePlayer, AMissingKeepaliveExitsFastFromHold)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  rig.run_while(TemplateState::kEnterB, 1);
  ASSERT_EQ(rig.player.state(), TemplateState::kHold);
  int quiet = 0;
  while (rig.player.state() == TemplateState::kHold && quiet < 100) {
    rig.tick();
    ++quiet;
  }
  // The last keepalive was at most 50 ms before the silence began.
  EXPECT_GT(quiet * kTick, TemplatePlayer::kKeepaliveTimeoutS - 0.05);
  EXPECT_LE(quiet * kTick, TemplatePlayer::kKeepaliveTimeoutS + kTick + 1e-9);
  ASSERT_EQ(rig.player.state(), TemplateState::kExitB);
  EXPECT_EQ(rig.last_status()["reason"], "keepalive_timeout");
  const int exit = rig.run_while(TemplateState::kExitB, std::nullopt);
  EXPECT_NEAR(exit * kTick, 0.6, 2 * kTick);
  EXPECT_EQ(rig.player.state(), TemplateState::kDone);
  EXPECT_EQ(rig.last_status()["reason"], "keepalive_timeout");
  EXPECT_EQ(rig.outputs.back().arm, start_arm());
}

TEST(TemplatePlayer, HoldEndsAtHoldMax)
{
  auto segment = segment_json();
  segment["hold_max_s"] = 0.5;
  Rig rig(whitelist_json(segment));
  ASSERT_TRUE(rig.admit(1));
  rig.run_while(TemplateState::kEnterB, 1);
  const int held = rig.run_while(TemplateState::kHold, 1);
  EXPECT_NEAR(held * kTick, 0.5, 2 * kTick);
  EXPECT_EQ(rig.player.state(), TemplateState::kExitB);
  EXPECT_EQ(rig.last_status()["reason"], "hold_max");
  const int exit = rig.run_while(TemplateState::kExitB, 1);
  EXPECT_NEAR(exit * kTick, 1.5, 2 * kTick);
  EXPECT_EQ(rig.last_status()["reason"], "hold_max");
}

TEST(TemplatePlayer, AFasterExitReplacesASlowerOneButNotTheOtherWayRound)
{
  Rig slow;
  ASSERT_TRUE(slow.admit(1));
  slow.run_while(TemplateState::kEnterB, 1);
  slow.release(1, false);
  for (int i = 0; i < 15; ++i) {slow.tick(1);}
  ASSERT_EQ(slow.player.state(), TemplateState::kExitB);
  slow.release(1, false);  // A repeat changes nothing.
  slow.tick(1);
  EXPECT_EQ(slow.last_status()["reason"], "release");
  slow.release(1, true);
  const int rest = slow.run_while(TemplateState::kExitB, 1);
  EXPECT_EQ(slow.last_status()["reason"], "release_fast");
  EXPECT_LT((16 + rest) * kTick, 1.5 - 0.3);
  EXPECT_EQ(slow.outputs.back().arm, start_arm());
  EXPECT_EQ(slow.player.counters().clamped, 0u);

  Rig fast;
  ASSERT_TRUE(fast.admit(1));
  fast.run_while(TemplateState::kEnterB, 1);
  fast.release(1, true);
  fast.tick(1);
  fast.release(1, false);
  const int left = fast.run_while(TemplateState::kExitB, 1);
  EXPECT_EQ(fast.last_status()["reason"], "release_fast");
  EXPECT_NEAR((1 + left) * kTick, 0.6, 2 * kTick);
}

TEST(TemplatePlayer, ARuntimeStopEndsPlayingAtOnce)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  rig.run_while(TemplateState::kEnterB, 1);
  rig.player.stop("revoked");
  EXPECT_EQ(rig.player.state(), TemplateState::kStopped);
  EXPECT_FALSE(rig.player.playing());
  EXPECT_EQ(rig.tick(1).kind, Output::Kind::kNone);
  const auto status = rig.last_status();
  EXPECT_EQ(status["state"], "STOPPED");
  EXPECT_EQ(status["reason"], "revoked");
  EXPECT_EQ(status["counters"]["stopped"], 1);
  rig.player.stop("epoch_changed");  // Not playing: nothing to stop.
  EXPECT_EQ(rig.player.counters().stopped, 1u);
  EXPECT_TRUE(rig.admit(2));
}

TEST(TemplatePlayer, EveryPlayedStepKeepsTheSpeedLimitEvenAfterALateTick)
{
  auto segment = segment_json();
  segment["t_in"] = 0.3;  // Peak 1.5625 rad/s of the 2 rad/s limit.
  segment["t_out"] = 0.3;
  segment["t_out_stop"] = 0.3;
  Rig rig(whitelist_json(segment));
  ASSERT_TRUE(rig.admit(1));
  rig.tick(1);
  rig.tick(1);
  rig.now += 3 * kTick;  // Three ticks missed: the player still moves one tick's worth.
  rig.tick(1);
  EXPECT_GE(rig.player.counters().clamped, 1u);
  EXPECT_LE(rig.max_step, rig.config.velocity[0] * kTick + 1e-12);
  rig.run_while(TemplateState::kEnterB, 1);
  for (int i = 0; i < 10; ++i) {rig.tick(1);}
  EXPECT_EQ(rig.outputs.back().arm, target_arm());  // It catches up within the limit.
  rig.release(1, false);
  rig.run_while(TemplateState::kHold, 1);
  rig.run_while(TemplateState::kExitB, 1);
  EXPECT_EQ(rig.player.state(), TemplateState::kDone);
  EXPECT_LE(rig.max_step, rig.config.velocity[0] * kTick + 1e-12);
}

TEST(TemplatePlayer, TheHeadFollowsThePublicStreamElseStaysWhereItStarted)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  rig.now += kTick;
  EXPECT_EQ(rig.player.step(rig.requested(), rig.now).head, (std::vector<double>{0.2, 0.3}));
  rig.now += kTick;
  EXPECT_EQ(rig.player.step(Output{}, rig.now).head, (std::vector<double>{0.1, -0.1}));
}

TEST(TemplatePlayer, StartsThatCannotBeJudgedAreRefusedWithTheirReason)
{
  const auto config = limits();
  TemplatePlayer disabled(std::nullopt, {}, true);
  auto start = message("start", 3);
  start["seg"] = "holy";
  disabled.on_message(start.dump(), 1.0);
  auto status = json::parse(disabled.take_status(1.0).back());
  EXPECT_EQ(status["state"], "REJECTED");
  EXPECT_EQ(status["reason"], "disabled");
  EXPECT_EQ(status["seg_id"], 3);

  TemplatePlayer unloaded(config, {}, false);
  unloaded.on_message(start.dump(), 1.0);
  EXPECT_EQ(json::parse(unloaded.take_status(1.0).back())["reason"], "no_whitelist");

  Rig rig;
  rig.start(4, "heart");
  EXPECT_FALSE(rig.player.start_pending());
  status = rig.last_status();
  EXPECT_EQ(status["reason"], "unknown_segment");
  EXPECT_EQ(status["seg"], "heart");

  // A release before the gate runs withdraws the start.
  rig.start(5);
  ASSERT_TRUE(rig.player.start_pending());
  rig.release(5, false);
  EXPECT_FALSE(rig.player.start_pending());
  EXPECT_FALSE(rig.player.evaluate_start(published(), 1, true, rig.now));
  status = rig.last_status();
  EXPECT_EQ(status["reason"], "released");
  EXPECT_EQ(status["seg_id"], 5);

  // The node refuses a start on a tick with nothing to judge.
  rig.start(6);
  rig.player.reject_pending("not_admitted");
  EXPECT_EQ(rig.last_status()["reason"], "not_admitted");
  EXPECT_EQ(rig.player.counters().rejected, 3u);
}

TEST(TemplatePlayer, TheGateRefusesEachUnsuitablePublishedDecision)
{
  struct Case
  {
    const char * reason;
    CbfDecision decision;
    bool enforce;
  };
  std::vector<Case> cases;
  auto none = published();
  none.output.kind = Output::Kind::kNone;
  cases.push_back({"not_upper_body", none, true});
  auto short_arm = published();
  short_arm.output.arm.pop_back();
  cases.push_back({"not_upper_body", short_arm, false});
  auto short_hand = published();
  short_hand.output.hand.pop_back();
  cases.push_back({"not_upper_body", short_hand, false});
  auto gripper = published();
  gripper.output.hand_sub_mode = 0;
  cases.push_back({"not_upper_body", gripper, false});
  auto epoch = published();
  epoch.epoch = 2;
  cases.push_back({"not_admitted", epoch, true});
  auto held = published();
  held.accept_new = false;
  cases.push_back({"cbf_intervening", held, true});
  auto released = published();
  released.released = true;
  cases.push_back({"cbf_intervening", released, true});
  auto modified = published();
  modified.modified = true;
  cases.push_back({"cbf_intervening", modified, true});
  auto off = published();
  off.output.arm[9] += 0.03 + 1e-6;
  cases.push_back({"arm_off_start", off, false});
  auto nan_arm = published();
  nan_arm.output.arm[2] = std::numeric_limits<double>::quiet_NaN();
  cases.push_back({"arm_off_start", nan_arm, false});
  auto off_hand = published();
  off_hand.output.hand[4] += 0.05 + 1e-6;
  cases.push_back({"hand_off_start", off_hand, false});
  // The right hand counts as much as the left, installed or not.
  auto off_right = published();
  off_right.output.hand[15] -= 0.05 + 1e-6;
  cases.push_back({"hand_off_start", off_right, true});
  auto nan_hand = published();
  nan_hand.output.hand[19] = std::numeric_limits<double>::quiet_NaN();
  cases.push_back({"hand_off_start", nan_hand, false});

  for (const auto & test : cases) {
    auto config = limits(test.enforce ? "enforce" : "shadow");
    Rig rig(whitelist_json(), config);
    rig.start(11);
    EXPECT_FALSE(rig.player.evaluate_start(test.decision, 1, test.enforce, rig.now)) << test.reason;
    const auto status = rig.last_status();
    EXPECT_EQ(status["state"], "REJECTED") << test.reason;
    EXPECT_EQ(status["reason"], test.reason);
    EXPECT_EQ(status["seg_id"], 11);
    EXPECT_FALSE(status.contains("P_actual"));
    EXPECT_FALSE(rig.player.playing());
  }
}

TEST(TemplatePlayer, ShadowAdmitsWhateverTheFilterDidAndEnforceNeedsAnUnmodifiedFrame)
{
  for (const bool enforce : {false, true}) {
    Rig rig(whitelist_json(), limits(enforce ? "enforce" : "shadow"));
    auto decision = published();
    decision.accept_new = false;
    decision.modified = true;
    decision.epoch = 9;
    rig.start(1);
    EXPECT_EQ(rig.player.evaluate_start(decision, 1, enforce, rig.now), !enforce);
  }
  // Within tolerance on every value is admitted.
  Rig rig;
  auto decision = published();
  for (std::size_t j = 0; j < 14; ++j) {decision.output.arm[j] += j == 8 ? -0.029 : 0.029;}
  for (auto & v : decision.output.hand) {v -= 0.049;}
  EXPECT_TRUE(rig.admit(1, decision));
}

TEST(TemplatePlayer, BusyRefusesTheNewStartAndLeavesThePlayingOneAlone)
{
  Rig rig;
  ASSERT_TRUE(rig.admit(1));
  rig.tick(1);
  rig.last_status();
  rig.start(2);
  EXPECT_FALSE(rig.player.start_pending());
  const auto messages = rig.player.take_status(rig.now);
  ASSERT_EQ(messages.size(), 1u);
  const auto busy = json::parse(messages.front());
  EXPECT_EQ(busy["state"], "REJECTED");
  EXPECT_EQ(busy["reason"], "busy");
  EXPECT_EQ(busy["seg_id"], 2);
  EXPECT_EQ(rig.player.state(), TemplateState::kEnterB);
  rig.now += TemplatePlayer::kStatusPeriodS;
  const auto current = rig.last_status();
  EXPECT_EQ(current["state"], "ENTER_B");
  EXPECT_EQ(current["seg_id"], 1);
  // Messages for another segment are counted, not acted on.
  rig.release(2, true);
  rig.keepalive(2);
  EXPECT_EQ(rig.player.counters().mismatched, 2u);
  EXPECT_EQ(rig.player.state(), TemplateState::kEnterB);
  rig.player.on_message("garbage", rig.now);
  EXPECT_EQ(rig.player.counters().malformed, 1u);

  // A second start while one is pending is busy too.
  Rig pending;
  pending.start(1);
  pending.start(2);
  const auto refused = pending.player.take_status(pending.now);
  ASSERT_FALSE(refused.empty());
  EXPECT_EQ(json::parse(refused.front())["reason"], "busy");
  EXPECT_EQ(json::parse(refused.back())["state"], "IDLE");
  EXPECT_TRUE(pending.player.evaluate_start(published(), 1, true, pending.now));
  EXPECT_EQ(pending.last_status()["seg_id"], 1);
}

TEST(TemplatePlayer, StatusIsSentOnEveryChangeAndEveryHundredMilliseconds)
{
  Rig rig;
  EXPECT_EQ(rig.player.take_status(rig.now).size(), 1u);
  EXPECT_TRUE(rig.player.take_status(rig.now + 0.05).empty());
  EXPECT_EQ(rig.player.take_status(rig.now + 0.1).size(), 1u);
  rig.now += 0.11;
  rig.start(4, "heart");
  // The refusal once, not twice.
  EXPECT_EQ(rig.player.take_status(rig.now).size(), 1u);
  EXPECT_TRUE(rig.player.take_status(rig.now + 0.01).empty());
  const auto again = rig.player.take_status(rig.now + 0.1);
  ASSERT_EQ(again.size(), 1u);
  EXPECT_EQ(json::parse(again.front())["reason"], "unknown_segment");
  const auto status = json::parse(again.front());
  EXPECT_EQ(status["schema"], "x2_template_status/v1");
  EXPECT_DOUBLE_EQ(status["at"].get<double>(), rig.now + 0.1);
  for (const char * key : {"started", "done", "rejected", "stopped", "malformed", "mismatched", "clamped"}) {
    EXPECT_TRUE(status["counters"].contains(key)) << key;
  }
}
