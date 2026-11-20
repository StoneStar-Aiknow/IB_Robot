// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The template playback port: the data plane plays a whitelisted contact
// segment by itself, P -> T -> P, while the mirror session only starts it,
// keeps it alive and releases it. No ROS or vendor types; times are seconds on
// the caller's monotonic clock. The interface (whitelist, messages, status) is
// /mnt/oee/X2-wt/template-play-M0-interface.md.
//
// A segment is played from what the data plane actually published on the tick
// its start was admitted, so the switch in and out of the CBF never jumps.
// Every played point stays inside the CBF's joint limits and per-tick velocity
// limit. A session-level exit (release, a missing keepalive, the hold limit, an
// internal error) plays back to that start point; a runtime-level exit
// (revocation, control timeout, epoch change) stops at once and publishes nothing.

#ifndef AIMDK_DATAPLANE__TEMPLATE_PLAY_HPP_
#define AIMDK_DATAPLANE__TEMPLATE_PLAY_HPP_

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "aimdk_dataplane/core.hpp"

namespace aimdk_dataplane
{

struct TemplateSegment
{
  std::string name;
  std::array<double, 14> P{};  // native radians, kVendorArmOrder
  std::array<double, 14> T{};
  std::array<double, 20> hand_P{};  // public-stream hand layout, L 10 then R 10, command values
  std::array<double, 20> hand_T{};
  double t_in = 0.0;
  double t_out = 0.0;
  double t_out_stop = 0.0;
  double hold_max_s = 0.0;
  double start_tol_rad = 0.0;
  double hand_start_tol = 0.0;
  std::string report_sha256;
};

// Read once at startup. A malformed file throws; a malformed segment is left
// out with its reason, and the others stay usable.
struct TemplateWhitelist
{
  std::map<std::string, TemplateSegment> segments;
  std::vector<std::string> rejected;  // "<name>: <reason>"

  static TemplateWhitelist from_json(const std::string & text, const CbfSafetyConfig & limits);
};

struct TemplateCommand
{
  enum class Op { kStart, kKeepalive, kRelease, kACap };
  Op op = Op::kKeepalive;
  std::string segment;
  uint64_t seg_id = 0;  // Not read for a_cap.
  bool fast = false;
  bool on = false;  // a_cap only, with the safety epoch it was sent for
  uint64_t epoch = 0;
  std::optional<uint64_t> attempt;  // and the session's own tag, echoed back.
};

// nullopt for anything that is not a well-formed x2_template_play/v1 message.
std::optional<TemplateCommand> parse_template_command(const std::string & text);

enum class TemplateState { kIdle, kEnterB, kHold, kExitB, kDone, kRejected, kStopped };
const char * template_state_name(TemplateState state);

struct TemplateCounters
{
  uint64_t started = 0;
  uint64_t done = 0;
  uint64_t rejected = 0;
  uint64_t stopped = 0;
  uint64_t malformed = 0;
  uint64_t mismatched = 0;
  uint64_t clamped = 0;  // Played ticks a joint, velocity or thumb limit changed; 0 for a valid segment.
  uint64_t a_cap_epoch_mismatch = 0;  // Ticks a live a_cap carried another safety epoch.
};

class TemplatePlayer
{
public:
  static constexpr double kKeepaliveTimeoutS = 0.15;
  // Judged on on-time ticks only (at most two periods after the previous one):
  // a stalled executor runs the late tick before the queued messages. At twice
  // the time it lapses on any tick.
  static constexpr double kACapTimeoutS = 0.3;
  static constexpr double kStatusPeriodS = 0.1;
  static constexpr double kMinExitS = 0.3;

  // ``limits`` is the CBF's configuration; without it every start is refused
  // as disabled. ``whitelist_loaded`` is false when the file could not be read.
  TemplatePlayer(std::optional<CbfSafetyConfig> limits, TemplateWhitelist whitelist, bool whitelist_loaded);
  // segment_ points into the player's own whitelist.
  TemplatePlayer(const TemplatePlayer &) = delete;
  TemplatePlayer & operator=(const TemplatePlayer &) = delete;

  // A message from the mirror session; counted and dropped if malformed.
  void on_message(const std::string & text, double now);

  // The template A segment's release cap, kept up by the session with a_cap
  // messages. Judged once per command tick, before the tick uses it: it holds
  // while the latest a_cap is on, for ``safety_epoch`` and not lapsed. Without
  // the CBF's configuration it never holds.
  bool judge_a_cap(uint64_t safety_epoch, double now);
  bool a_cap() const {return a_cap_valid_;}

  bool playing() const;
  bool start_pending() const {return pending_.has_value();}
  TemplateState state() const {return state_;}
  uint64_t epoch() const {return epoch_;}

  // The tick's CBF path decided ``decision`` and is about to publish it: admit
  // or refuse a pending start against it. True when admitted; the decision is
  // published unchanged either way and playing starts on the next tick. A start
  // not kept alive for kKeepaliveTimeoutS is refused as stale_start.
  // ``hand_reported_absent`` is the tick's snapshot: an installed side reported
  // as no hand has its thumb at phantom_thumb_native, never hand_P.
  bool evaluate_start(
    const CbfDecision & decision, uint64_t epoch, bool enforce, double now,
    const std::array<bool, 2> & hand_reported_absent);
  // Refuse a pending start on a tick that published nothing the gate can judge.
  void reject_pending(const std::string & reason);

  // The point to publish this tick while playing. ``requested`` is the public
  // stream's candidate, used for the head only. kNone after an internal stop.
  Output step(const Output & requested, double now);
  // Runtime-level exit: stop now, publish nothing more.
  void stop(const std::string & reason);

  // Status messages to publish now, oldest first: one-off refusals, then the
  // current state on a change or once per kStatusPeriodS.
  std::vector<std::string> take_status(double now);
  const TemplateCounters & counters() const {return counters_;}
  const TemplateWhitelist & whitelist() const {return whitelist_;}

private:
  static constexpr size_t kDims = 34;  // 14 arm then 20 hand
  using Vec = std::array<double, kDims>;
  struct Trajectory
  {
    double start = 0.0;
    double duration = 0.0;
    std::array<std::array<double, 6>, kDims> c{};  // position polynomial c0..c5 per dimension
  };
  struct Pending
  {
    std::string segment;
    uint64_t seg_id = 0;
    double received = 0.0;  // The start or its latest keepalive.
  };
  struct ACap
  {
    bool on = false;
    uint64_t epoch = 0;
    std::optional<uint64_t> attempt;
    double received = 0.0;
    bool lapsed = false;  // Until the next a_cap message.
  };
  struct Refusal
  {
    uint64_t seg_id = 0;
    std::string segment;
    std::string reason;
  };

  void transition(TemplateState state, const std::string & reason);
  // Every refusal is published once with its own seg_id; when nothing is
  // playing or waiting it also becomes the current state.
  void refuse(uint64_t seg_id, const std::string & segment, const std::string & reason);
  void plan(const Vec & q, const Vec & v, const Vec & a, const Vec & goal, double duration, double now);
  // An exit from the planned state at ``now`` back to the start point within
  // ``chosen`` seconds: proportional to the distance left, at least kMinExitS
  // (or ``chosen`` if shorter), longer if the velocity limit needs it, never beyond ``chosen``.
  void begin_exit(double chosen, const std::string & reason, double now);
  void evaluate(double now, Vec & q, Vec & v, Vec & a) const;
  Vec start_point() const;
  Vec target_point() const;
  std::string status_json(double now, std::optional<uint64_t> seg_id, const std::string & segment,
    TemplateState state, const std::string & reason) const;

  std::optional<CbfSafetyConfig> limits_;
  TemplateWhitelist whitelist_;
  bool whitelist_loaded_ = false;

  std::optional<Pending> pending_;
  TemplateState state_ = TemplateState::kIdle;
  std::string reason_;
  std::optional<uint64_t> seg_id_;
  std::string segment_name_;
  const TemplateSegment * segment_ = nullptr;
  uint64_t epoch_ = 0;
  bool have_start_point_ = false;
  std::array<double, 14> p_actual_{};
  std::array<double, 20> hand_p_{};
  std::vector<double> start_head_;
  int hand_sub_mode_ = kHandSubModeDexterousJoint;
  Trajectory trajectory_;
  double hold_since_ = 0.0;
  double last_keepalive_ = 0.0;
  std::optional<std::pair<double, std::string>> release_;  // A release for the next step: its time and reason.
  double exit_chosen_ = 0.0;  // The time chosen for the current exit.
  bool keepalive_lapsed_ = false;
  bool internal_error_ = false;
  std::optional<std::array<double, 14>> last_arm_;
  double last_step_ = 0.0;

  TemplateCounters counters_;
  std::vector<Refusal> outbox_;
  bool changed_ = false;
  double last_status_ = -1.0;
  std::optional<ACap> a_cap_;
  bool a_cap_valid_ = false;
  std::optional<ACap> a_cap_held_;  // The message it holds on, while it does.
  bool a_cap_changed_ = false;
  std::optional<double> last_tick_;  // The previous judge_a_cap.
};

}  // namespace aimdk_dataplane

#endif  // AIMDK_DATAPLANE__TEMPLATE_PLAY_HPP_
