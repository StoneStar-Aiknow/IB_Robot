// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
#include "aimdk_dataplane/template_play.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "nlohmann/json.hpp"

namespace aimdk_dataplane
{
namespace
{
using nlohmann::json;
constexpr std::array<size_t, 4> kThumbSlots{0, 1, 10, 11};
constexpr double kMaxBlendS = 2.0;
constexpr double kMinBlendS = 0.3;
constexpr double kMaxStopS = 0.6;
constexpr double kMaxHoldS = 10.0;
// Peak speed of a rest-to-rest quintic over distance d and time t is 1.875 d / t.
constexpr double kQuinticPeak = 1.875;
// Ticks land on a period; rounding must not push an end one tick later.
constexpr double kTimeEpsilonS = 1e-9;

template<size_t N>
std::array<double, N> vector_field(const json & segment, const char * key)
{
  if (!segment.contains(key)) {throw std::invalid_argument(std::string(key) + " missing");}
  const auto & entry = segment.at(key);
  if (!entry.is_array() || entry.size() != N) {
    throw std::invalid_argument(std::string(key) + " must have " + std::to_string(N) + " values");
  }
  std::array<double, N> out{};
  for (size_t i = 0; i < N; ++i) {
    if (!entry[i].is_number() || !std::isfinite(entry[i].get<double>())) {
      throw std::invalid_argument(std::string(key) + "[" + std::to_string(i) + "] is not a finite number");
    }
    out[i] = entry[i].get<double>();
  }
  return out;
}

double number_field(const json & segment, const char * key)
{
  if (!segment.contains(key) || !segment.at(key).is_number() || !std::isfinite(segment.at(key).get<double>())) {
    throw std::invalid_argument(std::string(key) + " must be a finite number");
  }
  return segment.at(key).get<double>();
}

std::string describe(double value)
{
  std::ostringstream text;
  text.precision(6);
  text << value;
  return text.str();
}

void within(const char * key, double value, double lower, double upper, bool lower_open)
{
  if ((lower_open ? value <= lower : value < lower) || value > upper) {
    throw std::invalid_argument(std::string(key) + " = " + describe(value) + " outside " +
            (lower_open ? "(" : "[") + describe(lower) + ", " + describe(upper) + "]");
  }
}

TemplateSegment parse_segment(const std::string & name, const json & value, const CbfSafetyConfig & limits)
{
  if (!value.is_object()) {throw std::invalid_argument("not an object");}
  TemplateSegment out;
  out.name = name;
  out.P = vector_field<14>(value, "P");
  out.T = vector_field<14>(value, "T");
  out.hand_P = vector_field<20>(value, "hand_P");
  out.hand_T = vector_field<20>(value, "hand_T");
  out.t_in = number_field(value, "t_in");
  out.t_out = number_field(value, "t_out");
  out.t_out_stop = number_field(value, "t_out_stop");
  out.hold_max_s = number_field(value, "hold_max_s");
  out.start_tol_rad = number_field(value, "start_tol_rad");
  out.hand_start_tol = number_field(value, "hand_start_tol");
  within("t_in", out.t_in, kMinBlendS, kMaxBlendS, false);
  within("t_out", out.t_out, kMinBlendS, kMaxBlendS, false);
  within("t_out_stop", out.t_out_stop, kMinBlendS, kMaxStopS, false);
  within("hold_max_s", out.hold_max_s, 0.0, kMaxHoldS, true);
  if (out.start_tol_rad <= 0.0) {throw std::invalid_argument("start_tol_rad must be positive");}
  if (out.hand_start_tol <= 0.0) {throw std::invalid_argument("hand_start_tol must be positive");}
  if (!value.contains("report_sha256") || !value.at("report_sha256").is_string()) {
    throw std::invalid_argument("report_sha256 missing");
  }
  out.report_sha256 = value.at("report_sha256").get<std::string>();
  if (out.report_sha256.size() != 64 ||
    !std::all_of(out.report_sha256.begin(), out.report_sha256.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }))
  {
    throw std::invalid_argument("report_sha256 must be 64 hexadecimal digits");
  }
  for (size_t j = 0; j < 14; ++j) {
    for (const auto & [key, pose] : {std::pair<const char *, const std::array<double, 14> *>{"P", &out.P},
        std::pair<const char *, const std::array<double, 14> *>{"T", &out.T}})
    {
      const double v = (*pose)[j];
      if (v < limits.lower[j] || v > limits.upper[j]) {
        throw std::invalid_argument(std::string(key) + "[" + std::to_string(j) + "] = " + describe(v) +
                " outside the CBF joint limits [" + describe(limits.lower[j]) + ", " + describe(limits.upper[j]) + "]");
      }
    }
    const double distance = std::abs(out.T[j] - out.P[j]);
    for (const auto & [key, time] : {std::pair<const char *, double>{"t_in", out.t_in},
        std::pair<const char *, double>{"t_out", out.t_out}, std::pair<const char *, double>{"t_out_stop", out.t_out_stop}})
    {
      const double speed = kQuinticPeak * distance / time;
      if (speed > limits.velocity[j]) {
        throw std::invalid_argument(std::string("joint ") + std::to_string(j) + " peaks at " + describe(speed) +
                " rad/s over " + key + " = " + describe(time) + " s, above the CBF velocity limit " +
                describe(limits.velocity[j]));
      }
    }
  }
  for (size_t axis = 0; axis < kThumbSlots.size(); ++axis) {
    const auto band = cbf_thumb_command_band(limits, axis);
    for (const auto & [key, hand] : {std::pair<const char *, const std::array<double, 20> *>{"hand_P", &out.hand_P},
        std::pair<const char *, const std::array<double, 20> *>{"hand_T", &out.hand_T}})
    {
      const double v = (*hand)[kThumbSlots[axis]];
      if (v < band[0] || v > band[1]) {
        throw std::invalid_argument(std::string(key) + "[" + std::to_string(kThumbSlots[axis]) + "] = " +
                describe(v) + " outside the thumb command band [" + describe(band[0]) + ", " + describe(band[1]) + "]");
      }
    }
  }
  return out;
}

bool finite_values(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);});
}
}  // namespace

TemplateWhitelist TemplateWhitelist::from_json(const std::string & text, const CbfSafetyConfig & limits)
{
  const auto value = json::parse(text);
  if (!value.is_object() || !value.contains("schema") || value.at("schema") != "x2_template_segments/v1") {
    throw std::invalid_argument("template whitelist schema is not x2_template_segments/v1");
  }
  if (!value.contains("segments") || !value.at("segments").is_object()) {
    throw std::invalid_argument("template whitelist has no segments object");
  }
  TemplateWhitelist out;
  for (const auto & [name, segment] : value.at("segments").items()) {
    try {
      out.segments.emplace(name, parse_segment(name, segment, limits));
    } catch (const std::exception & error) {
      out.rejected.push_back(name + ": " + error.what());
    }
  }
  return out;
}

std::optional<TemplateCommand> parse_template_command(const std::string & text)
{
  try {
    const auto value = json::parse(text);
    if (!value.is_object() || !value.contains("schema") || value.at("schema") != "x2_template_play/v1" ||
      !value.contains("op") || !value.at("op").is_string() || !value.contains("seg_id") ||
      !value.at("seg_id").is_number_integer() || value.at("seg_id").get<int64_t>() < 0)
    {
      return std::nullopt;
    }
    TemplateCommand out;
    out.seg_id = value.at("seg_id").get<uint64_t>();
    const auto op = value.at("op").get<std::string>();
    if (op == "start") {
      if (!value.contains("seg") || !value.at("seg").is_string() || value.at("seg").get<std::string>().empty()) {
        return std::nullopt;
      }
      out.op = TemplateCommand::Op::kStart;
      out.segment = value.at("seg").get<std::string>();
    } else if (op == "keepalive") {
      out.op = TemplateCommand::Op::kKeepalive;
    } else if (op == "release") {
      out.op = TemplateCommand::Op::kRelease;
      if (value.contains("fast")) {
        if (!value.at("fast").is_boolean()) {return std::nullopt;}
        out.fast = value.at("fast").get<bool>();
      }
    } else {
      return std::nullopt;
    }
    return out;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

const char * template_state_name(TemplateState state)
{
  switch (state) {
    case TemplateState::kIdle: return "IDLE";
    case TemplateState::kEnterB: return "ENTER_B";
    case TemplateState::kHold: return "HOLD";
    case TemplateState::kExitB: return "EXIT_B";
    case TemplateState::kDone: return "DONE";
    case TemplateState::kRejected: return "REJECTED";
    case TemplateState::kStopped: return "STOPPED";
  }
  return "IDLE";
}

TemplatePlayer::TemplatePlayer(
  std::optional<CbfSafetyConfig> limits, TemplateWhitelist whitelist, bool whitelist_loaded)
: limits_(std::move(limits)), whitelist_(std::move(whitelist)), whitelist_loaded_(whitelist_loaded) {}

bool TemplatePlayer::playing() const
{
  return state_ == TemplateState::kEnterB || state_ == TemplateState::kHold || state_ == TemplateState::kExitB;
}

void TemplatePlayer::on_message(const std::string & text, double now)
{
  const auto command = parse_template_command(text);
  if (!command) {
    ++counters_.malformed;
    return;
  }
  switch (command->op) {
    case TemplateCommand::Op::kStart:
      // A start nobody kept alive is not waited for: the session has long taken it as refused.
      if (pending_ && now - pending_->received > kKeepaliveTimeoutS) {
        reject_pending("stale_start");
      }
      if (playing() || pending_) {
        refuse(command->seg_id, command->segment, "busy");
      } else if (!limits_) {
        refuse(command->seg_id, command->segment, "disabled");
      } else if (!whitelist_loaded_) {
        refuse(command->seg_id, command->segment, "no_whitelist");
      } else if (whitelist_.segments.count(command->segment) == 0) {
        refuse(command->seg_id, command->segment, "unknown_segment");
      } else {
        pending_ = Pending{command->segment, command->seg_id, now};
      }
      return;
    case TemplateCommand::Op::kKeepalive:
      if (pending_ && pending_->seg_id == command->seg_id) {
        pending_->received = now;
      } else if (playing() && seg_id_ == command->seg_id) {
        last_keepalive_ = now;
      } else if (seg_id_ != command->seg_id) {
        ++counters_.mismatched;
      }
      return;
    case TemplateCommand::Op::kRelease:
      if (pending_ && pending_->seg_id == command->seg_id) {
        const auto pending = *pending_;
        pending_.reset();
        refuse(pending.seg_id, pending.segment, "released");
      } else if (playing() && seg_id_ == command->seg_id) {
        const double time = command->fast ? segment_->t_out_stop : segment_->t_out;
        if (!release_ || time < release_->first) {
          release_ = std::make_pair(time, std::string(command->fast ? "release_fast" : "release"));
        }
      } else if (seg_id_ != command->seg_id) {
        ++counters_.mismatched;
      }
      return;
  }
}

bool TemplatePlayer::evaluate_start(const CbfDecision & decision, uint64_t epoch, bool enforce, double now)
{
  if (!pending_) {return false;}
  const auto pending = *pending_;
  pending_.reset();
  if (now - pending.received > kKeepaliveTimeoutS) {
    refuse(pending.seg_id, pending.segment, "stale_start");
    return false;
  }
  const auto found = whitelist_.segments.find(pending.segment);
  if (!limits_) {
    refuse(pending.seg_id, pending.segment, "disabled");
    return false;
  }
  if (found == whitelist_.segments.end()) {
    refuse(pending.seg_id, pending.segment, "unknown_segment");
    return false;
  }
  const auto & out = decision.output;
  if (out.kind != Output::Kind::kUpperBody || out.arm.size() != 14 || out.hand.size() != 20 ||
    out.hand_sub_mode != kHandSubModeDexterousJoint)
  {
    refuse(pending.seg_id, pending.segment, "not_upper_body");
    return false;
  }
  if (enforce && decision.epoch != epoch) {
    refuse(pending.seg_id, pending.segment, "not_admitted");
    return false;
  }
  if (enforce && (!decision.accept_new || decision.released || decision.modified)) {
    refuse(pending.seg_id, pending.segment, "cbf_intervening");
    return false;
  }
  const auto & segment = found->second;
  // Written so a NaN fails every comparison. The start point must also lie in
  // the limits every played point is clamped to, so the exit can reach it exactly.
  for (size_t j = 0; j < 14; ++j) {
    if (!(std::abs(out.arm[j] - segment.P[j]) <= segment.start_tol_rad) ||
      !(out.arm[j] >= limits_->lower[j] && out.arm[j] <= limits_->upper[j]))
    {
      refuse(pending.seg_id, pending.segment, "arm_off_start");
      return false;
    }
  }
  for (size_t i = 0; i < 20; ++i) {
    if (!(std::abs(out.hand[i] - segment.hand_P[i]) <= segment.hand_start_tol)) {
      refuse(pending.seg_id, pending.segment, "hand_off_start");
      return false;
    }
  }
  for (size_t axis = 0; axis < kThumbSlots.size(); ++axis) {
    const auto band = cbf_thumb_command_band(*limits_, axis);
    const double v = out.hand[kThumbSlots[axis]];
    if (!(v >= band[0] && v <= band[1])) {
      refuse(pending.seg_id, pending.segment, "hand_off_start");
      return false;
    }
  }
  segment_ = &segment;
  segment_name_ = pending.segment;
  seg_id_ = pending.seg_id;
  epoch_ = epoch;
  std::copy(out.arm.begin(), out.arm.end(), p_actual_.begin());
  std::copy(out.hand.begin(), out.hand.end(), hand_p_.begin());
  have_start_point_ = true;
  start_head_ = out.head;
  hand_sub_mode_ = out.hand_sub_mode;
  last_keepalive_ = pending.received;
  release_.reset();
  exit_chosen_ = 0.0;
  keepalive_lapsed_ = false;
  internal_error_ = false;
  last_arm_ = p_actual_;
  last_step_ = now;
  plan(start_point(), Vec{}, Vec{}, target_point(), segment.t_in, now);
  ++counters_.started;
  transition(TemplateState::kEnterB, "");
  return true;
}

void TemplatePlayer::reject_pending(const std::string & reason)
{
  if (!pending_) {return;}
  const auto pending = *pending_;
  pending_.reset();
  refuse(pending.seg_id, pending.segment, reason);
}

Output TemplatePlayer::step(const Output & requested, double now)
{
  if (!playing() || !segment_ || !limits_) {return {};}
  if (state_ == TemplateState::kEnterB && now >= trajectory_.start + trajectory_.duration - kTimeEpsilonS) {
    hold_since_ = trajectory_.start + trajectory_.duration;
    transition(TemplateState::kHold, "");
  }
  // Session-level exits. Each trigger acts once; begin_exit keeps the shorter exit.
  if (release_) {
    const auto release = *release_;
    release_.reset();
    begin_exit(release.first, release.second, now);
  }
  if (!keepalive_lapsed_ && now - last_keepalive_ > kKeepaliveTimeoutS) {
    keepalive_lapsed_ = true;
    begin_exit(segment_->t_out_stop, "keepalive_timeout", now);
  }
  if (state_ == TemplateState::kHold && now - hold_since_ > segment_->hold_max_s) {
    begin_exit(segment_->t_out, "hold_max", now);
  }

  Vec point{};
  const bool finished =
    state_ == TemplateState::kExitB && now >= trajectory_.start + trajectory_.duration - kTimeEpsilonS;
  if (state_ == TemplateState::kHold) {
    point = target_point();
  } else if (finished) {
    point = start_point();
  } else {
    Vec v{}, a{};
    evaluate(now, point, v, a);
  }
  if (!std::all_of(point.begin(), point.end(), [](double x) {return std::isfinite(x);})) {
    if (state_ == TemplateState::kExitB || internal_error_) {
      stop("internal_error");
      return {};
    }
    internal_error_ = true;
    begin_exit(segment_->t_out_stop, "internal_error", now);
    Vec v{}, a{};
    evaluate(now, point, v, a);
    if (!std::all_of(point.begin(), point.end(), [](double x) {return std::isfinite(x);})) {
      stop("internal_error");
      return {};
    }
  }

  Output out;
  out.kind = Output::Kind::kUpperBody;
  out.sequence = requested.sequence;
  out.hand_sub_mode = hand_sub_mode_;
  out.head = requested.kind == Output::Kind::kUpperBody && requested.head.size() == 2 &&
    finite_values(requested.head) ? requested.head : start_head_;
  out.arm.assign(point.begin(), point.begin() + 14);
  out.hand.assign(point.begin() + 14, point.end());
  bool clamped = false;
  const double dt = std::min(limits_->period_s, std::max(0.0, now - last_step_));
  for (size_t j = 0; j < 14; ++j) {
    double value = std::clamp(out.arm[j], limits_->lower[j], limits_->upper[j]);
    if (last_arm_) {
      const double step = limits_->velocity[j] * dt;
      value = std::clamp(value, (*last_arm_)[j] - step, (*last_arm_)[j] + step);
    }
    clamped = clamped || value != out.arm[j];
    out.arm[j] = value;
  }
  // Both thumbs, installed or not: an absent hand is an empty flange.
  for (size_t axis = 0; axis < kThumbSlots.size(); ++axis) {
    const auto band = cbf_thumb_command_band(*limits_, axis);
    auto & value = out.hand[kThumbSlots[axis]];
    const double limited = std::clamp(value, band[0], band[1]);
    clamped = clamped || limited != value;
    value = limited;
  }
  if (clamped) {++counters_.clamped;}
  std::array<double, 14> arm{};
  std::copy(out.arm.begin(), out.arm.end(), arm.begin());
  last_arm_ = arm;
  last_step_ = now;
  // Done only on the exact start point; a limited last step plays on.
  if (finished && arm == p_actual_ && std::equal(hand_p_.begin(), hand_p_.end(), out.hand.begin())) {
    ++counters_.done;
    transition(TemplateState::kDone, reason_);
  }
  return out;
}

void TemplatePlayer::stop(const std::string & reason)
{
  if (!playing()) {return;}
  release_.reset();
  ++counters_.stopped;
  transition(TemplateState::kStopped, reason);
}

std::vector<std::string> TemplatePlayer::take_status(double now)
{
  std::vector<std::string> out;
  for (const auto & refusal : outbox_) {
    out.push_back(status_json(now, refusal.seg_id, refusal.segment, TemplateState::kRejected, refusal.reason));
  }
  const bool refused_now = !outbox_.empty();
  outbox_.clear();
  // A refusal that became the current state was just sent; the next periodic one repeats it.
  if (refused_now && state_ == TemplateState::kRejected && changed_) {
    changed_ = false;
    last_status_ = now;
    return out;
  }
  if (changed_ || last_status_ < 0.0 || now - last_status_ >= kStatusPeriodS - kTimeEpsilonS) {
    out.push_back(status_json(now, seg_id_, segment_name_, state_, reason_));
    changed_ = false;
    last_status_ = now;
  }
  return out;
}

void TemplatePlayer::transition(TemplateState state, const std::string & reason)
{
  state_ = state;
  reason_ = reason;
  changed_ = true;
}

void TemplatePlayer::refuse(uint64_t seg_id, const std::string & segment, const std::string & reason)
{
  ++counters_.rejected;
  outbox_.push_back(Refusal{seg_id, segment, reason});
  // Nor does it replace a transition not yet published (a DONE, a STOPPED).
  if (playing() || pending_ || (changed_ && state_ != TemplateState::kRejected)) {return;}
  seg_id_ = seg_id;
  segment_name_ = segment;
  segment_ = nullptr;
  have_start_point_ = false;
  transition(TemplateState::kRejected, reason);
}

void TemplatePlayer::plan(const Vec & q, const Vec & v, const Vec & a, const Vec & goal, double duration, double now)
{
  trajectory_.start = now;
  trajectory_.duration = duration;
  const double t = duration;
  for (size_t i = 0; i < kDims; ++i) {
    const double h = goal[i] - q[i];
    auto & c = trajectory_.c[i];
    c[0] = q[i];
    c[1] = v[i];
    c[2] = a[i] / 2.0;
    c[3] = (20.0 * h - 12.0 * v[i] * t - 3.0 * a[i] * t * t) / (2.0 * t * t * t);
    c[4] = (-30.0 * h + 16.0 * v[i] * t + 3.0 * a[i] * t * t) / (2.0 * t * t * t * t);
    c[5] = (12.0 * h - 6.0 * v[i] * t - a[i] * t * t) / (2.0 * t * t * t * t * t);
  }
}

void TemplatePlayer::evaluate(double now, Vec & q, Vec & v, Vec & a) const
{
  const double s = std::clamp(now - trajectory_.start, 0.0, trajectory_.duration);
  for (size_t i = 0; i < kDims; ++i) {
    const auto & c = trajectory_.c[i];
    q[i] = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))));
    v[i] = c[1] + s * (2.0 * c[2] + s * (3.0 * c[3] + s * (4.0 * c[4] + s * 5.0 * c[5])));
    a[i] = 2.0 * c[2] + s * (6.0 * c[3] + s * (12.0 * c[4] + s * 20.0 * c[5]));
  }
}

void TemplatePlayer::begin_exit(double chosen, const std::string & reason, double now)
{
  Vec q{}, v{}, a{};
  double duration = chosen;
  if (state_ == TemplateState::kHold) {
    q = target_point();
  } else {
    evaluate(now, q, v, a);
    if (state_ == TemplateState::kExitB) {
      const double remaining = trajectory_.start + trajectory_.duration - now;
      if (chosen >= exit_chosen_ - 1e-9 || remaining <= 0.0) {return;}
    }
    // The share of the way still to go back, by the arm joints, else the hand.
    const auto from = start_point();
    const auto to = target_point();
    double fraction = -1.0;
    for (const auto & [first, last] : {std::pair<size_t, size_t>{0, 14}, std::pair<size_t, size_t>{14, kDims}}) {
      for (size_t i = first; i < last; ++i) {
        const double span = std::abs(to[i] - from[i]);
        if (span > 1e-9) {fraction = std::max(fraction, std::abs(q[i] - from[i]) / span);}
      }
      if (fraction >= 0.0) {break;}
    }
    if (fraction < 0.0 || !std::isfinite(fraction)) {fraction = 1.0;}
    duration = std::clamp(chosen * fraction, std::min(kMinExitS, chosen), chosen);
  }
  const auto goal = start_point();
  const Trajectory previous = trajectory_;
  // Lengthen in steps of 10 % until the arm keeps within its speed limit, but
  // never beyond the chosen time; the per-tick limit covers what remains.
  for (;;) {
    plan(q, v, a, goal, duration, now);
    bool fast = false;
    for (int k = 0; k <= 64 && !fast; ++k) {
      Vec sq{}, sv{}, sa{};
      evaluate(now + duration * k / 64.0, sq, sv, sa);
      for (size_t j = 0; j < 14; ++j) {
        if (std::abs(sv[j]) > limits_->velocity[j]) {fast = true;}
      }
    }
    if (!fast || duration >= chosen) {break;}
    duration = std::min(chosen, duration * 1.1);
  }
  if (state_ == TemplateState::kExitB && trajectory_.duration >= previous.start + previous.duration - now) {
    trajectory_ = previous;
    return;
  }
  exit_chosen_ = chosen;
  transition(TemplateState::kExitB, reason);
}

TemplatePlayer::Vec TemplatePlayer::start_point() const
{
  Vec out{};
  std::copy(p_actual_.begin(), p_actual_.end(), out.begin());
  std::copy(hand_p_.begin(), hand_p_.end(), out.begin() + 14);
  return out;
}

TemplatePlayer::Vec TemplatePlayer::target_point() const
{
  Vec out{};
  if (!segment_) {return out;}
  std::copy(segment_->T.begin(), segment_->T.end(), out.begin());
  std::copy(segment_->hand_T.begin(), segment_->hand_T.end(), out.begin() + 14);
  return out;
}

std::string TemplatePlayer::status_json(double now, std::optional<uint64_t> seg_id, const std::string & segment,
  TemplateState state, const std::string & reason) const
{
  json out;
  out["schema"] = "x2_template_status/v1";
  out["seg_id"] = seg_id ? json(*seg_id) : json(nullptr);
  out["seg"] = segment;
  out["state"] = template_state_name(state);
  out["reason"] = reason;
  out["at"] = now;
  if (have_start_point_) {
    out["P_actual"] = p_actual_;
    out["hand_P"] = hand_p_;
  }
  out["counters"] = {
    {"started", counters_.started}, {"done", counters_.done}, {"rejected", counters_.rejected},
    {"stopped", counters_.stopped}, {"malformed", counters_.malformed}, {"mismatched", counters_.mismatched},
    {"clamped", counters_.clamped}};
  json names = json::array();
  for (const auto & entry : whitelist_.segments) {names.push_back(entry.first);}
  out["segments"] = names;
  return out.dump();
}

}  // namespace aimdk_dataplane
