// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// Where a command tick's time goes, for the status topic: observation only,
// nothing here changes what is published. Every tick and every callback on the
// executor thread is added to bounded cumulative histograms, so a 20 Hz status
// still accounts for every 50 Hz tick; the histograms themselves are reported
// once a second, the counters and the maxima since the previous report in every
// status. Timer invocations, CBF frames and published commands are counted
// apart: fewer CBF calls alone does not show the timer did not run. The
// thread's own clocks split a slow tick into computing and waiting for a CPU.

#ifndef AIMDK_DATAPLANE__TICK_TIMING_HPP_
#define AIMDK_DATAPLANE__TICK_TIMING_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "nlohmann/json.hpp"

namespace aimdk_dataplane
{

enum class TimingSegment : size_t
{
  kLate,  // a command tick's start after the previous one, less the period; never negative
  kTick,  // a command tick, its wait for the lock included
  kLockWait,  // its wait for the lock
  kSnapshot,  // CBF ticks: lock taken to the filter's start
  kFilter,  // CBF ticks: the filter, outside the lock
  kRelock,  // CBF ticks: the wait to take the lock back
  kPublish,  // the time inside publish(), on ticks that published
  kWarn,  // queueing the CBF_RELEASE log line, on ticks that logged one; printed in kLogs
  kOther,  // CBF ticks: the rest (checks, release, commit, notes)
  // Of a command tick, from the thread's own clocks: on a CPU, and ready to run
  // but waiting for one. The rest of its time it slept (the lock).
  kTickCpu,
  kTickRunqueue,
  kBetweenRunqueue,  // waiting for a CPU between two command ticks: makes a tick late
  kFilterCpu,  // CBF ticks: the filter on a CPU
  kFilterOffCpu,  // CBF ticks: the filter not on a CPU (it does not sleep: preempted)
  kTemplateStatus,
  kStatus,  // a status, its wait for the lock included
  kStatusLockWait,
  kTimingJson,  // building this report
  kJointStates,
  kControl,  // a control word; it publishes a status, timed in kStatus as well
  kTouch,
  kEcho,
  kStream,  // a public command channel message
  kLogs,  // printing queued log lines, on the node's log thread
  kBookkeeping,  // adding one tick to these figures
  kCount
};

constexpr std::array<const char *, static_cast<size_t>(TimingSegment::kCount)> kTimingSegmentNames{
  "late", "tick", "lock_wait", "snapshot", "filter", "relock", "publish", "warn", "other",
  "tick_cpu", "tick_runqueue", "between_runqueue", "filter_cpu", "filter_offcpu",
  "template_status", "status", "status_lock_wait", "timing_json", "joint_states", "control", "touch",
  "echo", "stream", "logs", "bookkeeping"};

struct TimingHistogram
{
  // counts[i] holds edges[i - 1] <= ms < edges[i]; the last bucket has no upper edge.
  static constexpr std::array<double, 12> kEdgesMs{0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0};
  std::array<uint64_t, kEdgesMs.size() + 1> counts{};
  uint64_t n = 0;
  double sum_ms = 0.0;
  double max_ms = 0.0;
  double window_max_ms = 0.0;  // since the previous report

  void add(double ms);
};

enum class TickPath
{
  kNoOutput,  // published nothing (no upper-body candidate under the CBF)
  kPlain,  // the path without the CBF
  kPlayed,  // a template frame
  kDiscarded,  // the CBF's frame was dropped: the epoch or context changed during the solve
  kCbf,
};

// The calling thread's clocks since it started: its CPU time, and from the
// scheduler its wait in the run queue and how many times it was put on a CPU.
struct ThreadTimes
{
  double cpu_ms = 0.0;
  double runqueue_ms = 0.0;
  uint64_t slices = 0;
  bool cpu = false;  // cpu_ms was read
  bool scheduler = false;  // runqueue_ms and slices were read
};

// One command tick, filled in as it runs. Clock readings left unset are skipped.
struct TickTrace
{
  using Clock = std::chrono::steady_clock;
  Clock::time_point entered, locked, unlocked, filtered, relocked, ended;
  ThreadTimes at_entered, at_ended;
  std::optional<double> unlocked_cpu_ms, filtered_cpu_ms;
  double publish_ms = 0.0;
  int publishes = 0;
  double warn_ms = 0.0;
  bool warned = false;
  TickPath path = TickPath::kNoOutput;
  // The published CBF decision, on kCbf ticks.
  bool enforce = false;
  bool accept_new = false;
  bool released = false;
  bool collision_refusal = false;
  bool upper_body = false;

  void decided(const CbfDecision & decision, bool enforce_mode);
};

class TickTiming
{
public:
  using Clock = std::chrono::steady_clock;
  static constexpr double kHistogramPeriodS = 1.0;
  // Different reasons are a handful of fixed strings; anything past this is "other".
  static constexpr size_t kMaxReasons = 48;

  explicit TickTiming(double period_s);

  // Every clock reading taken for these figures goes through here, to be counted.
  Clock::time_point now();
  ThreadTimes thread_times();
  double thread_cpu_ms();
  static double ms(Clock::duration duration);

  void add(TimingSegment segment, double ms);
  void published();
  // A finished command tick, on the thread that ran it. ``diagnostics`` is the
  // CBF's, on kCbf ticks, for the reason of a held or released frame.
  void tick(const TickTrace & trace, const CbfDiagnostics * diagnostics);
  // Timer slots that got no call, by rcl's skip rule (see tick()).
  uint64_t missed_ticks() const;
  // The report for the status at ``now_s`` (seconds, monotonic); starts the
  // next window of maxima. Histograms are included at most once a kHistogramPeriodS.
  nlohmann::json report(double now_s);

private:
  void add_locked(TimingSegment segment, double ms);
  static void count(std::map<std::string, uint64_t> & reasons, const std::string & reason);

  mutable std::mutex mutex_;
  double period_ms_;
  Clock::duration period_;
  double clock_ns_ = 0.0;
  double thread_times_ns_ = 0.0;
  double thread_cpu_ns_ = 0.0;
  bool scheduler_ = false;
  std::atomic<uint64_t> clock_reads_{0};
  std::atomic<uint64_t> thread_reads_{0};
  std::array<TimingHistogram, static_cast<size_t>(TimingSegment::kCount)> histograms_{};
  std::optional<Clock::time_point> previous_entered_;
  Clock::time_point next_due_{};  // the slot after the one the last tick served
  std::optional<ThreadTimes> previous_ended_;
  int tick_tid_ = 0;
  uint64_t invocations_ = 0;
  uint64_t missed_ticks_ = 0;
  uint64_t descheduled_ticks_ = 0;
  uint64_t publishes_ = 0;
  std::map<std::string, uint64_t> paths_;
  std::map<std::string, uint64_t> outcomes_;
  std::map<std::string, uint64_t> held_reasons_;
  std::map<std::string, uint64_t> released_reasons_;
  std::optional<double> histograms_reported_s_;
};

}  // namespace aimdk_dataplane
#endif  // AIMDK_DATAPLANE__TICK_TIMING_HPP_
