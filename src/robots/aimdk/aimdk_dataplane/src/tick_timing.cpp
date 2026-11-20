// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
#include "aimdk_dataplane/tick_timing.hpp"

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <utility>

namespace aimdk_dataplane
{
namespace
{
using nlohmann::json;

constexpr std::array<const char *, 5> kPathNames{"no_output", "plain", "played", "discarded", "cbf"};

// A configuration-wide prefix, not a reason of its own.
std::string strip(const std::string & reason, const std::string & prefix)
{
  return reason.compare(0, prefix.size(), prefix) == 0 ? reason.substr(prefix.size()) : reason;
}

double since(TickTiming::Clock::time_point begin, TickTiming::Clock::time_point end)
{
  return TickTiming::ms(end - begin);
}

bool set(TickTiming::Clock::time_point at)
{
  return at.time_since_epoch().count() != 0;
}

double cpu_ms()
{
  timespec at{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &at);
  return static_cast<double>(at.tv_sec) * 1e3 + static_cast<double>(at.tv_nsec) * 1e-6;
}

// The scheduler's figures for the calling thread: nanoseconds on a CPU (not
// used: it lags until the next scheduler tick), nanoseconds waiting in the run
// queue, times put on a CPU. The wait is added when the thread gets its CPU,
// so a reading by the thread itself is up to date.
ThreadTimes read_thread_times()
{
  ThreadTimes times;
  times.cpu_ms = cpu_ms();
  times.cpu = true;
  thread_local const int fd = ::open("/proc/thread-self/schedstat", O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return times;
  }
  char text[96];
  const auto size = ::pread(fd, text, sizeof(text) - 1, 0);
  if (size <= 0) {
    return times;
  }
  text[size] = '\0';
  char * end = nullptr;
  std::strtoull(text, &end, 10);
  char * next = nullptr;
  const auto runqueue_ns = std::strtoull(end, &next, 10);
  char * last = nullptr;
  const auto slices = std::strtoull(next, &last, 10);
  if (last == next) {
    return times;
  }
  times.runqueue_ms = static_cast<double>(runqueue_ns) * 1e-6;
  times.slices = slices;
  times.scheduler = true;
  return times;
}
}  // namespace

void TimingHistogram::add(double ms)
{
  const auto bucket = std::upper_bound(kEdgesMs.begin(), kEdgesMs.end(), ms) - kEdgesMs.begin();
  ++counts[static_cast<size_t>(bucket)];
  ++n;
  sum_ms += ms;
  max_ms = std::max(max_ms, ms);
  window_max_ms = std::max(window_max_ms, ms);
}

void TickTrace::decided(const CbfDecision & decision, bool enforce_mode)
{
  enforce = enforce_mode;
  accept_new = decision.accept_new;
  released = decision.released;
  collision_refusal = decision.collision_refusal;
  upper_body = decision.output.kind == Output::Kind::kUpperBody;
}

TickTiming::TickTiming(double period_s)
: period_ms_(period_s * 1e3),
  period_(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period_s)))
{
  // What one reading costs here, to tell the overhead of the readings apart.
  constexpr int kReadings = 1000;
  const auto begin = Clock::now();
  for (int i = 0; i < kReadings; ++i) {
    (void)Clock::now();
  }
  clock_ns_ = std::chrono::duration<double, std::nano>(Clock::now() - begin).count() / kReadings;
  constexpr int kThreadReadings = 200;
  const auto thread_begin = Clock::now();
  for (int i = 0; i < kThreadReadings; ++i) {
    scheduler_ = read_thread_times().scheduler;
  }
  thread_times_ns_ = std::chrono::duration<double, std::nano>(Clock::now() - thread_begin).count() /
    kThreadReadings;
  const auto cpu_begin = Clock::now();
  for (int i = 0; i < kThreadReadings; ++i) {
    (void)cpu_ms();
  }
  thread_cpu_ns_ = std::chrono::duration<double, std::nano>(Clock::now() - cpu_begin).count() / kThreadReadings;
}

TickTiming::Clock::time_point TickTiming::now()
{
  clock_reads_.fetch_add(1, std::memory_order_relaxed);
  return Clock::now();
}

ThreadTimes TickTiming::thread_times()
{
  thread_reads_.fetch_add(1, std::memory_order_relaxed);
  return read_thread_times();
}

double TickTiming::thread_cpu_ms()
{
  thread_reads_.fetch_add(1, std::memory_order_relaxed);
  return cpu_ms();
}

double TickTiming::ms(Clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

void TickTiming::add(TimingSegment segment, double ms)
{
  std::lock_guard<std::mutex> lock(mutex_);
  add_locked(segment, ms);
}

void TickTiming::add_locked(TimingSegment segment, double ms)
{
  histograms_[static_cast<size_t>(segment)].add(std::max(0.0, ms));
}

void TickTiming::published()
{
  std::lock_guard<std::mutex> lock(mutex_);
  ++publishes_;
}

void TickTiming::count(std::map<std::string, uint64_t> & reasons, const std::string & reason)
{
  const auto found = reasons.find(reason);
  if (found != reasons.end()) {
    ++found->second;
  } else if (reasons.size() < kMaxReasons) {
    reasons.emplace(reason, 1);
  } else {
    ++reasons["other"];
  }
}

void TickTiming::tick(const TickTrace & trace, const CbfDiagnostics * diagnostics)
{
  const auto begun = now();
  std::lock_guard<std::mutex> lock(mutex_);
  ++invocations_;
  if (previous_entered_) {
    const double gap = since(*previous_entered_, trace.entered);
    add_locked(TimingSegment::kLate, gap - period_ms_);
    // Missed ticks by rcl's timer rule (rcl_timer_call), not by the gap: a late
    // call serves its own slot and the next slot stays on the phase, so a late
    // tick and the short gap after it miss nothing; slots already past when a
    // call is made are skipped. The phase is the earliest start seen, since a
    // call never comes before its slot. A start right on a slot boundary can be
    // counted one off.
    if (trace.entered < next_due_) {next_due_ = trace.entered;}
    next_due_ += period_;
    if (next_due_ < trace.entered) {
      const auto skipped = static_cast<uint64_t>(std::ceil(since(next_due_, trace.entered) / period_ms_));
      missed_ticks_ += skipped;
      next_due_ += static_cast<Clock::duration::rep>(skipped) * period_;
    }
  } else {
    next_due_ = trace.entered + period_;
  }
  previous_entered_ = trace.entered;
  add_locked(TimingSegment::kTick, since(trace.entered, trace.ended));
  add_locked(TimingSegment::kLockWait, since(trace.entered, trace.locked));
  if (trace.publishes > 0) {add_locked(TimingSegment::kPublish, trace.publish_ms);}
  if (trace.warned) {add_locked(TimingSegment::kWarn, trace.warn_ms);}
  ++paths_[kPathNames[static_cast<size_t>(trace.path)]];
  const bool solved = (trace.path == TickPath::kCbf || trace.path == TickPath::kDiscarded) &&
    set(trace.unlocked) && set(trace.filtered) && set(trace.relocked);
  if (solved) {
    const double snapshot = since(trace.locked, trace.unlocked);
    const double filter = since(trace.unlocked, trace.filtered);
    const double relock = since(trace.filtered, trace.relocked);
    add_locked(TimingSegment::kSnapshot, snapshot);
    add_locked(TimingSegment::kFilter, filter);
    add_locked(TimingSegment::kRelock, relock);
    add_locked(TimingSegment::kOther, since(trace.relocked, trace.ended) - trace.publish_ms - trace.warn_ms);
    if (trace.unlocked_cpu_ms && trace.filtered_cpu_ms) {
      const double filter_cpu = *trace.filtered_cpu_ms - *trace.unlocked_cpu_ms;
      add_locked(TimingSegment::kFilterCpu, filter_cpu);
      add_locked(TimingSegment::kFilterOffCpu, filter - filter_cpu);
    }
  }
  const auto & entered = trace.at_entered;
  const auto & ended = trace.at_ended;
  if (entered.cpu && ended.cpu) {
    add_locked(TimingSegment::kTickCpu, ended.cpu_ms - entered.cpu_ms);
  }
  if (entered.scheduler && ended.scheduler) {
    add_locked(TimingSegment::kTickRunqueue, ended.runqueue_ms - entered.runqueue_ms);
    // Put on a CPU again during the tick: it lost its CPU at least once.
    if (ended.slices > entered.slices) {++descheduled_ticks_;}
    if (previous_ended_ && previous_ended_->scheduler) {
      add_locked(TimingSegment::kBetweenRunqueue, entered.runqueue_ms - previous_ended_->runqueue_ms);
    }
  }
  previous_ended_ = ended;
  if (tick_tid_ == 0) {tick_tid_ = static_cast<int>(::syscall(SYS_gettid));}
  if (trace.path == TickPath::kCbf && diagnostics) {
    const auto reason = strip(diagnostics->reason, "unverified_mapping/");
    if (!trace.enforce) {
      ++outcomes_["shadow"];
    } else if (trace.released) {
      ++outcomes_["released"];
      if (diagnostics->stuck_collision) {++outcomes_["released_collision"];}
      count(released_reasons_, strip(strip(diagnostics->reason, "released/"), "unverified_mapping/"));
    } else if (trace.accept_new) {
      ++outcomes_["verified"];
    } else {
      ++outcomes_[trace.upper_body ? "held" : "withheld"];
      if (trace.upper_body && trace.collision_refusal) {++outcomes_["held_collision"];}
      count(held_reasons_, reason);
    }
  }
  add_locked(TimingSegment::kBookkeeping, since(begun, Clock::now()));
  clock_reads_.fetch_add(1, std::memory_order_relaxed);
}

uint64_t TickTiming::missed_ticks() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return missed_ticks_;
}

json TickTiming::report(double now_s)
{
  std::lock_guard<std::mutex> lock(mutex_);
  json out = {
    {"period_ms", period_ms_}, {"invocations", invocations_}, {"missed_ticks", missed_ticks_},
    {"descheduled_ticks", descheduled_ticks_},
    {"publishes", publishes_}, {"paths", paths_}, {"outcomes", outcomes_},
    {"held_reasons", held_reasons_}, {"released_reasons", released_reasons_},
    {"clock_ns", clock_ns_}, {"clock_reads", clock_reads_.load(std::memory_order_relaxed)},
    {"scheduler", scheduler_}, {"thread_times_ns", thread_times_ns_}, {"thread_cpu_ns", thread_cpu_ns_},
    {"thread_reads", thread_reads_.load(std::memory_order_relaxed)},
  };
  if (tick_tid_ != 0) {
    // The command thread's nice, to see a renice took.
    errno = 0;
    const int nice = ::getpriority(PRIO_PROCESS, static_cast<id_t>(tick_tid_));
    out["tick_tid"] = tick_tid_;
    out["tick_nice"] = errno == 0 ? json(nice) : json(nullptr);
  }
  json window = json::object();
  for (size_t i = 0; i < histograms_.size(); ++i) {
    window[kTimingSegmentNames[i]] = std::exchange(histograms_[i].window_max_ms, 0.0);
  }
  out["window_max_ms"] = std::move(window);
  if (!histograms_reported_s_ || now_s - *histograms_reported_s_ >= kHistogramPeriodS) {
    json histograms = {{"edges_ms", TimingHistogram::kEdgesMs}};
    for (size_t i = 0; i < histograms_.size(); ++i) {
      const auto & h = histograms_[i];
      histograms[kTimingSegmentNames[i]] = {
        {"n", h.n}, {"sum_ms", h.sum_ms}, {"max_ms", h.max_ms}, {"counts", h.counts}};
    }
    out["histograms"] = std::move(histograms);
    histograms_reported_s_ = now_s;
  }
  return out;
}

}  // namespace aimdk_dataplane
