// Copyright 2026 IB-Robot contributors
// SPDX-License-Identifier: Apache-2.0
//
// The tick timing figures, without ROS: buckets, late and missed ticks, the
// segments of a solved tick, its time on a CPU and in the run queue, the outcome
// and reason of each CBF frame, the report.

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

#include "aimdk_dataplane/cbf_safety.hpp"
#include "aimdk_dataplane/tick_timing.hpp"
#include "nlohmann/json.hpp"

namespace
{

using aimdk_dataplane::CbfDecision;
using aimdk_dataplane::CbfDiagnostics;
using aimdk_dataplane::Output;
using aimdk_dataplane::TickPath;
using aimdk_dataplane::TickTiming;
using aimdk_dataplane::TickTrace;
using aimdk_dataplane::TimingHistogram;
using aimdk_dataplane::TimingSegment;
using Clock = std::chrono::steady_clock;

Clock::time_point at_ms(double ms)
{
  // Far from the clock's epoch: an unset reading is the epoch itself.
  return Clock::time_point{} + std::chrono::hours(1) +
         std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
}

TickTrace plain(double entered_ms, double tick_ms = 1.0)
{
  TickTrace trace;
  trace.path = TickPath::kPlain;
  trace.entered = at_ms(entered_ms);
  trace.locked = at_ms(entered_ms);
  trace.ended = at_ms(entered_ms + tick_ms);
  return trace;
}

TickTrace cbf(double entered_ms, bool accept_new, bool released, bool collision, bool upper_body = true)
{
  auto trace = plain(entered_ms, 8.0);
  trace.path = TickPath::kCbf;
  trace.locked = at_ms(entered_ms + 0.5);
  trace.unlocked = at_ms(entered_ms + 1.0);
  trace.filtered = at_ms(entered_ms + 6.0);
  trace.relocked = at_ms(entered_ms + 6.5);
  trace.publish_ms = 0.25;
  trace.publishes = 1;
  CbfDecision decision;
  decision.accept_new = accept_new;
  decision.released = released;
  decision.collision_refusal = collision;
  decision.output.kind = upper_body ? Output::Kind::kUpperBody : Output::Kind::kNone;
  trace.decided(decision, true);
  return trace;
}

const nlohmann::json & histogram(const nlohmann::json & report, const char * name)
{
  return report.at("histograms").at(name);
}

TEST(TimingHistogram, BucketsAreHalfOpenWithAnOpenTop)
{
  TimingHistogram h;
  for (double ms : {0.0, 0.49, 0.5, 19.99, 20.0, 99.9, 100.0, 1e6}) {h.add(ms);}
  EXPECT_EQ(h.counts[0], 2u);  // < 0.5
  EXPECT_EQ(h.counts[1], 1u);  // [0.5, 1)
  EXPECT_EQ(h.counts[8], 1u);  // [15, 20)
  EXPECT_EQ(h.counts[9], 1u);  // [20, 30)
  EXPECT_EQ(h.counts[11], 1u);  // [50, 100)
  EXPECT_EQ(h.counts[12], 2u);  // >= 100
  EXPECT_EQ(h.n, 8u);
  EXPECT_DOUBLE_EQ(h.max_ms, 1e6);
}

TEST(TickTiming, LateAndMissedTicksComeFromTheGapBetweenStarts)
{
  TickTiming timing(0.02);
  // 20, 21 (1 ms late), 41 (one missed tick), 70 (three missed), 19 (early).
  double t = 0.0;
  for (double gap : {0.0, 20.0, 21.0, 41.0, 70.0, 19.0}) {
    t += gap;
    timing.tick(plain(t), nullptr);
  }
  EXPECT_EQ(timing.missed_ticks(), 1u + 3u);
  const auto report = timing.report(0.0);
  EXPECT_EQ(report.at("invocations"), 6);
  const auto & late = histogram(report, "late");
  EXPECT_EQ(late.at("n"), 5);  // none for the first tick
  EXPECT_NEAR(late.at("max_ms").get<double>(), 50.0, 1e-6);
  EXPECT_NEAR(late.at("sum_ms").get<double>(), 0.0 + 1.0 + 21.0 + 50.0 + 0.0, 1e-6);
  EXPECT_EQ(late.at("counts").at(0), 2);  // on time, and early after a late one
  EXPECT_EQ(report.at("paths").at("plain"), 6);
  // A plain tick has no solve segments.
  EXPECT_EQ(histogram(report, "filter").at("n"), 0);
  EXPECT_EQ(histogram(report, "tick").at("n"), 6);
}

TEST(TickTiming, ASolvedTickIsSplitIntoItsSegments)
{
  TickTiming timing(0.02);
  auto trace = cbf(0.0, true, false, false);
  trace.warn_ms = 0.75;
  trace.warned = true;
  CbfDiagnostics diagnostics;
  diagnostics.reason = "verified_nominal_path";
  timing.tick(trace, &diagnostics);
  const auto report = timing.report(0.0);
  EXPECT_NEAR(histogram(report, "lock_wait").at("sum_ms").get<double>(), 0.5, 1e-6);
  EXPECT_NEAR(histogram(report, "snapshot").at("sum_ms").get<double>(), 0.5, 1e-6);
  EXPECT_NEAR(histogram(report, "filter").at("sum_ms").get<double>(), 5.0, 1e-6);
  EXPECT_NEAR(histogram(report, "relock").at("sum_ms").get<double>(), 0.5, 1e-6);
  EXPECT_NEAR(histogram(report, "publish").at("sum_ms").get<double>(), 0.25, 1e-6);
  EXPECT_NEAR(histogram(report, "warn").at("sum_ms").get<double>(), 0.75, 1e-6);
  // 8 ms in all: the rest after the lock came back, less publish and the WARN.
  EXPECT_NEAR(histogram(report, "other").at("sum_ms").get<double>(), 1.5 - 0.25 - 0.75, 1e-6);
  EXPECT_NEAR(histogram(report, "tick").at("sum_ms").get<double>(), 8.0, 1e-6);
  EXPECT_EQ(histogram(report, "bookkeeping").at("n"), 1);
  EXPECT_EQ(report.at("outcomes").at("verified"), 1);
  EXPECT_TRUE(report.at("held_reasons").empty());
}

TEST(TickTiming, EachFrameIsCountedByOutcomeAndHoldsByReason)
{
  TickTiming timing(0.02);
  CbfDiagnostics diagnostics;
  diagnostics.reason = "unverified_mapping/node_deadline";
  timing.tick(cbf(0.0, false, false, false), &diagnostics);
  diagnostics.reason = "node_deadline";
  timing.tick(cbf(20.0, false, false, false), &diagnostics);
  diagnostics.reason = "infeasible_fallback";
  timing.tick(cbf(40.0, false, false, true), &diagnostics);
  diagnostics.reason = "released/unverified_mapping/publication_feedback_unavailable";
  diagnostics.stuck_collision = false;
  timing.tick(cbf(60.0, true, true, false), &diagnostics);
  diagnostics.reason = "released/infeasible_fallback";
  diagnostics.stuck_collision = true;
  timing.tick(cbf(80.0, true, true, false), &diagnostics);
  diagnostics.reason = "awaiting_mirror_input";
  timing.tick(cbf(100.0, false, false, false, false), &diagnostics);
  // A dropped frame is a path, not an outcome.
  auto discarded = cbf(120.0, false, false, false);
  discarded.path = TickPath::kDiscarded;
  diagnostics.reason = "stale_snapshot_or_epoch";
  timing.tick(discarded, &diagnostics);

  const auto report = timing.report(0.0);
  const auto & outcomes = report.at("outcomes");
  EXPECT_EQ(outcomes.at("held"), 3);
  EXPECT_EQ(outcomes.at("held_collision"), 1);
  EXPECT_EQ(outcomes.at("released"), 2);
  EXPECT_EQ(outcomes.at("released_collision"), 1);
  EXPECT_EQ(outcomes.at("withheld"), 1);
  EXPECT_FALSE(outcomes.contains("verified"));
  EXPECT_EQ(report.at("held_reasons"), (nlohmann::json{
      {"node_deadline", 2}, {"infeasible_fallback", 1}, {"awaiting_mirror_input", 1}}));
  EXPECT_EQ(report.at("released_reasons"), (nlohmann::json{
      {"publication_feedback_unavailable", 1}, {"infeasible_fallback", 1}}));
  EXPECT_EQ(report.at("paths").at("cbf"), 6);
  EXPECT_EQ(report.at("paths").at("discarded"), 1);
  // The discarded tick still solved: its segments count.
  EXPECT_EQ(histogram(report, "filter").at("n"), 7);
}

TEST(TickTiming, ShadowFramesAreNotHolds)
{
  TickTiming timing(0.02);
  auto trace = cbf(0.0, false, false, true);
  trace.enforce = false;
  CbfDiagnostics diagnostics;
  diagnostics.reason = "infeasible_fallback";
  timing.tick(trace, &diagnostics);
  const auto report = timing.report(0.0);
  EXPECT_EQ(report.at("outcomes"), (nlohmann::json{{"shadow", 1}}));
  EXPECT_TRUE(report.at("held_reasons").empty());
}

TEST(TickTiming, ReasonsAreBounded)
{
  TickTiming timing(0.02);
  CbfDiagnostics diagnostics;
  for (size_t i = 0; i < TickTiming::kMaxReasons + 5; ++i) {
    diagnostics.reason = "reason_" + std::to_string(i);
    timing.tick(cbf(20.0 * static_cast<double>(i), false, false, false), &diagnostics);
  }
  const auto reasons = timing.report(0.0).at("held_reasons");
  EXPECT_EQ(reasons.size(), TickTiming::kMaxReasons + 1);
  EXPECT_EQ(reasons.at("other"), 5);
}

TEST(TickTiming, HistogramsOnceASecondMaximaPerReport)
{
  TickTiming timing(0.02);
  timing.add(TimingSegment::kStatus, 3.0);
  timing.published();
  auto report = timing.report(10.0);
  EXPECT_TRUE(report.contains("histograms"));
  EXPECT_EQ(report.at("histograms").at("edges_ms").size(), TimingHistogram::kEdgesMs.size());
  EXPECT_DOUBLE_EQ(report.at("window_max_ms").at("status").get<double>(), 3.0);
  EXPECT_EQ(report.at("publishes"), 1);

  timing.add(TimingSegment::kStatus, 1.0);
  report = timing.report(10.5);
  EXPECT_FALSE(report.contains("histograms"));
  EXPECT_DOUBLE_EQ(report.at("window_max_ms").at("status").get<double>(), 1.0);
  report = timing.report(10.6);
  EXPECT_DOUBLE_EQ(report.at("window_max_ms").at("status").get<double>(), 0.0);

  report = timing.report(11.0);
  ASSERT_TRUE(report.contains("histograms"));
  const auto & status = histogram(report, "status");
  EXPECT_EQ(status.at("n"), 2);
  EXPECT_DOUBLE_EQ(status.at("max_ms").get<double>(), 3.0);
}

TEST(TickTiming, ClockReadingsAreCountedAndPriced)
{
  TickTiming timing(0.02);
  (void)timing.now();
  (void)timing.now();
  (void)timing.thread_times();
  const auto report = timing.report(0.0);
  EXPECT_EQ(report.at("clock_reads"), 2);
  EXPECT_EQ(report.at("thread_reads"), 1);
  EXPECT_GT(report.at("clock_ns").get<double>(), 0.0);
  EXPECT_GT(report.at("thread_times_ns").get<double>(), 0.0);
  EXPECT_GT(report.at("thread_cpu_ns").get<double>(), 0.0);
  EXPECT_FALSE(report.contains("tick_tid"));  // no tick yet
}

aimdk_dataplane::ThreadTimes times(double cpu_ms, double runqueue_ms, uint64_t slices)
{
  aimdk_dataplane::ThreadTimes out;
  out.cpu_ms = cpu_ms;
  out.runqueue_ms = runqueue_ms;
  out.slices = slices;
  out.cpu = out.scheduler = true;
  return out;
}

TEST(TickTiming, ATickIsSplitIntoComputingAndWaitingForACpu)
{
  TickTiming timing(0.02);
  auto first = cbf(0.0, true, false, false);
  first.at_entered = times(100.0, 50.0, 10);
  first.at_ended = times(102.0, 50.0, 10);  // on a CPU throughout
  first.unlocked_cpu_ms = 100.5;
  first.filtered_cpu_ms = 101.5;  // 1 ms of the filter's 5
  timing.tick(first, nullptr);
  auto second = cbf(20.0, true, false, false);
  second.at_entered = times(103.0, 53.0, 12);  // 3 ms queued since the last tick
  second.at_ended = times(105.0, 57.0, 14);  // 4 ms queued during it
  timing.tick(second, nullptr);
  auto unread = plain(40.0);  // no scheduler figures: skipped, not zero
  unread.at_entered.cpu = unread.at_ended.cpu = true;
  timing.tick(unread, nullptr);

  const auto report = timing.report(0.0);
  EXPECT_EQ(report.at("descheduled_ticks"), 1);
  EXPECT_EQ(histogram(report, "tick_cpu").at("n"), 3);
  EXPECT_NEAR(histogram(report, "tick_cpu").at("sum_ms").get<double>(), 4.0, 1e-9);
  EXPECT_EQ(histogram(report, "tick_runqueue").at("n"), 2);
  EXPECT_NEAR(histogram(report, "tick_runqueue").at("sum_ms").get<double>(), 4.0, 1e-9);
  // Between the second and the unread one: no figures to subtract.
  EXPECT_EQ(histogram(report, "between_runqueue").at("n"), 1);
  EXPECT_NEAR(histogram(report, "between_runqueue").at("sum_ms").get<double>(), 3.0, 1e-9);
  EXPECT_EQ(histogram(report, "filter_cpu").at("n"), 1);
  EXPECT_NEAR(histogram(report, "filter_cpu").at("sum_ms").get<double>(), 1.0, 1e-9);
  EXPECT_NEAR(histogram(report, "filter_offcpu").at("sum_ms").get<double>(), 4.0, 1e-6);
  EXPECT_TRUE(report.contains("tick_tid"));
  EXPECT_TRUE(report.at("tick_nice").is_number_integer());
}

TEST(TickTiming, TheThreadClocksTellSpinningFromSleeping)
{
  TickTiming timing(0.02);
  const auto spin = [](std::chrono::milliseconds length) {
      const auto until = std::chrono::steady_clock::now() + length;
      while (std::chrono::steady_clock::now() < until) {}
    };
  const auto before = timing.thread_times();
  spin(std::chrono::milliseconds(30));
  const auto spun = timing.thread_times();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  const auto slept = timing.thread_times();
  ASSERT_TRUE(before.cpu && spun.cpu && slept.cpu);
  EXPECT_GT(spun.cpu_ms - before.cpu_ms, 15.0);
  EXPECT_LT(slept.cpu_ms - spun.cpu_ms, 5.0);
  if (!before.scheduler) {GTEST_SKIP() << "no /proc/thread-self/schedstat here";}
  EXPECT_GE(slept.runqueue_ms, before.runqueue_ms);
  // Woken from the sleep, it was put on a CPU again.
  EXPECT_GT(slept.slices, spun.slices);
}

}  // namespace
