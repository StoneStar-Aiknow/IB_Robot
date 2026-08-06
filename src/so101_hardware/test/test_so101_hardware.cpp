#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <unistd.h>
#include <utility>
#include <vector>

#include "so101_hardware/so101_system_hardware.hpp"

namespace
{
namespace detail = so101_hardware::detail;

class TestableSafeSMSSTS : public so101_hardware::SafeSMSSTS
{
public:
  int read_for_test(unsigned char * data, int length, unsigned long timeout_ms)
  {
    return readSCS(data, length, timeout_ms);
  }

  void set_fd_for_test(int value) {fd = value;}
};
} // namespace

TEST(
  SO101Hardware,
  SerialReadRejectsInvalidDescriptorWithoutWritingBeforeBuffer) {
  TestableSafeSMSSTS servo;
  unsigned char buffer[2] = {0xAA, 0xBB};
  servo.set_fd_for_test(-1);

  EXPECT_EQ(servo.read_for_test(buffer, 1, 1), -1);
  EXPECT_EQ(buffer[0], 0xAA);
  EXPECT_EQ(buffer[1], 0xBB);
}

TEST(SO101Hardware, SerialReadHandlesClosedPeerWithoutNegativeLength) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  close(descriptors[1]);
  TestableSafeSMSSTS servo;
  servo.set_fd_for_test(descriptors[0]);
  unsigned char buffer[2] = {0xAA, 0xBB};

  EXPECT_EQ(servo.read_for_test(buffer, 1, 10), 0);
  EXPECT_EQ(buffer[0], 0xAA);
  EXPECT_EQ(buffer[1], 0xBB);
  close(descriptors[0]);
  servo.set_fd_for_test(-1);
}

TEST(SO101Hardware, ActivationAbortDisablesEnabledMotorsInReverseOrder) {
  std::vector<std::pair<u8, u8>> calls;

  const bool disabled = so101_hardware::detail::disable_torque_on_abort(
    {1, 2}, [&calls](u8 id, u8 enabled) {
      calls.emplace_back(id, enabled);
      return 1;
    });

  EXPECT_TRUE(disabled);
  ASSERT_EQ(calls.size(), 2U);
  EXPECT_EQ(calls[0], std::make_pair(static_cast<u8>(2), static_cast<u8>(0)));
  EXPECT_EQ(calls[1], std::make_pair(static_cast<u8>(1), static_cast<u8>(0)));
}

TEST(SO101Hardware, ActivationAbortRetriesAndReportsTorqueDisableFailure) {
  int attempts = 0;

  const bool disabled =
    so101_hardware::detail::disable_torque_on_abort(
    {1}, [&attempts](u8, u8) {
      ++attempts;
      return 0;
    });

  EXPECT_FALSE(disabled);
  EXPECT_EQ(attempts, 3);
}

TEST(SO101Hardware, ActivationRollbackRelocksOnlyUnlockedMotorsBeforeClose) {
  // motor_ids {1, 2, 3}; only motor 2 was left unlocked at abort time.
  std::vector<std::pair<std::string, u8>> calls;

  const auto result = so101_hardware::detail::rollback_activation(
    {1, 2, 3}, std::set<u8>{2},
    [&calls](u8 id, u8 enabled) {
      calls.emplace_back("torque", id);
      EXPECT_EQ(enabled, 0);
      return 1;   // success
    },
    [&calls](u8 id) {
      calls.emplace_back("relock", id);
      return 1;   // success
    });

  EXPECT_TRUE(result.torque_disabled_all);
  EXPECT_TRUE(result.eprom_relocked_all);
  EXPECT_TRUE(result.relock_failures.empty());

  // Torque disable runs for every motor in reverse order first.
  ASSERT_EQ(calls.size(), 4U);
  EXPECT_EQ(
    calls[0],
    std::make_pair(std::string("torque"), static_cast<u8>(3)));
  EXPECT_EQ(
    calls[1],
    std::make_pair(std::string("torque"), static_cast<u8>(2)));
  EXPECT_EQ(
    calls[2],
    std::make_pair(std::string("torque"), static_cast<u8>(1)));
  // EPROM relock is only attempted for the unlocked motor, after torque-off.
  EXPECT_EQ(
    calls[3],
    std::make_pair(std::string("relock"), static_cast<u8>(2)));
}

TEST(
  SO101Hardware,
  ActivationRollbackReportsRelockFailureDistinctlyFromTorque) {
  // Both motors unlocked; torque disable succeeds, but motor 1 relock fails.
  const auto result = so101_hardware::detail::rollback_activation(
    {1, 2}, std::set<u8>{1, 2},
    [](u8, u8) {return 1;},                   // torque always succeeds
    [](u8 id) {return id == 2 ? 1 : 0;});     // relock fails for motor 1

  EXPECT_TRUE(result.torque_disabled_all);
  EXPECT_FALSE(result.eprom_relocked_all);
  ASSERT_EQ(result.relock_failures.size(), 1U);
  EXPECT_EQ(result.relock_failures[0], static_cast<u8>(1));
}

TEST(SO101Hardware, ActivationRollbackStaysFailClosedWhenBothOpsFail) {
  // A torque-disable failure must not skip the EPROM relock attempt: both are
  // best-effort and reported independently (fail-closed semantics).
  int lock_attempts = 0;

  const auto result = so101_hardware::detail::rollback_activation(
    {1}, std::set<u8>{1},
    [](u8, u8) {return 0;},     // torque never acknowledges
    [&lock_attempts](u8) {
      ++lock_attempts;
      return 0;   // relock never acknowledges
    });

  EXPECT_FALSE(result.torque_disabled_all);
  EXPECT_FALSE(result.eprom_relocked_all);
  ASSERT_EQ(result.relock_failures.size(), 1U);
  EXPECT_EQ(result.relock_failures[0], static_cast<u8>(1));
  // Relock was still attempted (3 retries) despite torque failing.
  EXPECT_EQ(lock_attempts, 3);
}

// ===== Initial sync feedback fail-closed tests =====
// These exercise detail::perform_initial_sync_feedback with injected Tx/Rx
// callbacks, so no real Feetech bus is required.

namespace
{
// Mirror of the plugin's ticks<->radian constant. Computed from a literal PI
// so the test does not depend on M_PI being defined.
constexpr double kPi = 3.14159265358979323846;
constexpr double kTicksPerRad = 4096.0 / (2.0 * kPi);
constexpr double kCurrentRawToAmpere = 0.0065;

// Sentinel proving a state slot was never written by the helper.
constexpr double kUntouched = -1234.5;
} // namespace

TEST(SO101Hardware, InitialSyncFeedbackAbortsWhenTransmitFails)
{
  std::vector<double> commands(2, kUntouched);
  std::vector<double> positions(2, kUntouched);
  std::vector<double> velocities(2, kUntouched);
  std::vector<double> currents(2, kUntouched);

  int rx_calls = 0;
  const auto outcome = detail::perform_initial_sync_feedback(
    {1, 2}, false, {0.0, 0.0}, kTicksPerRad, kCurrentRawToAmpere,
    []() {return 0;},   // syncReadPacketTx returned <= 0: bus reply failure
    [&rx_calls](u8, detail::FeedbackSample &) {
      ++rx_calls;
      return true;
    },
    commands, positions, velocities, currents);

  EXPECT_FALSE(outcome.success);
  EXPECT_TRUE(outcome.tx_failed);
  // Tx failure must short-circuit before any per-motor Rx is attempted.
  EXPECT_EQ(rx_calls, 0);
  // State vectors must be left untouched so activation rolls back cleanly.
  EXPECT_EQ(positions[0], kUntouched);
  EXPECT_EQ(commands[1], kUntouched);
}

TEST(SO101Hardware, InitialSyncFeedbackAbortsWhenOneMotorRxFails)
{
  std::vector<double> commands(2, kUntouched);
  std::vector<double> positions(2, kUntouched);
  std::vector<double> velocities(2, kUntouched);
  std::vector<double> currents(2, kUntouched);

  int rx_calls = 0;
  const auto outcome = detail::perform_initial_sync_feedback(
    {1, 2}, false, {0.0, 0.0}, kTicksPerRad, kCurrentRawToAmpere,
    []() {return 64;},   // Tx ok
    [&rx_calls](u8 id, detail::FeedbackSample & out) {
      ++rx_calls;
      if (id == 2) {
        return false;   // motor 2 fails to return a full/CRC-valid packet
      }
      out = detail::FeedbackSample{2048, 0, 0};
      return true;
    },
    commands, positions, velocities, currents);

  EXPECT_FALSE(outcome.success);
  EXPECT_FALSE(outcome.tx_failed);
  EXPECT_EQ(outcome.failed_motor_id, static_cast<u8>(2));
  // Motor 1 was queried before the fail-fast stop at motor 2.
  EXPECT_EQ(rx_calls, 2);
}

TEST(SO101Hardware, InitialSyncFeedbackSeedsStateWhenAllMotorsReply)
{
  std::vector<double> commands(2, kUntouched);
  std::vector<double> positions(2, kUntouched);
  std::vector<double> velocities(2, kUntouched);
  std::vector<double> currents(2, kUntouched);

  const auto outcome = detail::perform_initial_sync_feedback(
    {1, 2}, false, {0.0, 0.0}, kTicksPerRad, kCurrentRawToAmpere,
    []() {return 64;},
    [](u8 id, detail::FeedbackSample & out) {
      out = detail::FeedbackSample{id == 1 ? 2048 : 3072, 4096, 200};
      return true;
    },
    commands, positions, velocities, currents);

  ASSERT_TRUE(outcome.success);
  // position = (raw - 2048) / ticks
  EXPECT_NEAR(positions[0], 0.0, 1e-9);
  EXPECT_NEAR(positions[1], (3072.0 - 2048.0) / kTicksPerRad, 1e-9);
  // velocity = speed / ticks
  EXPECT_NEAR(velocities[0], 4096.0 / kTicksPerRad, 1e-9);
  // current = raw * 6.5mA
  EXPECT_NEAR(currents[0], 200.0 * kCurrentRawToAmpere, 1e-9);
  // Without configured reset positions, commands must mirror measured pose.
  EXPECT_NEAR(commands[0], positions[0], 1e-9);
  EXPECT_NEAR(commands[1], positions[1], 1e-9);
}

TEST(SO101Hardware, InitialSyncFeedbackUsesResetPositionsWhenConfigured)
{
  std::vector<double> commands(2, kUntouched);
  std::vector<double> positions(2, kUntouched);
  std::vector<double> velocities(2, kUntouched);
  std::vector<double> currents(2, kUntouched);

  const auto outcome = detail::perform_initial_sync_feedback(
    {1, 2}, true, {0.5, -0.25}, kTicksPerRad, kCurrentRawToAmpere,
    []() {return 64;},
    [](u8, detail::FeedbackSample & out) {
      out = detail::FeedbackSample{2048, 0, 0};
      return true;
    },
    commands, positions, velocities, currents);

  ASSERT_TRUE(outcome.success);
  // Commands come from reset_positions; state still reflects live feedback.
  EXPECT_DOUBLE_EQ(commands[0], 0.5);
  EXPECT_DOUBLE_EQ(commands[1], -0.25);
  EXPECT_NEAR(positions[0], 0.0, 1e-9);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
