#include <gtest/gtest.h>

#include <set>
#include <unistd.h>
#include <utility>
#include <vector>

#include "so101_hardware/so101_system_hardware.hpp"

namespace
{
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

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
