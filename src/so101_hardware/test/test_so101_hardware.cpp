#include <gtest/gtest.h>

#include <unistd.h>
#include <utility>
#include <vector>

#include "so101_hardware/so101_system_hardware.hpp"

namespace {
class TestableSafeSMSSTS : public so101_hardware::SafeSMSSTS {
public:
  int read_for_test(unsigned char *data, int length, unsigned long timeout_ms) {
    return readSCS(data, length, timeout_ms);
  }

  void set_fd_for_test(int value) { fd = value; }
};
} // namespace

TEST(SO101Hardware,
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
      so101_hardware::detail::disable_torque_on_abort({1}, [&attempts](u8, u8) {
        ++attempts;
        return 0;
      });

  EXPECT_FALSE(disabled);
  EXPECT_EQ(attempts, 3);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
