#pragma once

#include <array>
#include <cstddef>

namespace x2::cbf {

inline constexpr std::size_t kArmDof = 14;
inline constexpr std::size_t kHandDof = 20;
inline constexpr std::size_t kThumbDof = 6;

struct UpperBodySafetyCommand {
  std::array<double, kArmDof> arm_pos{};
  std::array<double, kHandDof> hand_pos{};
};

struct ThumbCommandState {
  std::array<double, kThumbDof> command_values{};
  bool mapping_valid{false};
};

struct SafetyCommandResult {
  std::array<double, kArmDof> safe_arm_pos{};
  std::array<double, kHandDof> hand_pos{};
};

ThumbCommandState extract_thumb_commands(
    const std::array<double, kHandDof>& hand_pos);

SafetyCommandResult pass_through_hand_and_arm(
    const UpperBodySafetyCommand& command,
    const std::array<double, kArmDof>& safe_arm_pos);

}  // namespace x2::cbf
