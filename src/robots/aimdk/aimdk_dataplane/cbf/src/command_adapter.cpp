#include "cbf/command_adapter.hpp"

namespace x2::cbf {

ThumbCommandState extract_thumb_commands(
    const std::array<double, kHandDof>& hand_pos) {
  ThumbCommandState result;
  result.command_values[0] = hand_pos[0];
  result.command_values[1] = hand_pos[1];
  result.command_values[2] = hand_pos[2];
  result.command_values[3] = hand_pos[10];
  result.command_values[4] = hand_pos[11];
  result.command_values[5] = hand_pos[12];
  return result;
}

SafetyCommandResult pass_through_hand_and_arm(
    const UpperBodySafetyCommand& command,
    const std::array<double, kArmDof>& safe_arm_pos) {
  SafetyCommandResult result;
  result.safe_arm_pos = safe_arm_pos;
  result.hand_pos = command.hand_pos;
  return result;
}

}  // namespace x2::cbf
