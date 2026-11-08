#pragma once

#include <array>
#include <cstddef>

namespace x2::cbf::generated {

inline constexpr std::size_t kArmDof = 14;
inline constexpr std::size_t kLogicalPairCount = 33;
inline constexpr std::size_t kPrimitivePairCount = 48;

inline constexpr std::array<const char*, kArmDof> kArmJointNames = {
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
};

inline constexpr std::array<std::array<double, 2>, kArmDof> kArmLimits = {{
    {-3.0805060000, 2.0402900000},
    {-0.0523600000, 3.1241390000},
    {-2.5307270000, 2.5307270000},
    {-2.3561940000, 0.0000000000},
    {-2.5481810000, 2.5481810000},
    {-0.5235990000, 0.5235990000},
    {-0.6981320000, 0.6981320000},
    {-3.0805060000, 2.0402900000},
    {-3.1241390000, 0.0523600000},
    {-2.5307270000, 2.5307270000},
    {-2.3561940000, 0.0000000000},
    {-2.5481810000, 2.5481810000},
    {-0.5235990000, 0.5235990000},
    {-0.6981320000, 0.6981320000},
}};

inline constexpr std::array<double, kArmDof> kNaturalStanding = {
    0.3925980000,
    -0.0252150000,
    -0.0135180000,
    -1.1290900000,
    0.0068070000,
    0.0116350000,
    0.0000000000,
    0.3906800000,
    0.0282820000,
    0.0221470000,
    -1.1471140000,
    0.0069990000,
    0.0001910000,
    0.0051500000,
};

inline constexpr std::array<double, 6> kFixedBaselineThumbUrdf = {
    0.9023352190,
    -0.4609547000,
    0.5891200000,
    0.3316125800,
    -0.4075056524,
    0.5891184357,
};

struct PairName {
  const char* first;
  const char* second;
};

inline constexpr std::array<PairName, kLogicalPairCount> kLogicalPairs = {{
    {"left_shoulder", "head"},
    {"left_shoulder", "torso"},
    {"left_shoulder", "left_thigh"},
    {"left_shoulder", "right_thigh"},
    {"left_forearm", "head"},
    {"left_forearm", "torso"},
    {"left_forearm", "left_thigh"},
    {"left_forearm", "right_thigh"},
    {"left_hand", "head"},
    {"left_hand", "torso"},
    {"left_hand", "left_thigh"},
    {"left_hand", "right_thigh"},
    {"right_shoulder", "head"},
    {"right_shoulder", "torso"},
    {"right_shoulder", "left_thigh"},
    {"right_shoulder", "right_thigh"},
    {"right_forearm", "head"},
    {"right_forearm", "torso"},
    {"right_forearm", "left_thigh"},
    {"right_forearm", "right_thigh"},
    {"right_hand", "head"},
    {"right_hand", "torso"},
    {"right_hand", "left_thigh"},
    {"right_hand", "right_thigh"},
    {"left_shoulder", "right_shoulder"},
    {"left_shoulder", "right_forearm"},
    {"left_shoulder", "right_hand"},
    {"left_forearm", "right_shoulder"},
    {"left_forearm", "right_forearm"},
    {"left_forearm", "right_hand"},
    {"left_hand", "right_shoulder"},
    {"left_hand", "right_forearm"},
    {"left_hand", "right_hand"},
}};

inline constexpr std::array<PairName, kPrimitivePairCount> kPrimitivePairs = {{
    {"left_shoulder", "head"},
    {"left_shoulder", "torso"},
    {"left_shoulder", "left_thigh"},
    {"left_shoulder", "right_thigh"},
    {"left_forearm", "head"},
    {"left_forearm", "torso"},
    {"left_forearm", "left_thigh"},
    {"left_forearm", "right_thigh"},
    {"left_hand__palm_four_fingers", "head"},
    {"left_hand__palm_four_fingers", "torso"},
    {"left_hand__palm_four_fingers", "left_thigh"},
    {"left_hand__palm_four_fingers", "right_thigh"},
    {"left_hand__thumb_swept", "head"},
    {"left_hand__thumb_swept", "torso"},
    {"left_hand__thumb_swept", "left_thigh"},
    {"left_hand__thumb_swept", "right_thigh"},
    {"right_shoulder", "head"},
    {"right_shoulder", "torso"},
    {"right_shoulder", "left_thigh"},
    {"right_shoulder", "right_thigh"},
    {"right_forearm", "head"},
    {"right_forearm", "torso"},
    {"right_forearm", "left_thigh"},
    {"right_forearm", "right_thigh"},
    {"right_hand__palm_four_fingers", "head"},
    {"right_hand__palm_four_fingers", "torso"},
    {"right_hand__palm_four_fingers", "left_thigh"},
    {"right_hand__palm_four_fingers", "right_thigh"},
    {"right_hand__thumb_swept", "head"},
    {"right_hand__thumb_swept", "torso"},
    {"right_hand__thumb_swept", "left_thigh"},
    {"right_hand__thumb_swept", "right_thigh"},
    {"left_shoulder", "right_shoulder"},
    {"left_shoulder", "right_forearm"},
    {"left_shoulder", "right_hand__palm_four_fingers"},
    {"left_shoulder", "right_hand__thumb_swept"},
    {"left_forearm", "right_shoulder"},
    {"left_forearm", "right_forearm"},
    {"left_forearm", "right_hand__palm_four_fingers"},
    {"left_forearm", "right_hand__thumb_swept"},
    {"left_hand__palm_four_fingers", "right_shoulder"},
    {"left_hand__palm_four_fingers", "right_forearm"},
    {"left_hand__palm_four_fingers", "right_hand__palm_four_fingers"},
    {"left_hand__palm_four_fingers", "right_hand__thumb_swept"},
    {"left_hand__thumb_swept", "right_shoulder"},
    {"left_hand__thumb_swept", "right_forearm"},
    {"left_hand__thumb_swept", "right_hand__palm_four_fingers"},
    {"left_hand__thumb_swept", "right_hand__thumb_swept"},
}};

}  // namespace x2::cbf::generated
