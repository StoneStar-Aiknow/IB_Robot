#pragma once

#include "cbf/math.hpp"

#include <array>
#include <cstddef>

namespace x2::cbf {

enum class PrimitiveKind { Sphere, Capsule, Box };

enum class PrimitiveId : std::size_t {
  Head = 0,
  Torso = 1,
  LeftThigh = 2,
  RightThigh = 3,
  LeftShoulder = 4,
  LeftForearm = 5,
  LeftPalm = 6,
  LeftThumb = 7,
  RightShoulder = 8,
  RightForearm = 9,
  RightPalm = 10,
  RightThumb = 11,
};

inline constexpr std::size_t kProxyPrimitiveCount = 12;
inline constexpr std::size_t kPrimitivePairCount = 48;

struct PrimitivePair {
  PrimitiveId first;
  PrimitiveId second;
};

inline constexpr std::array<PrimitivePair, kPrimitivePairCount> kPrimitivePairs = {{
    {PrimitiveId::LeftShoulder, PrimitiveId::Head},
    {PrimitiveId::LeftShoulder, PrimitiveId::Torso},
    {PrimitiveId::LeftShoulder, PrimitiveId::LeftThigh},
    {PrimitiveId::LeftShoulder, PrimitiveId::RightThigh},
    {PrimitiveId::LeftForearm, PrimitiveId::Head},
    {PrimitiveId::LeftForearm, PrimitiveId::Torso},
    {PrimitiveId::LeftForearm, PrimitiveId::LeftThigh},
    {PrimitiveId::LeftForearm, PrimitiveId::RightThigh},
    {PrimitiveId::LeftPalm, PrimitiveId::Head},
    {PrimitiveId::LeftPalm, PrimitiveId::Torso},
    {PrimitiveId::LeftPalm, PrimitiveId::LeftThigh},
    {PrimitiveId::LeftPalm, PrimitiveId::RightThigh},
    {PrimitiveId::LeftThumb, PrimitiveId::Head},
    {PrimitiveId::LeftThumb, PrimitiveId::Torso},
    {PrimitiveId::LeftThumb, PrimitiveId::LeftThigh},
    {PrimitiveId::LeftThumb, PrimitiveId::RightThigh},
    {PrimitiveId::RightShoulder, PrimitiveId::Head},
    {PrimitiveId::RightShoulder, PrimitiveId::Torso},
    {PrimitiveId::RightShoulder, PrimitiveId::LeftThigh},
    {PrimitiveId::RightShoulder, PrimitiveId::RightThigh},
    {PrimitiveId::RightForearm, PrimitiveId::Head},
    {PrimitiveId::RightForearm, PrimitiveId::Torso},
    {PrimitiveId::RightForearm, PrimitiveId::LeftThigh},
    {PrimitiveId::RightForearm, PrimitiveId::RightThigh},
    {PrimitiveId::RightPalm, PrimitiveId::Head},
    {PrimitiveId::RightPalm, PrimitiveId::Torso},
    {PrimitiveId::RightPalm, PrimitiveId::LeftThigh},
    {PrimitiveId::RightPalm, PrimitiveId::RightThigh},
    {PrimitiveId::RightThumb, PrimitiveId::Head},
    {PrimitiveId::RightThumb, PrimitiveId::Torso},
    {PrimitiveId::RightThumb, PrimitiveId::LeftThigh},
    {PrimitiveId::RightThumb, PrimitiveId::RightThigh},
    {PrimitiveId::LeftShoulder, PrimitiveId::RightShoulder},
    {PrimitiveId::LeftShoulder, PrimitiveId::RightForearm},
    {PrimitiveId::LeftShoulder, PrimitiveId::RightPalm},
    {PrimitiveId::LeftShoulder, PrimitiveId::RightThumb},
    {PrimitiveId::LeftForearm, PrimitiveId::RightShoulder},
    {PrimitiveId::LeftForearm, PrimitiveId::RightForearm},
    {PrimitiveId::LeftForearm, PrimitiveId::RightPalm},
    {PrimitiveId::LeftForearm, PrimitiveId::RightThumb},
    {PrimitiveId::LeftPalm, PrimitiveId::RightShoulder},
    {PrimitiveId::LeftPalm, PrimitiveId::RightForearm},
    {PrimitiveId::LeftPalm, PrimitiveId::RightPalm},
    {PrimitiveId::LeftPalm, PrimitiveId::RightThumb},
    {PrimitiveId::LeftThumb, PrimitiveId::RightShoulder},
    {PrimitiveId::LeftThumb, PrimitiveId::RightForearm},
    {PrimitiveId::LeftThumb, PrimitiveId::RightPalm},
    {PrimitiveId::LeftThumb, PrimitiveId::RightThumb},
}};

struct ProxyPrimitive {
  PrimitiveKind kind{PrimitiveKind::Sphere};
  Transform pose{};
  Vec3 size{};
  double radius{0.0};
  double length{0.0};
  double margin{0.0};
};

struct DistanceResult {
  double signed_distance{0.0};
  Vec3 point_a{};
  Vec3 point_b{};
  Vec3 normal{1.0, 0.0, 0.0};
  bool valid{false};
};

struct PrimitiveDistanceReport {
  std::array<DistanceResult, kPrimitivePairCount> distances{};
  double minimum_distance{0.0};
  std::size_t minimum_pair_index{0};
};

const char* primitive_name(PrimitiveId id);
DistanceResult distance_between(const ProxyPrimitive& first,
                                const ProxyPrimitive& second);

}  // namespace x2::cbf
