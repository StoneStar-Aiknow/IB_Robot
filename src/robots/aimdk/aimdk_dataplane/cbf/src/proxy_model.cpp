#include "cbf/proxy_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace x2::cbf {
namespace {

constexpr double kEpsilon = 1e-12;
Vec3 closest_point_on_segment(const Vec3& point, const Vec3& start,
                              const Vec3& end) {
  const Vec3 direction = end - start;
  const double denominator = squared_norm(direction);
  const double parameter = denominator > kEpsilon
                              ? std::clamp(dot(point - start, direction) / denominator, 0.0, 1.0)
                              : 0.0;
  return start + direction * parameter;
}

void segment_endpoints(const ProxyPrimitive& capsule, Vec3& start, Vec3& end) {
  const Vec3 half_axis = capsule.pose.rotation * Vec3{0.0, 0.0, capsule.length * 0.5};
  start = capsule.pose.translation - half_axis;
  end = capsule.pose.translation + half_axis;
}

Vec3 support_point(const ProxyPrimitive& box, const Vec3& direction) {
  const Mat3 inverse_rotation = transpose(box.pose.rotation);
  const Vec3 local_direction = inverse_rotation * direction;
  const Vec3 half = box.size * 0.5;
  const Vec3 local_point{
      local_direction.x >= 0.0 ? half.x : -half.x,
      local_direction.y >= 0.0 ? half.y : -half.y,
      local_direction.z >= 0.0 ? half.z : -half.z,
  };
  return transform_point(box.pose, local_point);
}

std::array<Vec3, 8> box_vertices(const ProxyPrimitive& box) {
  const Vec3 half = box.size * 0.5;
  std::array<Vec3, 8> vertices{};
  std::size_t index = 0;
  for (double x : {-half.x, half.x}) {
    for (double y : {-half.y, half.y}) {
      for (double z : {-half.z, half.z}) {
        vertices[index++] = transform_point(box.pose, {x, y, z});
      }
    }
  }
  return vertices;
}

struct BoxPointQuery {
  double signed_distance{0.0};
  Vec3 point_on_box{};
  Vec3 outward_normal{1.0, 0.0, 0.0};
};

BoxPointQuery point_box_query(const Vec3& point, const ProxyPrimitive& box) {
  const Mat3 inverse_rotation = transpose(box.pose.rotation);
  const Vec3 local_point = inverse_rotation * (point - box.pose.translation);
  const Vec3 half = box.size * 0.5;
  const Vec3 clamped{
      std::clamp(local_point.x, -half.x, half.x),
      std::clamp(local_point.y, -half.y, half.y),
      std::clamp(local_point.z, -half.z, half.z),
  };
  const Vec3 delta = clamped - local_point;
  const bool inside = std::abs(local_point.x) <= half.x &&
                      std::abs(local_point.y) <= half.y &&
                      std::abs(local_point.z) <= half.z;
  Vec3 local_surface = clamped;
  Vec3 local_normal{1.0, 0.0, 0.0};
  double signed_distance = 0.0;
  if (!inside) {
    const double outside_distance = norm(delta);
    signed_distance = outside_distance;
    local_normal = normalized(-delta);
  } else {
    const double distances[3] = {half.x - std::abs(local_point.x),
                                 half.y - std::abs(local_point.y),
                                 half.z - std::abs(local_point.z)};
    int axis = 0;
    if (distances[1] < distances[axis]) axis = 1;
    if (distances[2] < distances[axis]) axis = 2;
    local_surface = local_point;
    if (axis == 0) {
      local_surface.x = local_point.x >= 0.0 ? half.x : -half.x;
      local_normal = {local_point.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0};
      signed_distance = -distances[0];
    } else if (axis == 1) {
      local_surface.y = local_point.y >= 0.0 ? half.y : -half.y;
      local_normal = {0.0, local_point.y >= 0.0 ? 1.0 : -1.0, 0.0};
      signed_distance = -distances[1];
    } else {
      local_surface.z = local_point.z >= 0.0 ? half.z : -half.z;
      local_normal = {0.0, 0.0, local_point.z >= 0.0 ? 1.0 : -1.0};
      signed_distance = -distances[2];
    }
  }
  return {signed_distance, transform_point(box.pose, local_surface),
          normalized(box.pose.rotation * local_normal)};
}

DistanceResult sphere_sphere(const ProxyPrimitive& a, const ProxyPrimitive& b) {
  const Vec3 delta = b.pose.translation - a.pose.translation;
  const Vec3 normal = normalized(delta);
  const double distance = norm(delta) - a.radius - b.radius;
  return {distance, a.pose.translation + normal * a.radius,
          b.pose.translation - normal * b.radius, normal, true};
}

DistanceResult capsule_sphere(const ProxyPrimitive& capsule,
                              const ProxyPrimitive& sphere) {
  Vec3 start, end;
  segment_endpoints(capsule, start, end);
  const Vec3 centerline_point = closest_point_on_segment(sphere.pose.translation, start, end);
  const Vec3 normal = normalized(sphere.pose.translation - centerline_point);
  const double distance = norm(sphere.pose.translation - centerline_point) -
                          capsule.radius - sphere.radius;
  return {distance, centerline_point + normal * capsule.radius,
          sphere.pose.translation - normal * sphere.radius, normal, true};
}

double clamp01(double value) { return std::clamp(value, 0.0, 1.0); }

void closest_segment_points(const Vec3& p0, const Vec3& p1, const Vec3& q0,
                            const Vec3& q1, Vec3& p, Vec3& q) {
  const Vec3 d1 = p1 - p0;
  const Vec3 d2 = q1 - q0;
  const Vec3 r = p0 - q0;
  const double a = dot(d1, d1);
  const double e = dot(d2, d2);
  const double f = dot(d2, r);
  double s = 0.0;
  double t = 0.0;
  if (a <= kEpsilon && e <= kEpsilon) {
    p = p0;
    q = q0;
    return;
  }
  if (a <= kEpsilon) {
    s = 0.0;
    t = clamp01(f / e);
  } else {
    const double c = dot(d1, r);
    if (e <= kEpsilon) {
      t = 0.0;
      s = clamp01(-c / a);
    } else {
      const double b = dot(d1, d2);
      const double denominator = a * e - b * b;
      if (denominator != 0.0) {
        s = clamp01((b * f - c * e) / denominator);
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = clamp01(-c / a);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp01((b - c) / a);
      }
    }
  }
  p = p0 + d1 * s;
  q = q0 + d2 * t;
}

DistanceResult capsule_capsule(const ProxyPrimitive& a, const ProxyPrimitive& b) {
  Vec3 a_start, a_end, b_start, b_end;
  segment_endpoints(a, a_start, a_end);
  segment_endpoints(b, b_start, b_end);
  Vec3 point_a, point_b;
  closest_segment_points(a_start, a_end, b_start, b_end, point_a, point_b);
  const Vec3 normal = normalized(point_b - point_a);
  const double distance = norm(point_b - point_a) - a.radius - b.radius;
  return {distance, point_a + normal * a.radius,
          point_b - normal * b.radius, normal, true};
}

DistanceResult sphere_box(const ProxyPrimitive& sphere, const ProxyPrimitive& box) {
  const BoxPointQuery query = point_box_query(sphere.pose.translation, box);
  const double distance = query.signed_distance - sphere.radius;
  const Vec3 normal = query.signed_distance >= 0.0
                          ? -query.outward_normal
                          : query.outward_normal;
  return {distance, sphere.pose.translation + normal * sphere.radius,
          query.point_on_box, normal, true};
}

DistanceResult capsule_box(const ProxyPrimitive& capsule, const ProxyPrimitive& box) {
  Vec3 start, end;
  segment_endpoints(capsule, start, end);
  double best_distance = std::numeric_limits<double>::infinity();
  Vec3 best_segment{};
  Vec3 best_box{};
  Vec3 best_normal{1.0, 0.0, 0.0};
  constexpr int kSamples = 65;
  for (int index = 0; index < kSamples; ++index) {
    const double parameter = static_cast<double>(index) / (kSamples - 1);
    const Vec3 point = start + (end - start) * parameter;
    const BoxPointQuery query = point_box_query(point, box);
    if (query.signed_distance < best_distance) {
      best_distance = query.signed_distance;
      best_segment = point;
      best_box = query.point_on_box;
      best_normal = query.signed_distance >= 0.0
                        ? -query.outward_normal
                        : query.outward_normal;
    }
  }
  const double distance = best_distance - capsule.radius;
  return {distance, best_segment + best_normal * capsule.radius, best_box,
          best_normal, true};
}

DistanceResult box_box(const ProxyPrimitive& a, const ProxyPrimitive& b) {
  const Vec3 axes_a[3] = {
      {a.pose.rotation.value[0][0], a.pose.rotation.value[1][0], a.pose.rotation.value[2][0]},
      {a.pose.rotation.value[0][1], a.pose.rotation.value[1][1], a.pose.rotation.value[2][1]},
      {a.pose.rotation.value[0][2], a.pose.rotation.value[1][2], a.pose.rotation.value[2][2]},
  };
  const Vec3 axes_b[3] = {
      {b.pose.rotation.value[0][0], b.pose.rotation.value[1][0], b.pose.rotation.value[2][0]},
      {b.pose.rotation.value[0][1], b.pose.rotation.value[1][1], b.pose.rotation.value[2][1]},
      {b.pose.rotation.value[0][2], b.pose.rotation.value[1][2], b.pose.rotation.value[2][2]},
  };
  const Vec3 half_a = a.size * 0.5;
  const Vec3 half_b = b.size * 0.5;
  const Vec3 center_delta = b.pose.translation - a.pose.translation;
  double maximum_overlap = -std::numeric_limits<double>::infinity();
  double minimum_gap = std::numeric_limits<double>::infinity();
  bool separated = false;
  Vec3 best_axis{1.0, 0.0, 0.0};
  auto evaluate_axis = [&](Vec3 axis) {
    axis = normalized(axis);
    const double radius_a = std::abs(dot(axis, axes_a[0])) * half_a.x +
                            std::abs(dot(axis, axes_a[1])) * half_a.y +
                            std::abs(dot(axis, axes_a[2])) * half_a.z;
    const double radius_b = std::abs(dot(axis, axes_b[0])) * half_b.x +
                            std::abs(dot(axis, axes_b[1])) * half_b.y +
                            std::abs(dot(axis, axes_b[2])) * half_b.z;
    const double signed_axis_distance = std::abs(dot(center_delta, axis)) - radius_a - radius_b;
    if (dot(center_delta, axis) < 0.0) axis = -axis;
    if (signed_axis_distance > 0.0) {
      separated = true;
      if (signed_axis_distance < minimum_gap) {
        minimum_gap = signed_axis_distance;
        best_axis = axis;
      }
    } else if (!separated && signed_axis_distance > maximum_overlap) {
      maximum_overlap = signed_axis_distance;
      best_axis = axis;
    }
  };
  for (const Vec3& axis : axes_a) evaluate_axis(axis);
  for (const Vec3& axis : axes_b) evaluate_axis(axis);
  for (const Vec3& axis_a : axes_a) {
    for (const Vec3& axis_b : axes_b) {
      const Vec3 candidate = cross(axis_a, axis_b);
      if (squared_norm(candidate) > 1e-10) evaluate_axis(candidate);
    }
  }
  if (!separated) {
    const Vec3 point_a = support_point(a, best_axis);
    const Vec3 point_b = support_point(b, -best_axis);
    return {maximum_overlap, point_a, point_b, best_axis, true};
  }

  constexpr std::array<std::array<int, 2>, 12> kEdges = {{
      {{0, 1}}, {{0, 2}}, {{0, 4}}, {{1, 3}}, {{1, 5}}, {{2, 3}},
      {{2, 6}}, {{3, 7}}, {{4, 5}}, {{4, 6}}, {{5, 7}}, {{6, 7}},
  }};
  const std::array<Vec3, 8> vertices_a = box_vertices(a);
  const std::array<Vec3, 8> vertices_b = box_vertices(b);
  double best_squared_distance = std::numeric_limits<double>::infinity();
  Vec3 closest_a{};
  Vec3 closest_b{};

  auto update_closest = [&](const Vec3& candidate_a,
                            const Vec3& candidate_b) {
    const double squared_distance =
        squared_norm(candidate_b - candidate_a);
    if (squared_distance < best_squared_distance) {
      best_squared_distance = squared_distance;
      closest_a = candidate_a;
      closest_b = candidate_b;
    }
  };

  for (const Vec3& vertex : vertices_a) {
    const BoxPointQuery query = point_box_query(vertex, b);
    update_closest(vertex, query.point_on_box);
  }
  for (const Vec3& vertex : vertices_b) {
    const BoxPointQuery query = point_box_query(vertex, a);
    update_closest(query.point_on_box, vertex);
  }
  for (const auto& edge_a : kEdges) {
    for (const auto& edge_b : kEdges) {
      Vec3 point_a;
      Vec3 point_b;
      closest_segment_points(vertices_a[edge_a[0]], vertices_a[edge_a[1]],
                             vertices_b[edge_b[0]], vertices_b[edge_b[1]],
                             point_a, point_b);
      update_closest(point_a, point_b);
    }
  }
  const double distance = std::sqrt(best_squared_distance);
  return {distance, closest_a, closest_b,
          normalized(closest_b - closest_a, best_axis), true};
}

DistanceResult swap_result(const DistanceResult& result) {
  return {result.signed_distance, result.point_b, result.point_a, -result.normal, result.valid};
}

}  // namespace

const char* primitive_name(PrimitiveId id) {
  constexpr const char* kNames[kProxyPrimitiveCount] = {
      "head", "torso", "left_thigh", "right_thigh", "left_shoulder",
      "left_forearm", "left_hand__palm_four_fingers", "left_hand__thumb_swept",
      "right_shoulder", "right_forearm", "right_hand__palm_four_fingers",
      "right_hand__thumb_swept"};
  return kNames[static_cast<std::size_t>(id)];
}

DistanceResult distance_between(const ProxyPrimitive& first,
                                const ProxyPrimitive& second) {
  if (first.kind == PrimitiveKind::Sphere && second.kind == PrimitiveKind::Sphere) {
    return sphere_sphere(first, second);
  }
  if (first.kind == PrimitiveKind::Capsule && second.kind == PrimitiveKind::Sphere) {
    return capsule_sphere(first, second);
  }
  if (first.kind == PrimitiveKind::Sphere && second.kind == PrimitiveKind::Capsule) {
    return swap_result(capsule_sphere(second, first));
  }
  if (first.kind == PrimitiveKind::Capsule && second.kind == PrimitiveKind::Capsule) {
    return capsule_capsule(first, second);
  }
  if (first.kind == PrimitiveKind::Sphere && second.kind == PrimitiveKind::Box) {
    return sphere_box(first, second);
  }
  if (first.kind == PrimitiveKind::Box && second.kind == PrimitiveKind::Sphere) {
    return swap_result(sphere_box(second, first));
  }
  if (first.kind == PrimitiveKind::Capsule && second.kind == PrimitiveKind::Box) {
    return capsule_box(first, second);
  }
  if (first.kind == PrimitiveKind::Box && second.kind == PrimitiveKind::Capsule) {
    return swap_result(capsule_box(second, first));
  }
  if (first.kind == PrimitiveKind::Box && second.kind == PrimitiveKind::Box) {
    return box_box(first, second);
  }
  return {};
}

}  // namespace x2::cbf
