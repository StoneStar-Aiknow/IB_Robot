#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace x2::cbf {

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  Vec3& operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  Vec3& operator-=(const Vec3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  Vec3& operator*=(double scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }
};

inline Vec3 operator+(Vec3 lhs, const Vec3& rhs) { return lhs += rhs; }
inline Vec3 operator-(Vec3 lhs, const Vec3& rhs) { return lhs -= rhs; }
inline Vec3 operator-(const Vec3& value) { return {-value.x, -value.y, -value.z}; }
inline Vec3 operator*(Vec3 value, double scalar) { return value *= scalar; }
inline Vec3 operator*(double scalar, Vec3 value) { return value *= scalar; }
inline Vec3 operator/(Vec3 value, double scalar) { return value *= 1.0 / scalar; }

inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

inline double squared_norm(const Vec3& value) { return dot(value, value); }
inline double norm(const Vec3& value) { return std::sqrt(squared_norm(value)); }

inline Vec3 normalized(const Vec3& value, const Vec3& fallback = {1.0, 0.0, 0.0}) {
  const double length = norm(value);
  return length > 1e-12 ? value / length : fallback;
}

struct Mat3 {
  std::array<std::array<double, 3>, 3> value{{
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}},
  }};
};

inline Mat3 identity_mat3() { return {}; }

inline Vec3 operator*(const Mat3& matrix, const Vec3& vector) {
  return {
      matrix.value[0][0] * vector.x + matrix.value[0][1] * vector.y + matrix.value[0][2] * vector.z,
      matrix.value[1][0] * vector.x + matrix.value[1][1] * vector.y + matrix.value[1][2] * vector.z,
      matrix.value[2][0] * vector.x + matrix.value[2][1] * vector.y + matrix.value[2][2] * vector.z,
  };
}

inline Mat3 operator*(const Mat3& lhs, const Mat3& rhs) {
  Mat3 result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result.value[row][column] = 0.0;
      for (int index = 0; index < 3; ++index) {
        result.value[row][column] += lhs.value[row][index] * rhs.value[index][column];
      }
    }
  }
  return result;
}

inline Mat3 transpose(const Mat3& matrix) {
  Mat3 result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result.value[row][column] = matrix.value[column][row];
    }
  }
  return result;
}

inline Mat3 rpy(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  Mat3 result;
  result.value = {{
      {{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr}},
      {{sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr}},
      {{-sp, cp * sr, cp * cr}},
  }};
  return result;
}

inline Mat3 quaternion(double w, double x, double y, double z) {
  const double scale = 1.0 / std::sqrt(w * w + x * x + y * y + z * z);
  w *= scale;
  x *= scale;
  y *= scale;
  z *= scale;
  Mat3 result;
  result.value = {{
      {{1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)}},
      {{2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)}},
      {{2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)}},
  }};
  return result;
}

inline Mat3 axis_angle(Vec3 axis, double angle) {
  axis = normalized(axis);
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double one_minus_c = 1.0 - c;
  Mat3 result;
  result.value = {{
      {{c + axis.x * axis.x * one_minus_c,
        axis.x * axis.y * one_minus_c - axis.z * s,
        axis.x * axis.z * one_minus_c + axis.y * s}},
      {{axis.y * axis.x * one_minus_c + axis.z * s,
        c + axis.y * axis.y * one_minus_c,
        axis.y * axis.z * one_minus_c - axis.x * s}},
      {{axis.z * axis.x * one_minus_c - axis.y * s,
        axis.z * axis.y * one_minus_c + axis.x * s,
        c + axis.z * axis.z * one_minus_c}},
  }};
  return result;
}

struct Transform {
  Mat3 rotation{};
  Vec3 translation{};
};

inline Transform identity_transform() { return {}; }

inline Transform operator*(const Transform& lhs, const Transform& rhs) {
  return {lhs.rotation * rhs.rotation,
          lhs.translation + lhs.rotation * rhs.translation};
}

inline Vec3 transform_point(const Transform& transform, const Vec3& point) {
  return transform.translation + transform.rotation * point;
}

inline Transform inverse(const Transform& transform) {
  const Mat3 rotation = transpose(transform.rotation);
  return {rotation, -(rotation * transform.translation)};
}

inline Transform make_transform(const Vec3& translation, const Mat3& rotation) {
  return {rotation, translation};
}

}  // namespace x2::cbf
