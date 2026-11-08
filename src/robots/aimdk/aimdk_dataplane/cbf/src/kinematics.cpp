#include "cbf/kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace x2::cbf {
namespace {

constexpr double kPi = 3.14159265358979323846;

Transform body_step(const Transform& parent, const Vec3& position,
                    const Mat3& orientation, const Vec3& axis, double angle,
                    RevoluteJointFrame* joint_frame = nullptr) {
  const Transform joint_origin = parent * make_transform(position, orientation);
  if (joint_frame != nullptr) {
    joint_frame->origin_world = joint_origin.translation;
    joint_frame->axis_world = normalized(joint_origin.rotation * axis);
  }
  return joint_origin *
         make_transform({}, axis_angle(axis, angle));
}

Transform zero_body_step(const Transform& parent, const Vec3& position,
                         const Mat3& orientation) {
  return parent * make_transform(position, orientation);
}

Transform thumb_root_chain(const std::array<double, 2>& q, bool right) {
  Transform transform = identity_transform();
  if (!right) {
    transform = body_step(transform, {0.0122824449425584, -0.018521969911508, 0.0468117154849971},
                          identity_mat3(), {-1.0, 0.0, 0.0}, q[0]);
    transform = body_step(transform, {0.0089997, -0.0043278, 0.00022366},
                          rpy(0.0, 0.0, 1.5708), {-0.087155, 0.0, 0.99619}, q[1]);
  } else {
    transform = body_step(transform, {-0.0118175567463393, 0.018521969911508, 0.0468117154849971},
                          identity_mat3(), {1.0, 0.0, 0.0}, q[0]);
    transform = body_step(transform, {0.03310000187541, 0.00364597550394604, 0.00234265623980155},
                          identity_mat3(), {0.0, 0.418896236019724, -0.908034109187814}, q[1]);
  }
  return transform;
}

Transform hand_mount(bool right) {
  return make_transform({0.0, 0.0, -0.0415},
                        rpy(kPi, 0.0, right ? kPi / 2.0 : -kPi / 2.0));
}

Transform dynamic_thumb_pose(const Transform& wrist, bool right,
                             const std::array<double, 2>& q) {
  const Transform zero_chain = thumb_root_chain({0.0, 0.0}, right);
  const Transform current_chain = thumb_root_chain(q, right);
  const Transform delta_hand = current_chain * inverse(zero_chain);
  const Transform mount = hand_mount(right);
  const Transform delta_wrist = mount * delta_hand * inverse(mount);
  const Vec3 base_pos = right ? Vec3{0.0294, 0.0831, -0.0858}
                              : Vec3{0.0256, -0.0831, -0.0809};
  const Mat3 base_rotation = right ? rpy(1.6198, -0.1106, -0.1161)
                                   : rpy(1.4778, 0.0507, 0.0462);
  return wrist * delta_wrist * make_transform(base_pos, base_rotation);
}

void set_sphere(ProxyPrimitive& primitive, const Transform& body,
                const Vec3& local_position, double radius, double margin) {
  primitive.kind = PrimitiveKind::Sphere;
  primitive.pose = body * make_transform(local_position, identity_mat3());
  primitive.radius = radius;
  primitive.margin = margin;
}

void set_capsule(ProxyPrimitive& primitive, const Transform& body,
                 const Vec3& local_position, const Mat3& local_rotation,
                 double radius, double length, double margin) {
  primitive.kind = PrimitiveKind::Capsule;
  primitive.pose = body * make_transform(local_position, local_rotation);
  primitive.radius = radius;
  primitive.length = length;
  primitive.margin = margin;
}

void set_box(ProxyPrimitive& primitive, const Transform& body,
             const Vec3& local_position, const Mat3& local_rotation,
             const Vec3& size, double margin) {
  primitive.kind = PrimitiveKind::Box;
  primitive.pose = body * make_transform(local_position, local_rotation);
  primitive.size = size;
  primitive.margin = margin;
}

}  // namespace

KinematicsOutput forward_kinematics(const KinematicsInput& input) {
  KinematicsOutput output;
  output.hands_installed = input.hands_installed;
  const Transform pelvis = make_transform({0.0, 0.0, 0.68}, identity_mat3());
  const Transform waist_yaw = zero_body_step(pelvis, {0.0, 0.0, 0.06706}, identity_mat3());
  const Transform waist_pitch = zero_body_step(waist_yaw, {0.0, 0.0, 0.0880106}, identity_mat3());
  const Transform torso = waist_pitch;

  const Transform left_shoulder_pitch = body_step(
      torso, {0.00329907, 0.143032, 0.241648}, quaternion(0.994522, 0.104528, 0.0, 0.0),
      {0.0, 1.0, 0.0}, input.q_arm[0], &output.arm_joint_frames[0]);
  const Transform left_shoulder_roll = body_step(
      left_shoulder_pitch, {-0.000499991, 0.0495, 0.0}, identity_mat3(),
      {1.0, 0.0, 0.0}, input.q_arm[1], &output.arm_joint_frames[1]);
  const Transform left_shoulder_yaw = body_step(
      left_shoulder_roll, {0.00122424, 0.0, -0.121195}, quaternion(0.999996, 0.0, -0.00298782, 0.0),
      {0.0, 0.0, 1.0}, input.q_arm[2], &output.arm_joint_frames[2]);
  const Transform left_elbow = body_step(
      left_shoulder_yaw, {0.014, 0.000199986, -0.08295}, quaternion(0.999996, 0.0, 0.00298782, 0.0),
      {0.0, 1.0, 0.0}, input.q_arm[3], &output.arm_joint_frames[3]);
  const Transform left_wrist_yaw = body_step(
      left_elbow, {-0.0132797, -9.75488e-05, -0.120576}, identity_mat3(),
      {0.0, 0.0, 1.0}, input.q_arm[4], &output.arm_joint_frames[4]);
  const Transform left_wrist_pitch = body_step(
      left_wrist_yaw, {0.000490001, 0.0, -0.0819985}, quaternion(0.999996, 0.0, -0.00298783, 0.0),
      {0.0, 1.0, 0.0}, input.q_arm[5], &output.arm_joint_frames[5]);
  const Transform left_wrist_roll = body_step(
      left_wrist_pitch, {}, identity_mat3(), {1.0, 0.0, 0.0}, input.q_arm[6],
      &output.arm_joint_frames[6]);

  const Transform right_shoulder_pitch = body_step(
      torso, {0.00330094, -0.143031, 0.241648}, quaternion(0.994522, -0.10453, 0.0, 0.0),
      {0.0, 1.0, 0.0}, input.q_arm[7], &output.arm_joint_frames[7]);
  const Transform right_shoulder_roll = body_step(
      right_shoulder_pitch, {-0.0005, -0.0495, 0.0}, quaternion(1.0, 0.0, 7.70015e-05, 0.0),
      {1.0, 0.0, 0.0}, input.q_arm[8], &output.arm_joint_frames[8]);
  const Transform right_shoulder_yaw = body_step(
      right_shoulder_roll, {0.0005, 0.0, -0.1212}, identity_mat3(),
      {0.0, 0.0, 1.0}, input.q_arm[9], &output.arm_joint_frames[9]);
  const Transform right_elbow = body_step(
      right_shoulder_yaw, {0.014, -0.0002, -0.08295}, identity_mat3(),
      {0.0, 1.0, 0.0}, input.q_arm[10], &output.arm_joint_frames[10]);
  const Transform right_wrist_yaw = body_step(
      right_elbow, {-0.014, 9.75261e-05, -0.120694}, quaternion(1.0, 0.0, 0.0, 1.63671e-05),
      {0.0, 0.0, 1.0}, input.q_arm[11], &output.arm_joint_frames[11]);
  const Transform right_wrist_pitch = body_step(
      right_wrist_yaw, {0.0, 0.0, -0.0818}, quaternion(1.0, 0.0, 0.0, -1.63671e-05),
      {0.0, 1.0, 0.0}, input.q_arm[12], &output.arm_joint_frames[12]);
  const Transform right_wrist_roll = body_step(
      right_wrist_pitch, {}, identity_mat3(), {1.0, 0.0, 0.0}, input.q_arm[13],
      &output.arm_joint_frames[13]);

  output.body_transforms[static_cast<std::size_t>(PrimitiveId::Head)] =
      zero_body_step(torso, {0.00834773, 0.0, 0.309397}, identity_mat3()) *
      zero_body_step(identity_transform(), {0.0, 0.0, 0.0889}, identity_mat3());
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::Torso)] = torso;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::LeftThigh)] =
      zero_body_step(pelvis, {0.0, 0.13705, -0.13685}, identity_mat3());
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::RightThigh)] =
      zero_body_step(pelvis, {0.0, -0.13705, -0.13685}, identity_mat3());
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::LeftShoulder)] = left_shoulder_roll;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::LeftForearm)] = left_elbow;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::LeftPalm)] = left_wrist_roll;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::LeftThumb)] = left_wrist_roll;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::RightShoulder)] = right_shoulder_roll;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::RightForearm)] = right_elbow;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::RightPalm)] = right_wrist_roll;
  output.body_transforms[static_cast<std::size_t>(PrimitiveId::RightThumb)] = right_wrist_roll;

  set_sphere(output.primitives[0], output.body_transforms[0], {0.0, -0.007, -0.002}, 0.105, 0.005);
  set_box(output.primitives[1], output.body_transforms[1], {-0.013, 0.0, 0.0675}, identity_mat3(),
          {0.255, 0.285, 0.515}, 0.008);
  set_capsule(output.primitives[2], output.body_transforms[2], {0.0, 0.01, -0.007}, identity_mat3(),
              0.075, 0.27, 0.008);
  set_capsule(output.primitives[3], output.body_transforms[3], {0.0, 0.01, -0.007}, identity_mat3(),
              0.075, 0.27, 0.008);
  set_capsule(output.primitives[4], output.body_transforms[4], {0.0, 0.0, -0.06}, identity_mat3(),
              0.047, 0.12, 0.006);
  set_capsule(output.primitives[5], output.body_transforms[5], {-0.0064, -5.0e-05, -0.1013},
              rpy(0.0, 0.0631, 0.0), 0.055, 0.203, 0.006);
  set_box(output.primitives[6], output.body_transforms[6], {-0.005, 0.0, -0.1357}, identity_mat3(),
          {0.094, 0.064, 0.188}, 0.008);
  output.primitives[7].kind = PrimitiveKind::Capsule;
  output.primitives[7].pose = dynamic_thumb_pose(
      left_wrist_roll, false,
      {input.q_thumb_urdf[0], input.q_thumb_urdf[1]});
  output.primitives[7].radius = 0.029;
  output.primitives[7].length = 0.11;
  output.primitives[7].margin = 0.008;
  set_capsule(output.primitives[8], output.body_transforms[8], {0.0, 0.0, -0.06}, identity_mat3(),
              0.047, 0.12, 0.006);
  set_capsule(output.primitives[9], output.body_transforms[9], {-0.007, 5.0e-05, -0.10125},
              rpy(0.0, 0.069, 0.0), 0.055, 0.203, 0.006);
  set_box(output.primitives[10], output.body_transforms[10], {-0.005, 0.0, -0.1357}, identity_mat3(),
          {0.094, 0.064, 0.188}, 0.008);
  output.primitives[11].kind = PrimitiveKind::Capsule;
  output.primitives[11].pose = dynamic_thumb_pose(
      right_wrist_roll, true,
      {input.q_thumb_urdf[3], input.q_thumb_urdf[4]});
  output.primitives[11].radius = 0.030;
  output.primitives[11].length = 0.11;
  output.primitives[11].margin = 0.008;
  output.valid = input.thumb_pose_valid;
  return output;
}

bool proxy_pair_enabled(const KinematicsOutput& kinematics,
                        std::size_t pair_index) {
  if (pair_index >= kPrimitivePairs.size()) return false;
  const auto installed = [&](PrimitiveId primitive) {
    if (primitive == PrimitiveId::LeftPalm || primitive == PrimitiveId::LeftThumb) {
      return kinematics.hands_installed[0];
    }
    if (primitive == PrimitiveId::RightPalm || primitive == PrimitiveId::RightThumb) {
      return kinematics.hands_installed[1];
    }
    return true;
  };
  const auto pair = kPrimitivePairs[pair_index];
  return installed(pair.first) && installed(pair.second);
}

PrimitiveDistanceReport evaluate_proxy_distances(
    const KinematicsOutput& kinematics) {
  PrimitiveDistanceReport report;
  report.minimum_distance = 1e9;
  for (std::size_t index = 0; index < kPrimitivePairs.size(); ++index) {
    if (!proxy_pair_enabled(kinematics, index)) {
      report.distances[index].signed_distance = 1e9;
      report.distances[index].valid = true;
      continue;
    }
    const PrimitivePair pair = kPrimitivePairs[index];
    report.distances[index] = distance_between(
        kinematics.primitives[static_cast<std::size_t>(pair.first)],
        kinematics.primitives[static_cast<std::size_t>(pair.second)]);
    if (report.distances[index].signed_distance < report.minimum_distance) {
      report.minimum_distance = report.distances[index].signed_distance;
      report.minimum_pair_index = index;
    }
  }
  return report;
}

PrimitiveDistanceReport evaluate_proxy_distances(const KinematicsInput& input) {
  return evaluate_proxy_distances(forward_kinematics(input));
}

}  // namespace x2::cbf
