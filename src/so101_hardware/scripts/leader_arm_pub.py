#!/usr/bin/env python
import rclpy
from ibrobot_msgs.msg import JointCurrent
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from so101_hardware.calibration.interactive import load_calibration
from so101_hardware.calibration.constants import LEADER_CALIB_FILE, DEFAULT_SERIAL_PORT, JOINT_NAMES
from so101_hardware.motor_current import read_motor_currents
import time
import math
import pathlib

class LeaderArmPublisher(Node):
    def __init__(self):
        super().__init__('so101_leader_publisher')

        self.declare_parameter('port', DEFAULT_SERIAL_PORT)
        self.declare_parameter('calib_file', '')
        self.declare_parameter('publish_rate', 50.0)

        self.port = self.get_parameter('port').get_parameter_value().string_value
        calib_file_param = self.get_parameter('calib_file').get_parameter_value().string_value
        self.calib_file = pathlib.Path(calib_file_param) if calib_file_param else LEADER_CALIB_FILE
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.get_logger().info(f"Leader Publisher: Using port: {self.port}")
        self.get_logger().info(f"Leader Publisher: Calibration file: {self.calib_file}")

        self._last_error_stack_printed = False
        self._is_functional = True

        self.leader_state_pub = self.create_publisher(
            JointState,
            '/so101_leader/joint_states',
            10
        )
        self.leader_current_pub = self.create_publisher(
            JointCurrent,
            '/so101_leader/joint_currents',
            10
        )

        # 舵机定义
        self.joints = {
            "1": {"id": 1, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "2": {"id": 2, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "3": {"id": 3, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "4": {"id": 4, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
            "5": {"id": 5, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100}, # "5" 是全周电机 (Wrist Roll)
            "6": {"id": 6, "model": "sts3215", "mode": MotorNormMode.RANGE_0_100},
        }
        self.joint_names = list(self.joints.keys()) # ["1", "2", "3", "4", "5", "6"]

        self._home_offsets: dict[str, int] | None = None
        self._rad_per_step = (2 * math.pi) / 4096.0 # 转换系数
        # --- 存储 Leader "Home" 姿态的真实弧度偏移量 ---
        self._reset_rad_offsets: dict[str, float] = {}
        self.calibration_data: dict[str, MotorCalibration] | None = None
        self.bus_ = None

        # 连接总线
        try:
            motors = {
                name: Motor(params["id"], params["model"], params["mode"])
                for name, params in self.joints.items()
            }
            self.bus_ = FeetechMotorsBus(self.port, motors)
            self.bus_.connect()
            self.get_logger().info(f"Connected to motor bus on {self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to bus on {self.port}: {e}")
            self.bus_ = None
            rclpy.shutdown()
            return

        if not self._load_or_run_calibration():
            # 如果校准失败 (无论是自动加载还是手动), 都关闭节点
            self.get_logger().error("校准/加载失败。正在关闭...")
            # 标记为无效，防止 main() 中的 finally 再次尝试断开已关闭的连接
            bus_to_close = self.bus_
            self.bus_ = None 
            if bus_to_close:
                bus_to_close.disconnect()
            rclpy.shutdown()
            return

        # 从 MotorCalibration 对象中提取 home offsets
        self._home_offsets = {name: cal.homing_offset for name, cal in self.calibration_data.items()}
        self.get_logger().info(f"Leader 臂已加载 home offsets: {self._home_offsets}")

        # 计算 "Home" 位置的真实弧度偏移量
        self._calculate_reset_offsets()

        # 创建定时器
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"✅ Leader arm publisher 启动成功 (publish rate: {self.publish_rate} Hz), 正在读取舵机...")

    # 计算复位偏移量的方法
    def _calculate_reset_offsets(self):
        """
        使用【硬编码的原始步数】('reset_raw_pose')
        来计算 'Home' 姿态的真实弧度偏移量。
        """
        self.get_logger().info("正在计算 Leader 臂的复位弧度偏移量...")
        if self.calibration_data is None:
            self.get_logger().error("无法计算偏移量: 校准数据未加载。")
            return

        reset_raw_pose = {
            '1': 2060,
            '2': 884,
            '3': 3095,
            '4': 903,
            '5': 879,
            '6': 1595
        }
        self.get_logger().info(f"使用硬编码的 'Home' 姿态 (原始步数): {reset_raw_pose}")

        for name in self.joint_names:
            if name not in self.calibration_data or name not in reset_raw_pose:
                self.get_logger().warn(f"跳过关节 {name}: 在校准或'reset_raw_pose'中未找到。")
                self._reset_rad_offsets[name] = 0.0 # 默认为 0 偏移
                continue

            cal = self.calibration_data[name]
            home = cal.homing_offset
            # 直接使用正确的原始步数
            reset_raw = reset_raw_pose[name]

            # (真实弧度) = (原始步数 - 零点步数) * 弧度/步数
            reset_rad = (reset_raw - home) * self._rad_per_step
            self._reset_rad_offsets[name] = reset_rad

        self.get_logger().info("成功计算了 Leader 臂的复位弧度偏移量:")
        for name, offset in self._reset_rad_offsets.items():
             self.get_logger().info(f"  - 关节 {name}: {offset:.2f} rad")

    def _load_or_run_calibration(self) -> bool:
        """Load calibration from file."""
        if self.calib_file.is_file():
            self.get_logger().info(f"Found calibration file: {self.calib_file}. Auto-loading.")
            try:
                self.calibration_data = load_calibration(self.calib_file, self.joint_names, self.get_logger())
                self.get_logger().info("Writing loaded calibration to motor firmware...")
                self.bus_.write_calibration(self.calibration_data)
                self.get_logger().info("Calibration written to motors.")
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to load/write calibration ({e}).")
                self.get_logger().error("Run calibration tool:")
                self.get_logger().error(f"  ros2 run so101_hardware calibrate_arm --arm leader --port {self.port}")
                return False
        else:
            self.get_logger().error(f"Calibration file not found: {self.calib_file}")
            self.get_logger().error("Run calibration tool first:")
            self.get_logger().error(f"  ros2 run so101_hardware calibrate_arm --arm leader --port {self.port}")
            return False

    def timer_callback(self):
        # 如果总线未连接，直接退出
        if self.bus_ is None or not self.bus_.is_connected:
            self.get_logger().error("总线连接已断开，正在退出...")
            rclpy.shutdown()
            return

        try:
            # 正常读取流程
            raw_positions = self.bus_.sync_read("Present_Position", normalize=False)

            if not raw_positions or len(raw_positions) != 6:
                self.get_logger().warn(f"读取数据不全: {raw_positions}", throttle_duration_sec=5.0)
                return

            currents = read_motor_currents(
                self.bus_,
                self.joint_names,
                self.get_logger(),
                "读取电流失败",
            )

            positions_rad = []
            for n in self.joint_names:
                raw = raw_positions.get(n, 0)
                home = self._home_offsets.get(n, 0) # 现在这是来自校准的真实零点

                # 1. 计算真实弧度
                true_rad = (raw - home) * self._rad_per_step

                # 2. 获取此关节的 "Home" 弧度偏移量
                reset_offset = self._reset_rad_offsets.get(n, 0.0)

                # 3. 计算相对弧度 (真实弧度 - 偏移量)
                relative_rad = true_rad - reset_offset

                # 4. 添加相对弧度
                positions_rad.append(relative_rad)

            # 5. 只发布 Leader 自己的状态
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = positions_rad

            self.leader_state_pub.publish(joint_state_msg)

            current_msg = JointCurrent()
            current_msg.header.stamp = joint_state_msg.header.stamp
            current_msg.name = list(self.joint_names)
            current_msg.current = [currents.get(name, 0.0) for name in self.joint_names]
            self.leader_current_pub.publish(current_msg)

        except (IOError, OSError, ConnectionError) as e:
            # 捕获已知通讯异常并退出
            self.get_logger().error("======================================================")
            self.get_logger().error(f"通讯中断: {e}")
            self.get_logger().error("可能是 USB 线已拔掉或串口异常。")
            self.get_logger().error("请重新连接机械臂，并确保已赋予权限:")
            self.get_logger().error(f"  sudo chmod 666 {self.port}")
            self.get_logger().error("======================================================")
            
            # 尝试清理并关闭
            if self.bus_:
                try:
                    self.bus_.port_handler.closePort()
                except Exception:
                    pass
            
            # 退出节点
            rclpy.shutdown()
        except Exception as e:
            # 捕获未知异常
            self.get_logger().error(f"发生未知读取错误: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = None # 预先声明
    try:
        node = LeaderArmPublisher()
        # 只有在 __init__ 成功 (即校准成功) 且 bus 有效时才 spin
        if node.bus_ is not None and rclpy.ok() and node._home_offsets is not None:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node and node.bus_ is not None:
            node.get_logger().info("正在断开总线连接...")
            # 在断开连接前禁用扭矩
            try:
                node.bus_.disable_torque()
                node.get_logger().info("电机扭矩已禁用。")
            except Exception:
                pass
            
            try:
                node.bus_.disconnect()
            except Exception:
                pass
        if node and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
