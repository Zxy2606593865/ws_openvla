import math
from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose, PoseStamped
from tf2_ros import Buffer, TransformException, TransformListener

from openvla_interfaces.msg import OpenVlaAction


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def mat_vec_mul(R, v):
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def euler_to_rot_matrix(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def quat_to_rot_matrix(x: float, y: float, z: float, w: float):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def normalize_quaternion(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return normalize_quaternion([x, y, z, w])


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return normalize_quaternion([x, y, z, w])


class OpenVlaActionAdapter(Node):
    def __init__(self) -> None:
        super().__init__('openvla_action_adapter')

        # topics
        self.declare_parameter('raw_action_topic', '/openvla_action_raw')
        self.declare_parameter('target_pose_topic', '/openvla_target_pose')

        # frame assumptions
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_link', 'link_06')
        self.declare_parameter('camera_frame', 'front_camera/camera_link/rgb_camera')

        # transform source
        self.declare_parameter('transform_mode', 'tf')  # manual / tf
        self.declare_parameter('input_frame_assumption', 'camera')  # camera / base

        # because camera TF is not in tree yet, first use manual camera pose from SDF
        self.declare_parameter('manual_camera_roll', 0.0)
        self.declare_parameter('manual_camera_pitch', 0.90)
        self.declare_parameter('manual_camera_yaw', 2.10)

        # action scaling / limits
        self.declare_parameter('input_already_unnormalized', True)
        self.declare_parameter('scale_xyz', 0.01)
        self.declare_parameter('max_step_xyz', 0.02)

        # rotation/gripper first version: keep interface, disable by default
        self.declare_parameter('use_rotation', False)
        self.declare_parameter('scale_rpy', 0.05)
        self.declare_parameter('max_step_rpy', 0.10)

        self.raw_action_topic = self.get_parameter('raw_action_topic').value
        self.target_pose_topic = self.get_parameter('target_pose_topic').value

        self.base_frame = self.get_parameter('base_frame').value
        self.ee_link = self.get_parameter('ee_link').value
        self.camera_frame = self.get_parameter('camera_frame').value

        self.transform_mode = self.get_parameter('transform_mode').value
        self.input_frame_assumption = self.get_parameter('input_frame_assumption').value

        self.manual_camera_roll = float(self.get_parameter('manual_camera_roll').value)
        self.manual_camera_pitch = float(self.get_parameter('manual_camera_pitch').value)
        self.manual_camera_yaw = float(self.get_parameter('manual_camera_yaw').value)

        self.input_already_unnormalized = bool(self.get_parameter('input_already_unnormalized').value)
        self.scale_xyz = float(self.get_parameter('scale_xyz').value)
        self.max_step_xyz = float(self.get_parameter('max_step_xyz').value)

        self.use_rotation = bool(self.get_parameter('use_rotation').value)
        self.scale_rpy = float(self.get_parameter('scale_rpy').value)
        self.max_step_rpy = float(self.get_parameter('max_step_rpy').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)

        self.sub = self.create_subscription(
            OpenVlaAction,
            self.raw_action_topic,
            self.action_callback,
            10,
        )

        self.get_logger().info(
            f'Adapter started. raw_action_topic={self.raw_action_topic}, '
            f'target_pose_topic={self.target_pose_topic}, '
            f'input_frame_assumption={self.input_frame_assumption}, '
            f'transform_mode={self.transform_mode}, '
            f'input_already_unnormalized={self.input_already_unnormalized}'
        )

    def get_current_ee_pose(self) -> Optional[Pose]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_link,
                Time(),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'Failed to lookup TF {self.base_frame} -> {self.ee_link}: {exc}'
            )
            return None

        pose = Pose()
        pose.position.x = tf_msg.transform.translation.x
        pose.position.y = tf_msg.transform.translation.y
        pose.position.z = tf_msg.transform.translation.z
        pose.orientation = tf_msg.transform.rotation
        return pose

    def get_base_R_input(self):
        if self.input_frame_assumption == 'base':
            return [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]

        if self.transform_mode == 'tf':
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.camera_frame,
                    Time(),
                    timeout=Duration(seconds=1.0),
                )
                q = tf_msg.transform.rotation
                return quat_to_rot_matrix(q.x, q.y, q.z, q.w)
            except TransformException as exc:
                self.get_logger().error(
                    f'Failed to lookup TF {self.base_frame} -> {self.camera_frame}: {exc}'
                )
                return None

        # manual mode: use camera pose from SDF/world as temporary base-frame camera pose
        return euler_to_rot_matrix(
            self.manual_camera_roll,
            self.manual_camera_pitch,
            self.manual_camera_yaw,
        )

    def action_callback(self, msg: OpenVlaAction) -> None:
        current_pose = self.get_current_ee_pose()
        if current_pose is None:
            return

        base_R_input = self.get_base_R_input()
        if base_R_input is None:
            return

        # ---- translation ----
        raw_xyz = [float(msg.dx), float(msg.dy), float(msg.dz)]

        # current assumption: server already unnormalized, but still scale for safe step size
        scaled_xyz = [self.scale_xyz * v for v in raw_xyz]
        delta_base = mat_vec_mul(base_R_input, scaled_xyz)

        delta_base = [
            clamp(delta_base[0], self.max_step_xyz),
            clamp(delta_base[1], self.max_step_xyz),
            clamp(delta_base[2], self.max_step_xyz),
        ]

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.base_frame
        pose_msg.header.stamp = self.get_clock().now().to_msg()

        pose_msg.pose.position.x = current_pose.position.x + delta_base[0]
        pose_msg.pose.position.y = current_pose.position.y + delta_base[1]
        pose_msg.pose.position.z = current_pose.position.z + delta_base[2]

        # ---- rotation ----
        # first version: default keep current orientation
        if not self.use_rotation:
            pose_msg.pose.orientation = current_pose.orientation
        else:
            raw_rpy = [float(msg.d_roll), float(msg.d_pitch), float(msg.d_yaw)]
            scaled_rpy = [self.scale_rpy * v for v in raw_rpy]
            delta_rpy_base = mat_vec_mul(base_R_input, scaled_rpy)

            delta_rpy_base = [
                clamp(delta_rpy_base[0], self.max_step_rpy),
                clamp(delta_rpy_base[1], self.max_step_rpy),
                clamp(delta_rpy_base[2], self.max_step_rpy),
            ]

            q_current = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
            q_delta = euler_to_quaternion(
                delta_rpy_base[0],
                delta_rpy_base[1],
                delta_rpy_base[2],
            )

            # base-frame rotation increment -> left multiply
            q_target = quat_multiply(q_delta, q_current)

            pose_msg.pose.orientation.x = q_target[0]
            pose_msg.pose.orientation.y = q_target[1]
            pose_msg.pose.orientation.z = q_target[2]
            pose_msg.pose.orientation.w = q_target[3]

        self.pose_pub.publish(pose_msg)

        self.get_logger().info(
            'Adapted action -> target pose: '
            f'raw_xyz={[round(v, 4) for v in raw_xyz]}, '
            f'delta_base={[round(v, 4) for v in delta_base]}, '
            f'target_xyz=('
            f'{pose_msg.pose.position.x:.4f}, '
            f'{pose_msg.pose.position.y:.4f}, '
            f'{pose_msg.pose.position.z:.4f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OpenVlaActionAdapter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()