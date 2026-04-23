import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from openvla_interfaces.msg import OpenVlaAction

from vlagents.client import RemoteAgent
from vlagents.policies import Obs


class OpenVlaBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('openvla_bridge_node')

        # ---------- Parameters ----------
        self.declare_parameter('server_host', '127.0.0.1')
        self.declare_parameter('server_port', 8080)
        self.declare_parameter('image_topic', '/openvla/front_camera')
        self.declare_parameter('instruction', 'pick up the red cube')
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('jpeg_encoding', False)
        self.declare_parameter('raw_action_topic', '/openvla_action_raw')
        self.declare_parameter('publish_debug_log', True)
        self.declare_parameter('camera_key', 'rgb_side')
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('on_same_machine', False)

        self.server_host = self.get_parameter('server_host').value
        self.server_port = int(self.get_parameter('server_port').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.instruction = self.get_parameter('instruction').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.jpeg_encoding = bool(self.get_parameter('jpeg_encoding').value)
        self.raw_action_topic = self.get_parameter('raw_action_topic').value
        self.publish_debug_log = bool(self.get_parameter('publish_debug_log').value)
        self.camera_key = self.get_parameter('camera_key').value
        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.on_same_machine = bool(self.get_parameter('on_same_machine').value)

        # ---------- ROS ----------
        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.busy = False
        self.has_reset = False
        self.step_count = 0

        self.action_pub = self.create_publisher(
            OpenVlaAction,
            self.raw_action_topic,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # ---------- Remote agent ----------
        self.get_logger().info(
            f'Connecting to OpenVLA server: host={self.server_host}, port={self.server_port}'
        )

        self.agent = RemoteAgent(
            host=self.server_host,
            port=self.server_port,
            model='openvla',
            on_same_machine=self.on_same_machine,
            jpeg_encoding=self.jpeg_encoding,
        )

        period = 1.0 / self.rate_hz if self.rate_hz > 0.0 else 1.0
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'openvla_bridge_node started. image_topic={self.image_topic}, '
            f'raw_action_topic={self.raw_action_topic}, instruction="{self.instruction}"'
        )

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = self.preprocess_image(
                cv_image,
                size=(self.image_width, self.image_height),
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to convert image: {exc}')

    @staticmethod
    def preprocess_image(image: np.ndarray, size=(256, 256)) -> np.ndarray:
        import cv2

        if image is None:
            raise RuntimeError('Input image is None')

        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        arr = np.asarray(resized, dtype=np.uint8)

        expected_shape = (size[1], size[0], 3)
        if arr.shape != expected_shape:
            raise RuntimeError(
                f'Unexpected image shape: {arr.shape}, expected: {expected_shape}'
            )

        return arr

    @staticmethod
    def parse_action(act) -> np.ndarray:
        action = np.asarray(act.action, dtype=np.float32).reshape(-1)

        if action.shape[0] != 7:
            raise RuntimeError(f'Expected 7-dim action, got shape {action.shape}')

        if not np.all(np.isfinite(action)):
            raise RuntimeError(f'Action contains NaN/Inf: {action}')

        return action

    def make_obs(self, image: np.ndarray) -> Obs:
        return Obs(cameras={self.camera_key: image})

    def timer_callback(self) -> None:
        if self.busy:
            self.get_logger().warn('Previous inference still running, skip this tick.')
            return

        if self.latest_image is None:
            self.get_logger().warn('No image received yet.')
            return

        self.busy = True
        t0 = time.time()

        try:
            obs = self.make_obs(self.latest_image)

            if not self.has_reset:
                self.get_logger().info(
                    f'Resetting remote agent with instruction: "{self.instruction}"'
                )
                reset_info = self.agent.reset(obs, self.instruction)
                self.get_logger().info(f'Agent reset done, reset_info={reset_info}')
                self.has_reset = True

            act = self.agent.act(obs)
            action = self.parse_action(act)

            msg = OpenVlaAction()
            msg.dx = float(action[0])
            msg.dy = float(action[1])
            msg.dz = float(action[2])
            msg.d_roll = float(action[3])
            msg.d_pitch = float(action[4])
            msg.d_yaw = float(action[5])
            msg.gripper = float(action[6])
            self.action_pub.publish(msg)

            self.step_count += 1
            dt = time.time() - t0

            if self.publish_debug_log:
                self.get_logger().info(
                    f'step={self.step_count}, action={action.tolist()}, '
                    f'done={getattr(act, "done", None)}, infer_time={dt:.3f}s'
                )

        except Exception as exc:
            self.get_logger().error(f'Inference failed: {exc}')

        finally:
            self.busy = False

    def destroy_node(self) -> None:
        try:
            if hasattr(self, 'agent') and self.agent is not None:
                self.agent.close()
        except Exception as exc:
            self.get_logger().warn(f'Failed to close RemoteAgent cleanly: {exc}')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = OpenVlaBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            rclpy.shutdown()