import rclpy
from rclpy.node import Node

from openvla_interfaces.msg import OpenVlaAction


class ActionListener(Node):
    def __init__(self) -> None:
        super().__init__('openvla_action_listener')

        self.subscription = self.create_subscription(
            OpenVlaAction,
            '/openvla_action',
            self.action_callback,
            10,
        )

        self.get_logger().info('Listening on /openvla_action')

    def action_callback(self, msg: OpenVlaAction) -> None:
        self.get_logger().info(
            'Received action: '
            f'dx={msg.dx:.4f}, dy={msg.dy:.4f}, dz={msg.dz:.4f}, '
            f'd_roll={msg.d_roll:.4f}, d_pitch={msg.d_pitch:.4f}, d_yaw={msg.d_yaw:.4f}, '
            f'gripper={msg.gripper:.4f}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActionListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()