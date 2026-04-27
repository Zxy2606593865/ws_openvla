import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Bool

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive


class OpenVlaPoseExecutor(Node):
    def __init__(self) -> None:
        super().__init__('openvla_pose_executor')

        self.declare_parameter('target_pose_topic', '/openvla_target_pose')
        self.declare_parameter('executor_busy_topic', '/openvla_executor_busy')

        self.declare_parameter('move_group_action', 'move_action')
        self.declare_parameter('planning_group', 'ruka_arm_controller')
        self.declare_parameter('ee_link', 'link_06')

        # 这里只保留位置目标，先把位置链调通
        self.declare_parameter('position_tolerance', 0.06)
        self.declare_parameter('allowed_planning_time', 5.0)
        self.declare_parameter('num_planning_attempts', 5)
        self.declare_parameter('velocity_scaling', 0.2)
        self.declare_parameter('acceleration_scaling', 0.2)

        self.target_pose_topic = self.get_parameter('target_pose_topic').value
        self.executor_busy_topic = self.get_parameter('executor_busy_topic').value

        self.move_group_action = self.get_parameter('move_group_action').value
        self.planning_group = self.get_parameter('planning_group').value
        self.ee_link = self.get_parameter('ee_link').value

        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.allowed_planning_time = float(self.get_parameter('allowed_planning_time').value)
        self.num_planning_attempts = int(self.get_parameter('num_planning_attempts').value)
        self.velocity_scaling = float(self.get_parameter('velocity_scaling').value)
        self.acceleration_scaling = float(self.get_parameter('acceleration_scaling').value)

        self._busy = False

        # 用 transient_local，让后启动的 bridge 也能拿到最新 busy 状态
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.busy_pub = self.create_publisher(
            Bool,
            self.executor_busy_topic,
            status_qos,
        )

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            self.move_group_action,
        )

        self.sub = self.create_subscription(
            PoseStamped,
            self.target_pose_topic,
            self.pose_callback,
            10,
        )

        self.publish_busy(False)

        self.get_logger().info(f'Waiting for action server "{self.move_group_action}"...')
        self.move_group_client.wait_for_server()

        self.get_logger().info(
            f'Ready. target_pose_topic={self.target_pose_topic}, '
            f'executor_busy_topic={self.executor_busy_topic}, '
            f'group={self.planning_group}, ee_link={self.ee_link}'
        )

    def publish_busy(self, busy: bool) -> None:
        msg = Bool()
        msg.data = bool(busy)
        self.busy_pub.publish(msg)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.publish_busy(self._busy)

    def _build_goal_constraints(self, target_pose: Pose, frame_id: str) -> Constraints:
        constraints = Constraints()
        constraints.name = 'openvla_target_position_goal'

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = frame_id
        pos_constraint.link_name = self.ee_link

        region = BoundingVolume()

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            self.position_tolerance,
            self.position_tolerance,
            self.position_tolerance,
        ]

        region.primitives.append(primitive)
        region.primitive_poses.append(target_pose)

        pos_constraint.constraint_region = region
        pos_constraint.weight = 1.0

        constraints.position_constraints.append(pos_constraint)
        return constraints

    def pose_callback(self, msg: PoseStamped) -> None:
        if self._busy:
            self.get_logger().warn('Executor busy, dropping target pose')
            self.publish_busy(True)
            return

        self.set_busy(True)

        self.get_logger().info(
            'Sending target pose: '
            f'frame={msg.header.frame_id}, '
            f'x={msg.pose.position.x:.4f}, '
            f'y={msg.pose.position.y:.4f}, '
            f'z={msg.pose.position.z:.4f}'
        )

        request = MotionPlanRequest()
        request.group_name = self.planning_group
        request.num_planning_attempts = self.num_planning_attempts
        request.allowed_planning_time = self.allowed_planning_time
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling
        request.start_state.is_diff = True

        request.goal_constraints.append(
            self._build_goal_constraints(msg.pose, msg.header.frame_id)
        )

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        planning_options.look_around = False
        planning_options.replan = False

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options = planning_options

        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to send MoveGroup goal: {exc}')
            self.set_busy(False)
            return

        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal rejected')
            self.set_busy(False)
            return

        self.get_logger().info('MoveGroup goal accepted')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future) -> None:
        try:
            action_result = future.result().result
            error_val = action_result.error_code.val

            if error_val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info('MoveGroup execution succeeded')
            else:
                self.get_logger().error(
                    f'MoveGroup execution failed, error_code={error_val}'
                )

        except Exception as exc:
            self.get_logger().error(f'Failed to get MoveGroup result: {exc}')

        finally:
            self.set_busy(False)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = OpenVlaPoseExecutor()
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