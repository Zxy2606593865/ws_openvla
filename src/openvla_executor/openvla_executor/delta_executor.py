import math
from threading import Lock

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose
from tf2_ros import Buffer, TransformException, TransformListener

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from openvla_interfaces.msg import OpenVlaAction
from shape_msgs.msg import SolidPrimitive


class DeltaExecutor(Node):
    def __init__(self) -> None:
        super().__init__('openvla_delta_executor')

        self.declare_parameter('planning_group', 'ruka_arm_controller')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_link', 'link_06')
        self.declare_parameter('move_group_action', 'move_action')

        self.declare_parameter('max_step_xyz', 0.03)
        self.declare_parameter('position_tolerance', 0.03)
        self.declare_parameter('orientation_tolerance', 0.05)
        self.declare_parameter('allowed_planning_time', 2.0)
        self.declare_parameter('num_planning_attempts', 1)
        self.declare_parameter('velocity_scaling', 0.2)
        self.declare_parameter('acceleration_scaling', 0.2)

        self.planning_group = self.get_parameter('planning_group').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_link = self.get_parameter('ee_link').value
        self.move_group_action = self.get_parameter('move_group_action').value

        self.max_step_xyz = float(self.get_parameter('max_step_xyz').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.orientation_tolerance = float(self.get_parameter('orientation_tolerance').value)
        self.allowed_planning_time = float(self.get_parameter('allowed_planning_time').value)
        self.num_planning_attempts = int(self.get_parameter('num_planning_attempts').value)
        self.velocity_scaling = float(self.get_parameter('velocity_scaling').value)
        self.acceleration_scaling = float(self.get_parameter('acceleration_scaling').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.move_group_client = ActionClient(self, MoveGroup, self.move_group_action)

        self._busy_lock = Lock()
        self._is_busy = False

        self.subscription = self.create_subscription(
            OpenVlaAction,
            '/openvla_action',
            self.action_callback,
            10,
        )

        self.get_logger().info(
            f'Waiting for action server "{self.move_group_action}"...'
        )
        self.move_group_client.wait_for_server()
        self.get_logger().info(
            f'Ready. group={self.planning_group}, base_frame={self.base_frame}, ee_link={self.ee_link}'
        )
        self.get_logger().info('Listening on /openvla_action')

    def _clamp(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _get_current_ee_pose(self) -> Pose | None:
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

    # def _build_goal_constraints(self, target_pose: Pose) -> Constraints:
    #     constraints = Constraints()
    #     constraints.name = 'openvla_pose_goal'

    #     pos_constraint = PositionConstraint()
    #     pos_constraint.header.frame_id = self.base_frame
    #     pos_constraint.link_name = self.ee_link

    #     region = BoundingVolume()
    #     primitive = SolidPrimitive()
    #     primitive.type = SolidPrimitive.BOX
    #     primitive.dimensions = [
    #         self.position_tolerance,
    #         self.position_tolerance,
    #         self.position_tolerance,
    #     ]
    #     region.primitives.append(primitive)
    #     region.primitive_poses.append(target_pose)

    #     pos_constraint.constraint_region = region
    #     pos_constraint.weight = 1.0

    #     ori_constraint = OrientationConstraint()
    #     ori_constraint.header.frame_id = self.base_frame
    #     ori_constraint.link_name = self.ee_link
    #     ori_constraint.orientation = target_pose.orientation
    #     ori_constraint.absolute_x_axis_tolerance = self.orientation_tolerance
    #     ori_constraint.absolute_y_axis_tolerance = self.orientation_tolerance
    #     ori_constraint.absolute_z_axis_tolerance = self.orientation_tolerance
    #     ori_constraint.weight = 1.0

    #     constraints.position_constraints.append(pos_constraint)
    #     constraints.orientation_constraints.append(ori_constraint)
    #     return constraints


    def _build_goal_constraints(self, target_pose: Pose) -> Constraints:
        constraints = Constraints()
        constraints.name = 'openvla_pose_goal'

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.base_frame
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

    def action_callback(self, msg: OpenVlaAction) -> None:
        if self._is_busy:
            self.get_logger().warn('Executor busy, dropping this action')
            return

        with self._busy_lock:
            self._is_busy = True

        try:
            dx = self._clamp(msg.dx, self.max_step_xyz)
            dy = self._clamp(msg.dy, self.max_step_xyz)
            dz = self._clamp(msg.dz, self.max_step_xyz)

            current_pose = self._get_current_ee_pose()
            if current_pose is None:
                self._is_busy = False
                return

            target_pose = Pose()
            target_pose.position.x = current_pose.position.x + dx
            target_pose.position.y = current_pose.position.y + dy
            target_pose.position.z = current_pose.position.z + dz

            # 第一版先保持当前末端姿态不变
            target_pose.orientation = current_pose.orientation

            self.get_logger().info(
                'Sending pose goal: '
                f'x={target_pose.position.x:.4f}, '
                f'y={target_pose.position.y:.4f}, '
                f'z={target_pose.position.z:.4f}'
            )

            request = MotionPlanRequest()
            request.group_name = self.planning_group
            request.num_planning_attempts = self.num_planning_attempts
            request.allowed_planning_time = self.allowed_planning_time
            request.max_velocity_scaling_factor = self.velocity_scaling
            request.max_acceleration_scaling_factor = self.acceleration_scaling
            request.start_state.is_diff = True
            request.goal_constraints.append(self._build_goal_constraints(target_pose))

            planning_options = PlanningOptions()
            planning_options.plan_only = False
            planning_options.look_around = False
            planning_options.replan = False

            goal_msg = MoveGroup.Goal()
            goal_msg.request = request
            goal_msg.planning_options = planning_options

            send_goal_future = self.move_group_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(self._goal_response_callback)

        except Exception as exc:
            self.get_logger().error(f'Executor exception: {exc}')
            self._is_busy = False

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to send MoveGroup goal: {exc}')
            self._is_busy = False
            return

        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal rejected')
            self._is_busy = False
            return

        self.get_logger().info('MoveGroup goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
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
            self._is_busy = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeltaExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()