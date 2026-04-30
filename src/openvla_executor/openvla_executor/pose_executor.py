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
    OrientationConstraint,
)
from shape_msgs.msg import SolidPrimitive


class OpenVlaPoseExecutor(Node):
    """
    Execute /openvla_target_pose through MoveIt MoveGroup.

    Default mode:
      position-only constraint
      suitable for current RUKA base_link dx/dy/dz validation.

    Optional mode:
      orientation constraint can be enabled with:
        -p use_orientation_constraint:=true

    Optional planner selection:
      pipeline_id / planner_id can be set later for Pilz LIN testing.
    """

    def __init__(self) -> None:
        super().__init__("openvla_pose_executor")

        # Topics / MoveGroup action
        self.declare_parameter("target_pose_topic", "/openvla_target_pose")
        self.declare_parameter("executor_busy_topic", "/openvla_executor_busy")
        self.declare_parameter("move_group_action", "move_action")

        # MoveIt group / end-effector
        self.declare_parameter("planning_group", "ruka_arm_controller")
        self.declare_parameter("ee_link", "link_06")

        # Optional planner selection.
        # Default empty string means using MoveIt's default pipeline/planner.
        #
        # Example for later testing:
        #   pipeline_id = pilz_industrial_motion_planner
        #   planner_id = LIN
        self.declare_parameter("pipeline_id", "")
        self.declare_parameter("planner_id", "")

        # Position tolerance is treated as +/- tolerance in meters.
        # Example:
        #   position_tolerance = 0.005
        # means the target box side length is 0.010 m.
        self.declare_parameter("position_tolerance", 0.005)

        # Orientation constraint is disabled by default.
        # Hard orientation constraints made OMPL fail in your current setup.
        self.declare_parameter("use_orientation_constraint", False)

        # Used only when use_orientation_constraint=True.
        # Start loose when testing, for example 0.50 rad.
        self.declare_parameter("orientation_tolerance", 0.50)

        # Planning behavior
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("acceleration_scaling", 0.1)

        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.executor_busy_topic = str(self.get_parameter("executor_busy_topic").value)
        self.move_group_action = str(self.get_parameter("move_group_action").value)

        self.planning_group = str(self.get_parameter("planning_group").value)
        self.ee_link = str(self.get_parameter("ee_link").value)

        self.pipeline_id = str(self.get_parameter("pipeline_id").value)
        self.planner_id = str(self.get_parameter("planner_id").value)

        self.position_tolerance = float(
            self.get_parameter("position_tolerance").value
        )
        self.use_orientation_constraint = bool(
            self.get_parameter("use_orientation_constraint").value
        )
        self.orientation_tolerance = float(
            self.get_parameter("orientation_tolerance").value
        )

        self.allowed_planning_time = float(
            self.get_parameter("allowed_planning_time").value
        )
        self.num_planning_attempts = int(
            self.get_parameter("num_planning_attempts").value
        )
        self.velocity_scaling = float(
            self.get_parameter("velocity_scaling").value
        )
        self.acceleration_scaling = float(
            self.get_parameter("acceleration_scaling").value
        )

        self._busy = False

        # Use transient local QoS so late-started nodes can still read latest busy state.
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

        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            self.target_pose_topic,
            self.pose_callback,
            10,
        )

        self.publish_busy(False)

        self.get_logger().info(
            f'Waiting for action server "{self.move_group_action}"...'
        )
        self.move_group_client.wait_for_server()

        self.get_logger().info(
            "Ready. "
            f"target_pose_topic={self.target_pose_topic}, "
            f"executor_busy_topic={self.executor_busy_topic}, "
            f"group={self.planning_group}, "
            f"ee_link={self.ee_link}, "
            f"pipeline_id={self.pipeline_id or '<default>'}, "
            f"planner_id={self.planner_id or '<default>'}, "
            f"position_tolerance=+/-{self.position_tolerance}, "
            f"use_orientation_constraint={self.use_orientation_constraint}, "
            f"orientation_tolerance={self.orientation_tolerance}, "
            f"velocity_scaling={self.velocity_scaling}, "
            f"acceleration_scaling={self.acceleration_scaling}"
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
        constraints.name = "openvla_target_constraints"

        # -------------------------
        # Position constraint
        # -------------------------
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = frame_id
        pos_constraint.link_name = self.ee_link

        region = BoundingVolume()

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX

        # MoveIt expects full box dimensions, not +/- tolerance.
        # Therefore:
        #   tolerance 0.005 m -> box side length 0.010 m.
        box_size = max(2.0 * self.position_tolerance, 1e-4)
        primitive.dimensions = [box_size, box_size, box_size]

        region.primitives.append(primitive)
        region.primitive_poses.append(target_pose)

        pos_constraint.constraint_region = region
        pos_constraint.weight = 1.0

        constraints.position_constraints.append(pos_constraint)

        # -------------------------
        # Optional orientation constraint
        # -------------------------
        if self.use_orientation_constraint:
            ori_constraint = OrientationConstraint()
            ori_constraint.header.frame_id = frame_id
            ori_constraint.link_name = self.ee_link
            ori_constraint.orientation = target_pose.orientation
            ori_constraint.absolute_x_axis_tolerance = self.orientation_tolerance
            ori_constraint.absolute_y_axis_tolerance = self.orientation_tolerance
            ori_constraint.absolute_z_axis_tolerance = self.orientation_tolerance
            ori_constraint.weight = 1.0

            constraints.orientation_constraints.append(ori_constraint)

        return constraints

    def pose_callback(self, msg: PoseStamped) -> None:
        if self._busy:
            self.get_logger().warn("Executor busy, dropping target pose")
            self.publish_busy(True)
            return

        self.set_busy(True)

        self.get_logger().info(
            "Sending target pose: "
            f"frame={msg.header.frame_id}, "
            f"x={msg.pose.position.x:.4f}, "
            f"y={msg.pose.position.y:.4f}, "
            f"z={msg.pose.position.z:.4f}, "
            f"qx={msg.pose.orientation.x:.4f}, "
            f"qy={msg.pose.orientation.y:.4f}, "
            f"qz={msg.pose.orientation.z:.4f}, "
            f"qw={msg.pose.orientation.w:.4f}"
        )

        request = MotionPlanRequest()
        request.group_name = self.planning_group

        if self.pipeline_id:
            request.pipeline_id = self.pipeline_id

        if self.planner_id:
            request.planner_id = self.planner_id

        request.num_planning_attempts = self.num_planning_attempts
        request.allowed_planning_time = self.allowed_planning_time
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling

        # Use current robot state as start state.
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
            self.get_logger().error(f"Failed to send MoveGroup goal: {exc}")
            self.set_busy(False)
            return

        if not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal rejected")
            self.set_busy(False)
            return

        self.get_logger().info("MoveGroup goal accepted")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future) -> None:
        try:
            action_result = future.result().result
            error_val = action_result.error_code.val

            if error_val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info("MoveGroup execution succeeded")
            else:
                self.get_logger().error(
                    f"MoveGroup execution failed, error_code={error_val}"
                )
        except Exception as exc:
            self.get_logger().error(f"Failed to get MoveGroup result: {exc}")
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


if __name__ == "__main__":
    main()