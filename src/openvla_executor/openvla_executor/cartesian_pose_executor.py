import math
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath


class OpenVlaCartesianPoseExecutor(Node):
    """
    Cartesian executor for OpenVLA/RUKA small-step control.

    Input:
      /openvla_target_pose : geometry_msgs/msg/PoseStamped

    Output:
      /openvla_executor_busy : std_msgs/msg/Bool

    Core idea:
      - receive target pose from openvla_adapter
      - ask MoveIt to compute a Cartesian path
      - execute the returned trajectory
      - publish busy=True during execution, busy=False after done

    This executor is intended for:
      - keyboard teleoperation
      - clean small-step data collection
      - base_link dx/dy/dz validation

    It is better than the old position-only pose_executor for this stage because
    Cartesian path planning tends to preserve end-effector pose continuity better.
    """

    def __init__(self) -> None:
        super().__init__("openvla_cartesian_pose_executor")

        # Topics
        self.declare_parameter("target_pose_topic", "/openvla_target_pose")
        self.declare_parameter("executor_busy_topic", "/openvla_executor_busy")

        # MoveIt interfaces
        self.declare_parameter("cartesian_service", "/compute_cartesian_path")
        self.declare_parameter("execute_trajectory_action", "/execute_trajectory")

        # Robot config
        self.declare_parameter("planning_group", "ruka_arm_controller")
        self.declare_parameter("ee_link", "link_06")

        # Cartesian path config
        # max_step: Cartesian interpolation resolution in meters.
        # 0.002 = 2 mm interpolation, suitable for 5 mm / 1 cm small steps.
        self.declare_parameter("max_step", 0.002)

        # jump_threshold = 0.0 usually means disabling jump threshold check.
        # For first validation this is more tolerant.
        self.declare_parameter("jump_threshold", 0.0)

        # Minimum acceptable Cartesian path completion ratio.
        self.declare_parameter("min_fraction", 0.95)

        # Collision checking.
        # For current simulation validation, false is more tolerant.
        # Later, after planning scene is configured, you can set this true.
        self.declare_parameter("avoid_collisions", False)

        # Workspace boundary for stable red-cube approach demo.
        # Red cube is around x=0.38, y=-0.12, z=0.02 in world/base_link.
        # End-effector should approach above the cube, not collide with floor/cube.
        self.declare_parameter("x_min", 0.02)
        self.declare_parameter("x_max", 0.48)
        self.declare_parameter("y_min", -0.22)
        self.declare_parameter("y_max", 0.10)
        self.declare_parameter("z_min", 0.07)
        self.declare_parameter("z_max", 0.60)

        self.declare_parameter("max_consecutive_failures", 3)

        # Velocity / acceleration scaling.
        # Some MoveIt versions expose these fields in GetCartesianPath.
        # The code sets them only if the fields exist.
        self.declare_parameter("max_velocity_scaling_factor", 0.10)
        self.declare_parameter("max_acceleration_scaling_factor", 0.10)

        # If the returned trajectory has no useful timing, this executor assigns
        # simple increasing timestamps to joint trajectory points.
        self.declare_parameter("point_time_step", 0.25)

        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.executor_busy_topic = str(self.get_parameter("executor_busy_topic").value)

        self.cartesian_service = str(self.get_parameter("cartesian_service").value)
        self.execute_trajectory_action = str(
            self.get_parameter("execute_trajectory_action").value
        )

        self.planning_group = str(self.get_parameter("planning_group").value)
        self.ee_link = str(self.get_parameter("ee_link").value)

        self.max_step = float(self.get_parameter("max_step").value)
        self.jump_threshold = float(self.get_parameter("jump_threshold").value)
        self.min_fraction = float(self.get_parameter("min_fraction").value)
        self.avoid_collisions = bool(self.get_parameter("avoid_collisions").value)

        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)

        self.max_consecutive_failures = int(
            self.get_parameter("max_consecutive_failures").value
        )
        self.consecutive_failures = 0

        self.max_velocity_scaling_factor = float(
            self.get_parameter("max_velocity_scaling_factor").value
        )
        self.max_acceleration_scaling_factor = float(
            self.get_parameter("max_acceleration_scaling_factor").value
        )

        self.point_time_step = float(self.get_parameter("point_time_step").value)

        self._busy = False

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.busy_pub = self.create_publisher(
            Bool,
            self.executor_busy_topic,
            status_qos,
        )

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            self.cartesian_service,
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            self.execute_trajectory_action,
        )

        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            self.target_pose_topic,
            self.pose_callback,
            10,
        )

        self.publish_busy(False)

        self.get_logger().info(
            f'Waiting for Cartesian path service "{self.cartesian_service}"...'
        )
        self.cartesian_client.wait_for_service()

        self.get_logger().info(
            f'Waiting for execute trajectory action "{self.execute_trajectory_action}"...'
        )
        self.execute_client.wait_for_server()

        self.get_logger().info(
            "Ready. "
            f"target_pose_topic={self.target_pose_topic}, "
            f"executor_busy_topic={self.executor_busy_topic}, "
            f"planning_group={self.planning_group}, "
            f"ee_link={self.ee_link}, "
            f"cartesian_service={self.cartesian_service}, "
            f"execute_trajectory_action={self.execute_trajectory_action}, "
            f"max_step={self.max_step}, "
            f"jump_threshold={self.jump_threshold}, "
            f"min_fraction={self.min_fraction}, "
            f"avoid_collisions={self.avoid_collisions}, "
            f"max_velocity_scaling_factor={self.max_velocity_scaling_factor}, "
            f"max_acceleration_scaling_factor={self.max_acceleration_scaling_factor}, "
            f"point_time_step={self.point_time_step}"
        )

    def publish_busy(self, busy: bool) -> None:
        msg = Bool()
        msg.data = bool(busy)
        self.busy_pub.publish(msg)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.publish_busy(self._busy)

    def workspace_ok(self, msg: PoseStamped) -> bool:
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        ok = (
            self.x_min <= x <= self.x_max and
            self.y_min <= y <= self.y_max and
            self.z_min <= z <= self.z_max
        )

        if not ok:
            self.get_logger().warn(
                "[ACTION_REJECTED] target outside workspace: "
                f"x={x:.4f}, y={y:.4f}, z={z:.4f}, "
                f"allowed x=[{self.x_min:.2f}, {self.x_max:.2f}], "
                f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}]"
            )

        return ok


    def mark_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.get_logger().warn(
            f"[EXECUTION_FAILURE] {reason}; "
            f"consecutive_failures={self.consecutive_failures}/"
            f"{self.max_consecutive_failures}"
        )

        if self.consecutive_failures >= self.max_consecutive_failures:
            self.get_logger().warn(
                "[RECOVERY_REQUIRED] too many failed/rejected actions. "
                "Return to openvla_ready before continuing."
            )
            self.consecutive_failures = 0


    def mark_success(self) -> None:
        if self.consecutive_failures != 0:
            self.get_logger().info(
                f"Reset consecutive_failures: {self.consecutive_failures} -> 0"
            )
        self.consecutive_failures = 0

    @staticmethod
    def _set_if_exists(obj, field_name: str, value) -> bool:
        if hasattr(obj, field_name):
            setattr(obj, field_name, value)
            return True
        return False

    @staticmethod
    def _duration_from_seconds(seconds: float) -> Duration:
        seconds = max(0.0, float(seconds))
        sec = int(math.floor(seconds))
        nanosec = int(round((seconds - sec) * 1_000_000_000))

        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000

        return Duration(sec=sec, nanosec=nanosec)

    def _ensure_trajectory_timing(self, trajectory: RobotTrajectory) -> RobotTrajectory:
        points = trajectory.joint_trajectory.points

        if not points:
            return trajectory

        needs_timing = False
        previous_time = -1.0

        for point in points:
            t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            if t <= previous_time:
                needs_timing = True
                break
            previous_time = t

        if not needs_timing:
            return trajectory

        self.get_logger().warn(
            "Returned Cartesian trajectory has invalid or missing timing; "
            "assigning simple time_from_start values."
        )

        for i, point in enumerate(points):
            point.time_from_start = self._duration_from_seconds(
                (i + 1) * self.point_time_step
            )

        return trajectory

    def _build_cartesian_request(self, msg: PoseStamped) -> GetCartesianPath.Request:
        request = GetCartesianPath.Request()

        request.header.frame_id = msg.header.frame_id
        request.group_name = self.planning_group
        request.link_name = self.ee_link

        # Start from the current robot state.
        request.start_state.is_diff = True

        # One target waypoint is enough for a single small step.
        # The target pose should already contain the intended orientation from adapter.
        request.waypoints = [msg.pose]

        request.max_step = self.max_step
        request.jump_threshold = self.jump_threshold
        request.avoid_collisions = self.avoid_collisions

        # These fields exist in newer MoveIt 2 / moveit_msgs versions.
        # Use guarded assignment to stay compatible.
        self._set_if_exists(
            request,
            "max_velocity_scaling_factor",
            self.max_velocity_scaling_factor,
        )
        self._set_if_exists(
            request,
            "max_acceleration_scaling_factor",
            self.max_acceleration_scaling_factor,
        )

        return request

    def pose_callback(self, msg: PoseStamped) -> None:
        if self._busy:
            self.get_logger().warn("Executor busy, dropping target pose")
            self.publish_busy(True)
            return

        if not msg.header.frame_id:
            self.get_logger().error("Received target pose with empty frame_id")
            return

        self.set_busy(True)
        self.get_logger().info(
            "Received target pose: "
            f"frame={msg.header.frame_id}, "
            f"x={msg.pose.position.x:.4f}, "
            f"y={msg.pose.position.y:.4f}, "
            f"z={msg.pose.position.z:.4f}, "
            f"qx={msg.pose.orientation.x:.4f}, "
            f"qy={msg.pose.orientation.y:.4f}, "
            f"qz={msg.pose.orientation.z:.4f}, "
            f"qw={msg.pose.orientation.w:.4f}"
        )

        if not self.workspace_ok(msg):
            self.mark_failure("target outside workspace")
            self.set_busy(False)
            return

        request = self._build_cartesian_request(msg)
        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self._cartesian_response_callback)

    def _cartesian_response_callback(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"GetCartesianPath service call failed: {exc}")
            self.set_busy(False)
            return

        error_val = response.error_code.val
        fraction = float(response.fraction)

        self.get_logger().info(
            f"Cartesian path response: fraction={fraction:.3f}, "
            f"error_code={error_val}"
        )

        if error_val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Cartesian path planning failed, error_code={error_val}"
            )
            self.mark_failure(f"cartesian planning failed, error_code={error_val}")
            self.set_busy(False)
            return

        if fraction < self.min_fraction:
            self.get_logger().error(
                f"Cartesian path fraction too low: "
                f"{fraction:.3f} < {self.min_fraction:.3f}"
            )
            self.mark_failure(
                f"cartesian fraction too low: {fraction:.3f} < {self.min_fraction:.3f}"
            )
            self.set_busy(False)
            return

        trajectory = response.solution

        if not trajectory.joint_trajectory.points:
            self.get_logger().error("Cartesian path returned empty trajectory")
            self.mark_failure("empty cartesian trajectory")
            self.set_busy(False)
            return

        trajectory = self._ensure_trajectory_timing(trajectory)

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        send_goal_future = self.execute_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._execute_goal_response_callback)

    def _execute_goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to send ExecuteTrajectory goal: {exc}")
            self.set_busy(False)
            return

        if not goal_handle.accepted:
            self.get_logger().error("ExecuteTrajectory goal rejected")
            self.mark_failure("execute trajectory goal rejected")
            self.set_busy(False)
            return

        self.get_logger().info("ExecuteTrajectory goal accepted")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._execute_result_callback)

    def _execute_result_callback(self, future) -> None:
        try:
            result = future.result().result
            error_val = result.error_code.val

            if error_val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info("ExecuteTrajectory succeeded")
                self.mark_success()
            else:
                self.get_logger().error(
                    f"ExecuteTrajectory failed, error_code={error_val}"
                )
                self.mark_failure(f"execute trajectory failed, error_code={error_val}")
        except Exception as exc:
            self.get_logger().error(f"Failed to get ExecuteTrajectory result: {exc}")
        finally:
            self.set_busy(False)


def main(args=None) -> None:
    rclpy.init(args=args)

    node: Optional[OpenVlaCartesianPoseExecutor] = None

    try:
        node = OpenVlaCartesianPoseExecutor()
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