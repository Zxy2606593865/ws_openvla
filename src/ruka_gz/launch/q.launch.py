from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default=False)
    package_name = 'ruka_gz'

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="moveit.rviz",
        description="RViz configuration file",
    )

    # sim_time = DeclareLaunchArgument(
    #    "use_sim_time",
    #     default_value= "True",
    #     description="RViz configuration file",
    # )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [FindPackageShare('ruka_gz'),
                 'config', 'ruka_gz.urdf.xacro']
            ),
        ]
    )
    robot_description = {'robot_description': robot_description_content}

    moveit_config = (
        MoveItConfigsBuilder("ruka_gz", package_name="ruka_gz")
    
        .robot_description_semantic(file_path="config/ruka_gz.srdf")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["chomp", "pilz_industrial_motion_planner", "ompl"]
        )
        # .moveit_config_dict.update({'use_sim_time' : True})
        .to_moveit_configs()
    )


    move_group_path = os.path.join(get_package_share_directory(package_name), 'config', 'move_group.yaml')

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(),
                    # moveit_config.update,
                    move_group_path],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz
    rviz_base = LaunchConfiguration("rviz_config")
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("ruka_gz"), "config", rviz_base]
    )

    # rviz_config = PathJoinSubstitution(
    #     [FindPackageShare("ruka"), "config", "moveit.rviz"]
    # )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            # moveit_config.robot_description,
            robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            move_group_path
        ],

    )



    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )
   
    
    camera_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_static_transform_publisher",
        output="log",
        arguments=[
            "--x", "0.85",
            "--y", "-0.65",
            "--z", "0.95",
            "--yaw", "2.10",
            "--pitch", "0.90",
            "--roll", "0.0",
            "--frame-id", "base_link",
            "--child-frame-id", "front_camera/camera_link/rgb_camera",
        ],
    )


    camera_optical_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_static_transform_publisher",
        output="log",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.0",
            "--yaw", "-1.57079632679",
            "--pitch", "0.0",
            "--roll", "-1.57079632679",
            "--frame-id", "front_camera/camera_link/rgb_camera",
            "--child-frame-id", "front_camera_optical",
        ],
    )


    robot_controllers = os.path.join(get_package_share_directory(package_name), 'config', 'shok_controllers.yaml')


    # hand_controllers = os.path.join(get_package_share_directory(package_name), 'config', 'hand_controllers.yaml')

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description,
                    move_group_path] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name',
                   'ruka_gz', '-allow_renaming', 'false'],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            "joint_state_broadcaster",
        ],
    )

    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'ruka_arm_controller',
            '--param-file',
            robot_controllers,
            ],
    )

    ruka_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ruka_hand_controller",
                   '--param-file',
        robot_controllers,],
    )


    gazebo_ros_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[
            {'config_file': os.path.join(get_package_share_directory('ruka_gz'), 'config', 'ros_gz_bridge.yaml')},
        ]
    )


    

    world_sdf = os.path.join(
        get_package_share_directory('ruka_gz'),
        'world',
        'openvla_rgb_floor_world.sdf'
    )
    gz_args = f' -r -v 4 {world_sdf} --physics-engine gz-physics-bullet-featherstone-plugin'
    return LaunchDescription([
        rviz_config_arg,
        move_group_node,
        #rviz_node,
        static_tf_node,
        camera_static_tf_node,
        camera_optical_tf_node,
        gazebo_ros_bridge,

        node_robot_state_publisher,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                       'launch',
                                       'gz_sim.launch.py'
                    ])]
            ),
            launch_arguments={'gz_args': gz_args}.items(),
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[joint_trajectory_controller_spawner],
            )
        ),        
        gz_spawn_entity,
        # Launch Arguments
        # DeclareLaunchArgument(
        #     name='use_sim_time',
        #      value=True
        #   ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock'),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[rviz_node],
            )
        ),  
        ruka_hand_controller_spawner

        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=gz_spawn_entity,
        #         on_exit=[sim_time],
        #     )
        # ),  

      
    #    rviz_node
    ])