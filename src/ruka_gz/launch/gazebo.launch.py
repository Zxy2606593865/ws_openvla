# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.actions import RegisterEventHandler
# from launch.event_handlers import OnProcessExit
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
# from ament_index_python.packages import get_package_share_directory

# from launch_ros.actions import Node
# from launch_ros.substitutions import FindPackageShare
# import os


# def generate_launch_description():
#     # Launch Arguments
#     use_sim_time = LaunchConfiguration('use_sim_time', default=True)
#     package_name = 'ruka_gz'
#     ros2_control_hardware_type = DeclareLaunchArgument(
#         "ros2_control_hardware_type",
#        default_value="gz_ros2_control",
#        description="Specify the hardware type for ros2_control.",
#     )
#     robot_description_content = Command(
#         [
#             PathJoinSubstitution([FindExecutable(name='xacro')]),
#             ' ',
#             PathJoinSubstitution(
#                 [FindPackageShare('ruka_gz'),
#                  'config', 'ruka_gz.urdf.xacro']
#             ),
#         ]
#     )
#     robot_description = {'robot_description': robot_description_content}

#     robot_controllers = os.path.join(get_package_share_directory(package_name), 'config', 'shok_controllers.yaml')
    
#     ros2_control_node = Node(
#         package="controller_manager",
#         executable="ros2_control_node",
#         # parameters=[robot_controllers],
        
#         parameters=[{'robot_description': robot_description},
#                     robot_controllers],

#         # remappings=[
#         #     ("/controller_manager/robot_description", "/robot_description"),
#         # ],
#         output="both",
#         # namespace='ruka_gz',
#     )

   

#     node_robot_state_publisher = Node(
#         package='robot_state_publisher',
#         executable='robot_state_publisher',
#         output='both',
#         parameters=[robot_description],
#         # namespace='ruka_gz',
#     )

#     gz_spawn_entity = Node(
#         package='ros_gz_sim',
#         executable='create',
#         output='screen',
#         arguments=['-topic', 'robot_description', '-name',
#                    'ruka_gz', '-allow_renaming', 'true'],
#     )

#     joint_state_broadcaster_spawner = Node(
#         package='controller_manager',
#         executable='spawner',
#         arguments=[
#             "joint_state_broadcaster",
#             "--controller-manager",
#             "/controller_manager",
#         ],
#     )
#     # ruka_arm_controller_spawner = Node(
#     #     package='controller_manager',
#     #     executable='spawner',
#     #     arguments=[
#     #         'ruka_gz_arm_controller_gz',
#     #          "--param-file", robot_controllers,
#     #         ],
#     # )

#     joint_trajectory_controller_spawner = Node(
#         package='controller_manager',
#         executable='spawner',
#         arguments=[
#             # 'ruka_arm_controller',
#             # '--param-file',
#             # robot_controllers,
#             "ruka_arm_controller", "-c", "/controller_manager"
#             ],
#     )

#     return LaunchDescription([

#         node_robot_state_publisher,
#         ros2_control_hardware_type,
#         IncludeLaunchDescription(
#             PythonLaunchDescriptionSource(
#                 [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
#                                        'launch',
#                                        'gz_sim.launch.py'])]),
#             launch_arguments=[('gz_args', [' -r -v 4 empty.sdf'])]),
#         ros2_control_node,
        
#         RegisterEventHandler(
#             event_handler=OnProcessExit(
#                 target_action=gz_spawn_entity,
#                 on_exit=[joint_state_broadcaster_spawner],
#             )
#         ),
#         RegisterEventHandler(
#             event_handler=OnProcessExit(
#                 target_action=joint_state_broadcaster_spawner,
#                 on_exit=[joint_trajectory_controller_spawner],
#             )
#         ),        
#         gz_spawn_entity,
#         # Launch Arguments
#         DeclareLaunchArgument(
#             'use_sim_time',
#             default_value=use_sim_time,
#             description='If true, use simulated clock'),
#     ])

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    package_name = 'ruka_gz'

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


    robot_controllers = os.path.join(get_package_share_directory(package_name), 'config', 'shok_controllers.yaml')

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name',
                   'ruka_gz', '-allow_renaming', 'true'],
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

    return LaunchDescription([

        node_robot_state_publisher,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                       'launch',
                                       'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', [' -r -v 4 empty.sdf'])]),
        # ros2_control_node,
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
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock'),
    ])