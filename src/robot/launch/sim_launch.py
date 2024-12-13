from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import ExecuteProcess

def generate_launch_description():
    package_name = 'robot'
    package_share_dir = get_package_share_directory(package_name)
    config = os.path.join(package_share_dir, 'config', 'sim_launch.yaml')
    sim_node = Node(
        package="robot",
        executable="sim_node",
        name='sim_node',
        output='screen',
        parameters=[config]
    )
    robot_0_node = Node(
        package="robot",
            executable="robot_node",
            name='robot_0_node',
            output='screen',
            parameters=[config]
    )
    robot_1_node = Node(
        package="robot",
            executable="robot_node",
            name='robot_1_node',
            output='screen',
            parameters=[config]
    )
    
    rviz_config = os.path.join(
      package_share_dir,
      'config',
      'rviz2',
      'exp_map.rviz'
      )
    print(rviz_config)
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    # rviz2_node = ExecuteProcess(
    #     cmd=['rviz2', '-d', rviz_config],
    #     output='screen'
    # )
    
    ld = LaunchDescription(
        [
            sim_node,
            rviz2_node,
            robot_0_node,
            robot_1_node
        ]
    )
   
    
    return ld