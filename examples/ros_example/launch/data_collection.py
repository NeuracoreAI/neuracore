from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the bimanual robot data collection setup."""

    return LaunchDescription([
        Node(
            package="ros_example",
            executable="simulation_node",
        ),
        Node(
            package="ros_example",
            executable="data_logger_node",
        ),
    ])
