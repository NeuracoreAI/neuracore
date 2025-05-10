from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the bimanual robot data collection setup."""

    # Declare the max_episodes parameter with a default value
    max_episodes_arg = DeclareLaunchArgument(
        "max_episodes",
        default_value="10",
        description="Number of episodes to run before termination",
    )

    # Create the simulation node with the max_episodes parameter
    simulation_node = Node(
        package="ros_example",
        executable="simulation_node",
        parameters=[{"max_episodes": LaunchConfiguration("max_episodes")}],
        output="screen",
        emulate_tty=True,
    )

    # Create the data logger node
    data_logger_node = Node(
        package="ros_example",
        executable="data_logger_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        max_episodes_arg,
        simulation_node,
        data_logger_node,
    ])
