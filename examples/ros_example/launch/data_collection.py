from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the bimanual robot data collection setup."""

    record = LaunchConfiguration("record")
    num_episodes = LaunchConfiguration("num_episodes")
    dataset_name = LaunchConfiguration("dataset_name")

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            "record",
            default_value="False",
            description="Whether to record data with neuracore",
        ),
        DeclareLaunchArgument(
            "num_episodes", default_value="5", description="Number of episodes to run"
        ),
        DeclareLaunchArgument(
            "dataset_name",
            default_value="ROS2_BimanualVX300s_Dataset",
            description="Name for the dataset",
        ),
        Node(
            package="ros_example",
            executable="simulation_node",
            name="simulation_node",
        ),
        Node(
            package="ros_example",
            executable="data_logger_node",
            name="data_logger_node",
            parameters=[{
                "record": record,
                "dataset_name": dataset_name,
            }],
        ),
        Node(
            package="ros_example",
            executable="action_generator_node",
            name="action_generator_node",
            output="screen",
            parameters=[{
                "num_episodes": num_episodes,
            }],
        ),
    ])
