from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the bimanual robot data collection setup."""

    # Declare the training_run_name parameter with a default value
    training_run_name_arg = DeclareLaunchArgument(
        "training_run_name",
        description="Name of the training run to use for the policy",
    )

    # Create the simulation node with the training_run_name parameter
    simulation_node = Node(
        package="ros_example",
        executable="simulation_node_prediction",
        parameters=[{"training_run_name": LaunchConfiguration("training_run_name")}],
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
        training_run_name_arg,
        simulation_node,
        data_logger_node,
    ])
