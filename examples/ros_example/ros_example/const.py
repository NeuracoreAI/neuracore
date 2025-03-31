from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
