import neuracore as nc # pip install neuracore
import time
import numpy as np

# ensure you have an account at neuracore.com
nc.login()

nc.connect_robot("ur5e")

# Create a dataset for recording
dataset = nc.create_dataset(
    name="COG",
    description="Example dataset with multiple data types"
)

# nc.connect_dataset(dataset)
nc.start_recording()
# generate an np array of image shape
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
nc.log_custom_1d("my_custom_data", np.array([1, 2, 3, 4, 5]), timestamp=time.time())
for i in range(5):
    nc.log_joint_positions({"a": 0.5}, timestamp=time.time())
    nc.log_rgb("camera1", image, timestamp=time.time())
nc.stop_recording()