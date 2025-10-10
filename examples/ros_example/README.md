# Neuracore ROS2 Example

Example showing how there is no need to synchronise your data across nodes -- Neuracore does this for you!

```bash
docker build --file examples/ros_example/Dockerfile   -t ros_example:latest .
nc-login
docker run -it --rm -v  ~/.neuracore:/root/.neuracore --network host ros_example:latest
```

If you want to see an example of making asynchronous predictions from a trained model, you can run:

```bash
docker run -it --rm -v  ~/.neuracore:/root/.neuracore --network host ros_example:latest ros2 launch ros_example prediction.py training_run_name:=MY_RUN_NAME
```

Note: for this you must ensure that you have a policy available trained through neuracore.
