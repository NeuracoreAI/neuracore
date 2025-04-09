# Neuarcore ROS2 Example

Example showing how there is no need to synchronise your data across nodes -- Neuracore does this for you!

```bash
docker build --file examples/ros_example/Dockerfile   -t ros_example:latest .
docker run -it --rm -v  ~/.neuracore:/root/.neuracore --network host ros_example:latest
```
