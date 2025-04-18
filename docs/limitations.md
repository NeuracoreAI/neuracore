<!-- omit in toc -->
# Limitations

While we strive to make the **Neuracore** platform as powerful and flexible as possible, certain limitations are inevitable due to real-world hardware constraints. This document outlines those limitations and provides guidance on how to mitigate them.

<!-- omit in toc -->
## Contents

- [Data Logging Limits](#data-logging-limits)
  - [Bandwidth \& Buffering](#bandwidth--buffering)
  - [Processing Overhead](#processing-overhead)
    - [Enable Hardware Acceleration](#enable-hardware-acceleration)
    - [Live Data Monitoring](#live-data-monitoring)
    - [Distribute Data Collection](#distribute-data-collection)

## Data Logging Limits

Neuracore streams data live as it is collected, eliminating the need for local storage and enabling real-time monitoring. However, live streaming introduces several considerations:

### Bandwidth & Buffering

A stable and sufficiently fast internet connection is required to support the frequency and volume of data being streamed. Data awaiting transmission is buffered in memory, and buffer overflows can occur if the connection is too slow or unstable.

If you're encountering buffer overflows, try the following:

- **Increase available memory** to accommodate larger buffers.
- **Improve connection quality**, preferably using a wired connection.
- **Reduce logging frequency**:
  - Avoid logging redundant or duplicate data points.
- **Reduce data volume**:
  - Ensure camera streams are logged at the appropriate resolution.
  - Avoid logging stationary joints.

> **Note:**  
> Be consistent with which joints you log. Each unique combination of joints is treated as a separate data stream. A maximum of 50 concurrent streams are supported per instance.

### Processing Overhead

Video streams (e.g., RGB or depth) can be significantly larger than other types of data. To mitigate bandwidth usage, Neuracore uses lossless `h264` compression. However, encoding high-resolution video can be CPU-intensive.

If video encoding is creating a performance bottleneck, consider the following:

#### Enable Hardware Acceleration

By default, PyAV (used for video encoding) utilizes a software-based FFmpeg build for compatibility. For improved performance, compile FFmpeg with hardware acceleration.

Useful resources:

- [NVIDIA FFmpeg Transcoding Guide](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)
- [StackOverflow: PyAV & FFmpeg Hardware Support](https://stackoverflow.com/questions/71618462/why-pyav-package-in-python-can-not-recognize-h264-cuvid-codec-while-ffmpeg-can-d)
- [PyAV Hardware Acceleration Discussion](https://github.com/PyAV-Org/PyAV/issues/307)

#### Live Data Monitoring

Each viewer connected to the robot via the [Neuracore dashboard](https://www.neuracore.app/dashboard/robots) adds an additional video stream. Although monitoring uses lossy compression and is capped at 30 FPS, it increases encoding load—especially when combined with recording.

To reduce load:

- Limit the number of simultaneous viewers.
- Disable live data sharing if not needed

> **Info:**
> You can disable live data sharing at any time with:
> ```python
>  nc.stop_live_data(robot_name, instance)
>  # If `robot_name` and `instance` are omitted, the last active robot will be used.
> ```
> This will not affect recording in any way.

#### Distribute Data Collection

You can log different sensor data from separate machines for the same robot instance. This can help balance resource usage. To ensure all data is associated correctly, use the same `robot_name` and `instance` number across machines.