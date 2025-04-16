# Limitations 

While we make every effort to make the neuracore platform as flexible as possible there are a number of real world practical limitations. 


## Data Logging Limits

When logging your data it is streamed live to the neuracore platform, this means your data is immediately available and it eliminates your storage requirements. However streaming live data comes with its own limitations and considerations:

### Bandwidth and Buffering

The internet connection for the datasource sufficient for the for the frequency and size of data provided. Logged data that is waiting to be sent is buffered, if your connection is inconsistent or insufficient these buffers will use memory. If your finding the buffers are overflowing consider these options:

 - Increasing the available memory
 - Using a Faster/More stable connection, try using a wired connection where possible.
 - Lowering the frequency that you log data.
    - Check that you are not logging duplicates or redundant data points.
 - Lowering the quantity of data logged
    - Check that you are logging cameras at the correct resolution.
    - Check you are not logging any stationary or anomalous joints.

> [!NOTE]
>
> Try to be consistent in which joints that you are logging, each collection of joints is logged separately


### Processing

Video data is especially bandwidth intensive, we use lossless `h264` compression to reduce your bandwidth requirements. With many high resolution streams the processor may not be able to keep up. If you are finding the CPU utilization is too high while encoding your video stream here are some options:


 - *Hardware Acceleration*: We use PyAv to do this encoding, by default for compatibility reasons this comes with a version of FFmpeg that uses software rendering. Building your own hardware enabled version can offer performance improvements.
    - [nvidia-ffmpeg-transcoding-guide](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)
    - [py-av-ffmpeg-build](https://stackoverflow.com/questions/71618462/why-pyav-package-in-python-can-not-recognize-h264-cuvid-codec-while-ffmpeg-can-d)
    - [py-av-hwaccel-discussion](https://github.com/PyAV-Org/PyAV/issues/307)
 - *Live Data sharing*: Data provided to the web interface needs to be encoded differently if recording at the same time this essentially doubles the encoding work. you can disable live data sharing at any time by calling `nc.stop_live_data()` this will use the currently active robot instance unless specified.



