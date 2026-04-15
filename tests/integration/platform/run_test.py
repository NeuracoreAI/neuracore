from test_runtime_data_flow import HIGH_TIME_TO_STOP_S, TestConfig, run_streaming_test

import neuracore

neuracore.stop_live_data()
if __name__ == "__main__":
    config = TestConfig(
        fps=10,
        duration_sec=1,
        image_width=1920,
        image_height=1080,
        num_cameras=1,
        use_depth=True,
        synched_time=True,
        stop_wait_timeout_s=HIGH_TIME_TO_STOP_S,
    )

    run_streaming_test(config)
