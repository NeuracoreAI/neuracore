import asyncio
import base64
import io
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque

import numpy as np
import websockets
from PIL import Image

from neuracore.const import API_URL

from .auth import get_auth
from .exceptions import StreamingError
from .robot import Robot

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    DROP = "drop"
    BUFFER = "buffer"


@dataclass
class RateLimit:
    messages_per_second: float
    strategy: RateLimitStrategy
    max_buffer_size: int = 10000


@dataclass
class QueuedMessage:
    timestamp: float
    data: dict


class RateEstimator:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add_message(self) -> float:
        with self._lock:
            now = time.time()
            self.timestamps.append(now)

            if len(self.timestamps) < 2:
                return 0.0

            window_duration = now - self.timestamps[0]
            return (
                (len(self.timestamps) - 1) / window_duration
                if window_duration > 0
                else 0.0
            )

    def get_sleep_time(self, target_rate: float) -> float:
        current_rate = self.get_rate()
        if current_rate <= target_rate:
            return 0.0

        with self._lock:
            if len(self.timestamps) < 2:
                return 0.0

            target_interval = 1.0 / target_rate
            if len(self.timestamps) >= 2:
                last_interval = self.timestamps[-1] - self.timestamps[-2]
                if last_interval < target_interval:
                    return target_interval - last_interval
            return 0.0

    def get_rate(self) -> float:
        with self._lock:
            if len(self.timestamps) < 2:
                return 0.0
            window_duration = self.timestamps[-1] - self.timestamps[0]
            return (
                (len(self.timestamps) - 1) / window_duration
                if window_duration > 0
                else 0.0
            )


class RateLimitedQueue:
    def __init__(self, name: str, rate_limit: RateLimit, message_formatter):
        self.name = name
        self._message_formatter = message_formatter
        self._queue = queue.Queue(
            maxsize=(
                rate_limit.max_buffer_size
                if rate_limit.strategy == RateLimitStrategy.BUFFER
                else 1
            )
        )
        self._rate_limit = rate_limit
        self._rate_estimator = RateEstimator()

    def put(self, raw_data: Any) -> bool:
        current_rate = self._rate_estimator.get_rate()

        if current_rate >= self._rate_limit.messages_per_second:
            if self._rate_limit.strategy == RateLimitStrategy.DROP:
                logger.debug(
                    f"{self.name}: Dropping message due to "
                    f"rate limit ({current_rate:.1f} msgs/sec)"
                )
                return False

        try:
            message = QueuedMessage(
                timestamp=time.time(), data=self._message_formatter(raw_data)
            )
            self._queue.put(message, block=False)
            self._rate_estimator.add_message()
            return True
        except queue.Full:
            logger.warning(f"{self.name}: Buffer full, dropping message")
            return False

    def get(self) -> QueuedMessage | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def empty(self) -> bool:
        return self._queue.empty()

    def get_sleep_time(self) -> float:
        return self._rate_estimator.get_sleep_time(self._rate_limit.messages_per_second)


class QueueProcessor:
    def __init__(
        self, name: str, queue: RateLimitedQueue, websocket_url: str, auth_headers: dict
    ):
        self.name = name
        self._queue = queue
        self._websocket_url = websocket_url
        self._auth_headers = auth_headers
        self._ws = None
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        logger.info(f"Started {self.name} queue processor")

    async def stop(self):
        self._running = False
        if self._task:
            await self._task
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info(f"Stopped {self.name} queue processor")

    async def _process_queue(self):
        while self._running:
            try:
                if not self._ws:
                    self._ws = await websockets.connect(
                        self._websocket_url, additional_headers=self._auth_headers
                    )

                if not self._queue.empty():
                    sleep_time = self._queue.get_sleep_time()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                    queued_message = self._queue.get()
                    if queued_message is not None:
                        message = {
                            "timestamp": queued_message.timestamp,
                            **queued_message.data,
                        }
                        await self._ws.send(json.dumps(message))
                else:
                    await asyncio.sleep(0.001)

            except websockets.WebSocketException as e:
                logger.error(f"{self.name} WebSocket error: {e}")
                if self._ws:
                    await self._ws.close()
                    self._ws = None
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"{self.name} processing error: {e}")
                await asyncio.sleep(0.1)


class DataStream:
    def __init__(self, robot_id: str):
        self._robot_id = robot_id
        self._auth = get_auth()
        self._running = False
        self._rate_limit = RateLimit(
            messages_per_second=100, strategy=RateLimitStrategy.BUFFER
        )

        self._thread = None
        self._loop = None

        base_url = API_URL.replace("http://", "ws://").replace("https://", "wss://")
        auth_headers = self._auth.get_headers()

        # Create independent queues with their formatters
        self._queues = {
            "states": RateLimitedQueue(
                "states",
                self._rate_limit,
                lambda data: {"joint_positions": data["joint_states"]},
            ),
            "actions": RateLimitedQueue(
                "actions", self._rate_limit, lambda data: {"action": data["action"]}
            ),
            "images": RateLimitedQueue(
                "images",
                self._rate_limit,
                lambda data: {
                    "type": data["type"],
                    "camera_id": data["camera_id"],
                    "data": data["image_data"],
                    "resolution": data["resolution"],
                    "encoding": data.get("encoding", "jpg"),
                },
            ),
        }

        self._processors = {
            name: QueueProcessor(
                name,
                queue,
                f"{base_url}/robots/ws/{robot_id}/{name}/ingest",
                auth_headers,
            )
            for name, queue in self._queues.items()
        }

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_processors())
            self._loop.run_forever()
        finally:
            self._stop_event_loop()

    def _stop_event_loop(self):
        if self._loop:
            self._loop.run_until_complete(self._stop_processors())
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            finally:
                self._loop.close()
                self._loop = None

    async def _start_processors(self):
        await asyncio.gather(
            *[processor.start() for processor in self._processors.values()]
        )

    async def _stop_processors(self):
        await asyncio.gather(
            *[processor.stop() for processor in self._processors.values()]
        )

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop)
        self._thread.daemon = True
        self._thread.start()
        time.sleep(0.1)  # Small delay to ensure event loop is running
        logger.info("DataStream started")

    def stop(self):
        if not self._running:
            return

        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("DataStream thread did not shut down cleanly")
            self._thread = None

        logger.info("DataStream stopped")

    def queue_state_data(self, data: dict[str, Any]) -> bool:
        return self._queues["states"].put(data)

    def queue_action_data(self, data: dict[str, Any]) -> bool:
        return self._queues["actions"].put(data)

    def queue_image_data(self, data: dict[str, Any]) -> bool:
        return self._queues["images"].put(data)

    def wait_until_queues_empty(self):
        while any(not q.empty() for q in self._queues.values()):
            logger.info("Waiting for queues to empty...")
            time.sleep(0.5)


@dataclass
class SensorData:
    sensor_type: str
    shape: list[int]


class SensorRegister:
    def __init__(self):
        self._sensors = {}

    def register_sensor(self, sensor_id: str, sensor: Any):
        self._sensors[sensor_id] = sensor

    def validate(self, sensor_name: str, sensor_type: str, data: np.ndarray) -> Any:
        active_sensor: SensorData = self._sensors.get(sensor_name)
        if not active_sensor:
            active_sensor = self._sensors[sensor_name] = SensorData(
                sensor_type=sensor_type, shape=data.shape
            )
        if active_sensor.sensor_type != sensor_type:
            raise StreamingError(
                "Sensor type mismatch! "
                f"Expected: {active_sensor.sensor_type}, got: {sensor_type}. "
                "Each sensor must have a unique name."
            )
        if active_sensor.shape != data.shape:
            raise StreamingError(
                "Sensor data shape mismatch! "
                f"Expected: {active_sensor.shape}, got: {data.shape}"
            )
        return self._sensors.get(sensor_name)


# Helper functions for image encoding
def _encode_image(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image.astype("uint8"))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Global registries and functions
_streams: dict[str, DataStream] = {}
_sensor_registers: dict[str, SensorRegister] = {}


def _get_or_create_stream(robot_id: str) -> DataStream:
    if robot_id not in _streams:
        stream = DataStream(robot_id)
        stream.start()
        _streams[robot_id] = stream
    return _streams[robot_id]


def _get_or_create_sensor_register(robot_id: str) -> SensorRegister:
    if robot_id not in _sensor_registers:
        _sensor_registers[robot_id] = SensorRegister()
    return _sensor_registers[robot_id]


# Public API functions
def log_joints(robot: Robot, joint_positions: dict[str, float]):
    stream = _get_or_create_stream(robot.id)
    stream.queue_state_data({"joint_states": joint_positions})


def log_action(robot: Robot, action: dict[str, float]):
    stream = _get_or_create_stream(robot.id)
    stream.queue_action_data({"action": action})


def log_rgb(
    robot: Robot, camera_id: str, image: np.ndarray, resolution: list[int] | None = None
):
    sensor_register = _get_or_create_sensor_register(robot.id)
    sensor_register.validate(camera_id, "RGB", image)

    if image.dtype != np.uint8:
        image = (image * 255 if image.max() <= 1 else image).astype(np.uint8)

    stream = _get_or_create_stream(robot.id)
    stream.queue_image_data({
        "type": "rgb",
        "camera_id": camera_id,
        "image_data": _encode_image(image),
        "resolution": resolution or [image.shape[1], image.shape[0]],
    })


def log_depth(
    robot: Robot, camera_id: str, depth: np.ndarray, resolution: list[int] | None = None
):
    sensor_register = _get_or_create_sensor_register(robot.id)
    sensor_register.validate(camera_id, "DEPTH", depth)

    depth = depth / 1000.0 if depth.max() > 100 else depth
    depth_img = (depth * 1000).astype(np.uint16)

    pil_image = Image.fromarray(depth_img)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    stream = _get_or_create_stream(robot.id)
    stream.queue_image_data({
        "type": "depth",
        "camera_id": camera_id,
        "image_data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "resolution": resolution or [depth.shape[1], depth.shape[0]],
    })


def stop_streaming(robot: Robot):
    if robot.id in _streams:
        _streams[robot.id].stop()
        del _streams[robot.id]


def stop_all_streams():
    for stream in _streams.values():
        stream.stop()
    _streams.clear()


def wait_until_stream_empty(robot: Robot):
    if robot.id in _streams:
        _streams[robot.id].wait_until_queues_empty()
