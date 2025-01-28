import asyncio
import base64
import io
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import websockets
from PIL import Image

from neuracore.const import API_URL

from .auth import get_auth
from .exceptions import StreamingError
from .robot import Robot

logger = logging.getLogger(__name__)

MAX_DEPTH = 10.0


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


class RateLimitedQueue:
    def __init__(self, name: str, rate_limit: RateLimit, message_formatter):
        self.name = name
        self._message_formatter = message_formatter
        self._queue = queue.Queue(
            maxsize=(
                2
                if rate_limit.strategy == RateLimitStrategy.DROP
                else rate_limit.max_buffer_size
            )
        )
        self._rate_limit = rate_limit
        self._last_processed_time = 0.0

    def can_put(self) -> bool:
        current_time = time.time()
        if self._rate_limit.strategy == RateLimitStrategy.DROP:
            min_interval = 1 / self._rate_limit.messages_per_second
            if current_time - self._last_processed_time < min_interval:
                return False
        return True

    def put(self, raw_data: Any) -> bool:
        try:
            if not self.can_put():
                return False
            message = QueuedMessage(timestamp=time.time(), data=raw_data)
            self._queue.put(message, block=False)
            return True
        except queue.Full:
            return False

    def get(self) -> QueuedMessage | None:
        try:
            message = self._queue.get_nowait()
            message.data = self._message_formatter(message.data)
            self._last_processed_time = time.time()
            return message
        except queue.Empty:
            return None

    def empty(self) -> bool:
        return self._queue.empty()

    def get_sleep_time(self) -> float:
        if self._rate_limit.strategy == RateLimitStrategy.BUFFER:
            min_interval = 1 / self._rate_limit.messages_per_second
            return max(0, self._last_processed_time + min_interval - time.time())
        return 0


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
    def __init__(self, robot_id: str, stream_type: str):

        assert stream_type in ["states", "actions", "images"]

        self._robot_id = robot_id
        self._auth = get_auth()
        self._running = False
        self._rate_limit = RateLimit(
            messages_per_second=60, strategy=_rate_limit_strategy
        )

        self._thread = None
        self._loop = None

        base_url = API_URL.replace("http://", "ws://").replace("https://", "wss://")
        auth_headers = self._auth.get_headers()

        if stream_type == "states":
            self._queue = RateLimitedQueue(
                "states",
                self._rate_limit,
                lambda data: {"joint_positions": data["joint_states"]},
            )
        elif stream_type == "actions":
            self._queue = RateLimitedQueue(
                "actions",
                self._rate_limit,
                lambda data: {"action": data["action"]},
            )
        elif stream_type == "images":
            self._queue = RateLimitedQueue(
                "images",
                self._rate_limit,
                lambda data: {
                    "type": data["type"],
                    "camera_id": data["camera_id"],
                    "data": (
                        _encode_image(data["image_data"])
                        if data["type"] == "rgb"
                        else _encode_depth_image(data["image_data"])
                    ),
                    "resolution": data["resolution"],
                    "encoding": data.get("encoding", "jpg"),
                },
            )

        else:
            ValueError(f"Invalid stream type: {stream_type}")

        self._processor = QueueProcessor(
            stream_type,
            self._queue,
            f"{base_url}/robots/ws/{robot_id}/{stream_type}/ingest",
            auth_headers,
        )

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_processor())
            self._loop.run_forever()
        finally:
            self._stop_event_loop()

    def _stop_event_loop(self):
        if self._loop:
            self._loop.run_until_complete(self._stop_processor())
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

    async def _start_processor(self):
        await self._processor.start()

    async def _stop_processor(self):
        await self._processor.stop()

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

    def can_put(self) -> bool:
        return self._queue.can_put()

    def queue_data(self, data: dict[str, Any]) -> bool:
        return self._queue.put(data)

    def wait_until_queues_empty(self):
        while not self._queue.empty():
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


def _encode_depth_image(
    depth: np.ndarray, min_depth: float = 0, max_depth: float = MAX_DEPTH
) -> str:
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    depth_normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    pil_image = Image.fromarray((depth_normalized * 65535).astype(np.uint16))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Global registries and functions
_streams: dict[str, DataStream] = {}
_sensor_registers: dict[str, SensorRegister] = {}
_rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.DROP


def _get_or_create_stream(
    robot_id: str, stream_name: str, stream_type: str
) -> DataStream:
    if stream_name not in _streams:
        stream = DataStream(robot_id, stream_type)
        stream.start()
        _streams[stream_name] = stream
    return _streams[stream_name]


def _get_or_create_sensor_register(robot_id: str) -> SensorRegister:
    if robot_id not in _sensor_registers:
        _sensor_registers[robot_id] = SensorRegister()
    return _sensor_registers[robot_id]


# Public API functions
def log_joints(robot: Robot, joint_positions: dict[str, float]):
    stream = _get_or_create_stream(robot.id, f"{robot.id}_states", "states")
    if not stream.can_put():
        return
    stream.queue_data({"joint_states": joint_positions})


def log_action(robot: Robot, action: dict[str, float]):
    stream = _get_or_create_stream(robot.id, f"{robot.id}_actions", "actions")
    if not stream.can_put():
        return
    stream.queue_data({"action": action})


def log_rgb(
    robot: Robot, camera_id: str, image: np.ndarray, resolution: list[int] | None = None
):
    sensor_register = _get_or_create_sensor_register(robot.id)
    sensor_register.validate(camera_id, "RGB", image)
    stream = _get_or_create_stream(robot.id, f"{robot.id}_rgb_{camera_id}", "images")
    if not stream.can_put():
        return
    stream.queue_data({
        "type": "rgb",
        "camera_id": camera_id,
        "image_data": image,
        "resolution": resolution or [image.shape[1], image.shape[0]],
    })


def log_depth(
    robot: Robot, camera_id: str, depth: np.ndarray, resolution: list[int] | None = None
):
    sensor_register = _get_or_create_sensor_register(robot.id)
    sensor_register.validate(camera_id, "DEPTH", depth)
    stream = _get_or_create_stream(robot.id, f"{robot.id}_depth_{camera_id}", "images")
    if not stream.can_put():
        return
    stream.queue_data({
        "type": "depth",
        "camera_id": camera_id,
        "image_data": depth,
        "resolution": resolution or [depth.shape[1], depth.shape[0]],
    })


def set_rate_limit_strategy(strategy: RateLimitStrategy):
    global _rate_limit_strategy
    _rate_limit_strategy = strategy
    for stream in _streams.values():
        stream._rate_limit.strategy = strategy


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
