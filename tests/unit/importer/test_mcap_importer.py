from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from neuracore_types import DataType
from PIL import Image

from neuracore.core.utils.depth_utils import MAX_DEPTH
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.mcap_importer import MCAPDatasetImporter
from neuracore.importer.utils.mcap.cache import CachedMessage, MessageCache
from neuracore.importer.utils.mcap.config import MCAPImportConfig
from neuracore.importer.utils.mcap.decoder import (
    ImageDecoder,
    JSONDecoderFactory,
    MCAPMessageDecoder,
    RawPassthroughDecoderFactory,
    list_decoder_factories,
)
from neuracore.importer.utils.mcap.logger import MessageLogger
from neuracore.importer.utils.mcap.paths import resolve_path, split_topic_path
from neuracore.importer.utils.mcap.preprocessor import MessagePreprocessor
from neuracore.importer.utils.mcap.progress import (
    EmitProgressReporter,
    NullProgressReporter,
)
from neuracore.importer.utils.mcap.session import RecordingSession
from neuracore.importer.utils.mcap.topics import TopicMapper


def _make_mapping_item(
    name: str,
    *,
    source_name: str | None = None,
    index=None,
    index_range=None,
):
    return SimpleNamespace(
        name=name,
        source_name=source_name,
        index=index,
        index_range=index_range,
        transforms=lambda value: value,
    )


def _make_import_config(source: str, mapping: list[SimpleNamespace]):
    return SimpleNamespace(
        source=source,
        mapping=mapping,
        format=SimpleNamespace(language_type=None),
    )


def _make_dataset_config(data_import_config: dict):
    return SimpleNamespace(
        data_import_config=data_import_config,
        robot=SimpleNamespace(name="test_robot"),
        frequency=30,
    )


def test_split_topic_path_and_resolve_path():
    topic, path = split_topic_path("/camera/color.image.data")
    assert topic == "/camera/color"
    assert path == ["image", "data"]

    payload = {
        "outer": {
            "values": [
                {"x": 1},
                {"x": 2},
            ]
        }
    }
    value = resolve_path(payload, ["outer", "values", "1", "x"])
    assert value == 2


def test_topic_mapper_builds_mixed_absolute_and_relative_topics():
    mapper = TopicMapper(
        _make_dataset_config({
            DataType.RGB_IMAGES: _make_import_config(
                source="/camera/color",
                mapping=[
                    _make_mapping_item(
                        "cam_left",
                        source_name="/camera/color/image_cam1.data",
                    ),
                    _make_mapping_item("color_main", source_name="image"),
                ],
            )
        })
    )

    topics = mapper.get_all_topics()
    assert topics == ["/camera/color", "/camera/color/image_cam1"]

    relative_cfg = mapper.get_configs_for_topic("/camera/color")[0]
    absolute_cfg = mapper.get_configs_for_topic("/camera/color/image_cam1")[0]

    assert [item.name for item in relative_cfg.import_config.mapping] == ["color_main"]
    assert absolute_cfg.mapping_item.name == "cam_left"
    assert absolute_cfg.item_base_path == ["data"]


def test_topic_mapper_requires_source_for_relative_items():
    with pytest.raises(ImportError, match="Relative mapping entries require"):
        TopicMapper(
            _make_dataset_config({
                DataType.RGB_IMAGES: _make_import_config(
                    source="",
                    mapping=[_make_mapping_item("cam", source_name="image")],
                )
            })
        )


def test_json_decoder_factory_decodes_json():
    decoder = JSONDecoderFactory().decoder_for("json", None)
    assert decoder is not None
    assert decoder(b'{"a": 1}') == {"a": 1}


def test_raw_passthrough_decoder_factory_decodes_unknown_payload():
    decoder = RawPassthroughDecoderFactory().decoder_for("custom", None)
    assert decoder is not None
    assert decoder(b"\x01\x02") == b"\x01\x02"


def test_list_decoder_factories_has_raw_fallback_last():
    factories = list_decoder_factories(
        enable_discovery=False, include_raw_fallback=True
    )
    assert factories
    assert isinstance(factories[-1], RawPassthroughDecoderFactory)


def test_message_cache_round_trip(tmp_path: Path):
    pytest.importorskip("msgpack")

    cache_path = tmp_path / "events.msgpack"
    expected = [
        CachedMessage(
            data_type=DataType.RGB_IMAGES.value,
            name="rgb",
            timestamp=1.0,
            log_time_ns=100,
            transformed_data=np.array([[1, 2], [3, 4]], dtype=np.uint8),
            source_topic="/camera",
        ),
        CachedMessage(
            data_type=DataType.LANGUAGE.value,
            name="lang",
            timestamp=2.0,
            log_time_ns=200,
            transformed_data="pick",
            source_topic="/task",
        ),
    ]

    with MessageCache(cache_path, mode="wb") as cache:
        for item in expected:
            cache.write_message(item)

    with MessageCache(cache_path, mode="rb") as cache:
        actual = list(cache.read_messages())

    assert len(actual) == 2
    assert actual[0].name == "rgb"
    np.testing.assert_array_equal(
        actual[0].transformed_data, expected[0].transformed_data
    )
    assert actual[1].transformed_data == "pick"


def test_message_decoder_transform_message():
    mapper = TopicMapper(
        _make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic.data",
                mapping=[
                    _make_mapping_item("value"),
                    _make_mapping_item("inner", source_name="inner.v"),
                ],
            )
        })
    )

    image_decoder = ImageDecoder(logging.getLogger(__name__))
    decoder = MCAPMessageDecoder(
        topic_mapper=mapper,
        prepare_log_data=lambda **kwargs: kwargs["source_data"],
        image_decoder=image_decoder,
    )

    decoded = {"data": {"inner": {"v": [1, 2, 3]}}}
    events = decoder.transform_message(
        "/topic",
        decoded,
        timestamp=1.5,
        log_time_ns=10,
    )

    assert [event.name for event in events] == ["value", "inner"]
    assert isinstance(events[0].transformed_data, dict)
    np.testing.assert_array_equal(events[1].transformed_data, np.array([1, 2, 3]))


def test_message_decoder_iter_transformed_messages():
    mapper = TopicMapper(
        _make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic.data",
                mapping=[_make_mapping_item("value")],
            )
        })
    )

    decoder = MCAPMessageDecoder(
        topic_mapper=mapper,
        prepare_log_data=lambda **kwargs: kwargs["source_data"],
        image_decoder=ImageDecoder(logging.getLogger(__name__)),
    )

    decoded = {"data": [1, 2, 3]}
    events = list(
        decoder.iter_transformed_messages(
            "/topic",
            decoded,
            timestamp=2.0,
            log_time_ns=20,
        )
    )

    assert len(events) == 1
    np.testing.assert_array_equal(events[0].transformed_data, np.array([1, 2, 3]))


def test_message_decoder_clips_depth_values_to_max_depth():
    mapper = TopicMapper(
        _make_dataset_config({
            DataType.DEPTH_IMAGES: _make_import_config(
                source="/depth",
                mapping=[_make_mapping_item("depth_main")],
            )
        })
    )

    decoder = MCAPMessageDecoder(
        topic_mapper=mapper,
        prepare_log_data=lambda **kwargs: kwargs["source_data"],
        image_decoder=ImageDecoder(logging.getLogger(__name__)),
    )

    decoded = np.array([[0.5, MAX_DEPTH + 50.0]], dtype=np.float32)
    events = decoder.transform_message(
        "/depth",
        decoded,
        timestamp=1.0,
        log_time_ns=1,
    )

    assert len(events) == 1
    assert isinstance(events[0].transformed_data, np.ndarray)
    assert events[0].transformed_data.dtype == np.float16
    assert float(np.max(events[0].transformed_data)) <= MAX_DEPTH


def test_preprocessor_estimate_total_messages():
    preprocessor = MessagePreprocessor.__new__(MessagePreprocessor)
    summary = SimpleNamespace(
        statistics=SimpleNamespace(channel_message_counts={1: 10, 2: 5}),
        channels={
            1: SimpleNamespace(topic="/a"),
            2: SimpleNamespace(topic="/b"),
        },
    )
    assert preprocessor._estimate_total_messages(summary, ["/a", "/b"]) == 15


def test_preprocessor_handles_decode_errors_via_on_message_error(
    monkeypatch,
    tmp_path: Path,
):
    mcap_path = tmp_path / "sample.mcap"
    mcap_path.write_bytes(b"mcap")

    class _FakeReader:
        def get_header(self):
            return None

        def get_summary(self):
            return None

        def iter_messages(self, *, topics, log_time_order):
            assert topics == ["/topic"]
            assert log_time_order is True
            yield (
                None,
                SimpleNamespace(topic="/topic"),
                SimpleNamespace(channel_id=1, log_time=1, publish_time=0, data=b"\x01"),
            )
            yield (
                None,
                SimpleNamespace(topic="/topic"),
                SimpleNamespace(channel_id=1, log_time=2, publish_time=0, data=b"\x02"),
            )

    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.preprocessor.make_reader",
        lambda *_args, **_kwargs: _FakeReader(),
    )

    written_events: list[CachedMessage] = []

    class _FakeMessageCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def write_message(self, msg: CachedMessage) -> None:
            written_events.append(msg)

    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.preprocessor.MessageCache",
        _FakeMessageCache,
    )

    errors: list[tuple[int, str, int, str]] = []

    def _on_message_error(
        message_index: int,
        topic: str,
        log_time_ns: int,
        exc: Exception,
    ) -> bool:
        errors.append((message_index, topic, log_time_ns, str(exc)))
        return True

    def _iter_transformed_messages(topic, decoded, *, timestamp, log_time_ns):
        assert topic == "/topic"
        assert isinstance(decoded, dict)
        yield CachedMessage(
            data_type=DataType.CUSTOM_1D.value,
            name="value",
            timestamp=timestamp,
            log_time_ns=log_time_ns,
            transformed_data=np.array([1.0], dtype=np.float32),
            source_topic=topic,
        )

    preprocessor = MessagePreprocessor(
        mcap_file=mcap_path,
        topic_mapper=SimpleNamespace(get_all_topics=lambda: ["/topic"]),
        message_decoder=SimpleNamespace(
            normalize_decoded_message=lambda decoded: decoded,
            iter_transformed_messages=_iter_transformed_messages,
        ),
        decoder_factories=[],
        progress_reporter=NullProgressReporter(),
        logger=logging.getLogger(__name__),
        on_message_error=_on_message_error,
    )

    decode_call_count = 0

    def _fake_decode_message(schema, channel, message):
        nonlocal decode_call_count
        decode_call_count += 1
        if decode_call_count == 1:
            raise ValueError("decode failed")
        return {"value": [1.0]}

    monkeypatch.setattr(preprocessor, "_decode_message", _fake_decode_message)

    stats = preprocessor.preprocess_to_cache(tmp_path / "cache.msgpack")

    assert stats.message_count == 2
    assert stats.event_count == 1
    assert len(written_events) == 1
    assert errors == [(1, "/topic", 1, "decode failed")]


def test_emit_progress_reporter_throttles_updates_and_emits_final():
    calls: list[tuple[int, int | None, str | None]] = []
    reporter = EmitProgressReporter(
        emit_progress=lambda completed, total, label: calls.append(
            (completed, total, label)
        ),
        label="episode-1",
        report_every=5,
    )

    reporter.start_phase("logging", 12)
    for step in range(1, 12):
        reporter.update(step)
    reporter.finish_phase()

    assert [entry[0] for entry in calls] == [0, 5, 10, 11]
    assert calls[-1][2] == "episode-1 [logging]"


def test_recording_session_rotates_when_interval_exceeded(monkeypatch):
    state = {
        "recording": False,
        "start_calls": 0,
        "stop_calls": 0,
    }

    def _is_recording():
        return state["recording"]

    def _start_recording():
        state["recording"] = True
        state["start_calls"] += 1

    def _stop_recording(wait=True):
        state["recording"] = False
        state["stop_calls"] += 1

    def _get_dataset(name=None, id=None):
        return SimpleNamespace(id="dataset-1")

    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.session.nc.is_recording", _is_recording
    )
    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.session.nc.start_recording",
        _start_recording,
    )
    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.session.nc.stop_recording", _stop_recording
    )
    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.session.nc.get_dataset", _get_dataset
    )

    session = RecordingSession(
        dataset_name="out",
        logger=logging.getLogger(__name__),
        rotation_interval_seconds=1,
    )

    session.ensure_active()
    assert state["start_calls"] == 1

    session._session_start_time -= 2.0
    session.ensure_active()
    assert state["start_calls"] == 2
    assert state["stop_calls"] >= 1

    session.stop()
    assert not state["recording"]


def test_mcap_config_rotation_default_tracks_warning_threshold():
    from neuracore.importer.utils.mcap import config as mcap_config

    if mcap_config.DEFAULT_SESSION_ROTATION_SECONDS == 260:
        pytest.skip("RecordingStateManager unavailable in this test environment.")

    from neuracore.core.streaming.recording_state_manager import RecordingStateManager

    expected = max(
        1,
        int(RecordingStateManager.RECORDING_EXPIRY_WARNING)
        - mcap_config.SESSION_ROTATION_WARNING_BUFFER_SECONDS,
    )
    assert mcap_config.SESSION_ROTATION_WARNING_BUFFER_SECONDS == 10
    assert mcap_config.DEFAULT_SESSION_ROTATION_SECONDS == expected


def test_mcap_config_reads_stage_dir_and_progress_interval_from_env(
    monkeypatch,
    tmp_path: Path,
):
    stage_dir = tmp_path / "stage_cache"
    monkeypatch.setenv("NEURACORE_MCAP_STAGE_DIR", str(stage_dir))
    monkeypatch.setenv("NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL", "7")

    cfg = MCAPImportConfig.from_env(skip_on_error="episode")

    assert cfg.stage_dir == stage_dir
    assert cfg.progress_emit_interval == 7
    assert stage_dir.exists()


@pytest.mark.parametrize("interval", ["0", "-1", "abc"])
def test_mcap_config_rejects_invalid_progress_emit_interval_env(
    monkeypatch,
    interval: str,
):
    monkeypatch.setenv("NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL", interval)
    with pytest.raises(
        ValueError,
        match="NEURACORE_MCAP_PROGRESS_EMIT_INTERVAL must be a positive integer",
    ):
        MCAPImportConfig.from_env(skip_on_error="episode")


def test_mcap_config_rotation_default_fallback_when_warning_constant_unavailable(
    monkeypatch,
):
    from neuracore.importer.utils.mcap import config as mcap_config

    module_path = Path(mcap_config.__file__)
    source = module_path.read_text(encoding="utf-8")
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "neuracore.core.streaming.recording_state_manager":
            raise ImportError("simulated missing recording state manager")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    module_name = "_mcap_config_fallback_test"
    module = ModuleType(module_name)
    module.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, module_name, module)
    exec(compile(source, str(module_path), "exec"), module.__dict__)

    assert module.SESSION_ROTATION_WARNING_BUFFER_SECONDS == 10
    assert module.DEFAULT_BACKEND_RECORDING_TTL_SECONDS == 300
    assert module.DEFAULT_SESSION_ROTATION_SECONDS == 260


def test_message_logger_replays_cache(tmp_path: Path):
    pytest.importorskip("msgpack")

    cache_path = tmp_path / "events.msgpack"
    with MessageCache(cache_path, mode="wb") as cache:
        cache.write_message(
            CachedMessage(
                data_type=DataType.LANGUAGE.value,
                name="instruction",
                timestamp=1.0,
                log_time_ns=10,
                transformed_data="open drawer",
            )
        )
        cache.write_message(
            CachedMessage(
                data_type=DataType.CUSTOM_1D.value,
                name="value",
                timestamp=2.0,
                log_time_ns=20,
                transformed_data=np.array([1.0, 2.0]),
            )
        )

    logged: list[tuple[DataType, str]] = []

    class _FakeSession:
        def __init__(self):
            self.ensure_calls = 0
            self.session_count = 1
            self.stopped = False

        def ensure_active(self):
            self.ensure_calls += 1

        def stop(self):
            self.stopped = True

    fake_session = _FakeSession()
    logger = MessageLogger(
        session=fake_session,
        data_logger=lambda data_type, _payload, name, _ts: logged.append(
            (data_type, name)
        ),
        progress_reporter=NullProgressReporter(),
        logger=logging.getLogger(__name__),
    )

    stats = logger.log_from_cache(cache_path, expected_total_messages=2)

    assert stats.message_count == 2
    assert stats.session_count == 1
    assert fake_session.ensure_calls >= 3
    assert fake_session.stopped is True
    assert logged == [(DataType.LANGUAGE, "instruction"), (DataType.CUSTOM_1D, "value")]


def test_message_logger_ensures_session_immediately_before_each_log(tmp_path: Path):
    pytest.importorskip("msgpack")

    cache_path = tmp_path / "events.msgpack"
    with MessageCache(cache_path, mode="wb") as cache:
        cache.write_message(
            CachedMessage(
                data_type=DataType.LANGUAGE.value,
                name="instruction",
                timestamp=1.0,
                log_time_ns=10,
                transformed_data="open drawer",
            )
        )
        cache.write_message(
            CachedMessage(
                data_type=DataType.LANGUAGE.value,
                name="instruction_2",
                timestamp=2.0,
                log_time_ns=20,
                transformed_data="close drawer",
            )
        )

    calls: list[str] = []

    class _FakeSession:
        def __init__(self):
            self.session_count = 1

        def ensure_active(self):
            calls.append("ensure")

        def stop(self):
            calls.append("stop")

    logger = MessageLogger(
        session=_FakeSession(),
        data_logger=lambda _dtype, _payload, name, _ts: calls.append(f"log:{name}"),
        progress_reporter=NullProgressReporter(),
        logger=logging.getLogger(__name__),
        max_replay_bytes_per_second=0,
    )

    logger.log_from_cache(cache_path, expected_total_messages=2)

    log_indices = [idx for idx, value in enumerate(calls) if value.startswith("log:")]
    assert log_indices, "Expected at least one logged message"
    for idx in log_indices:
        assert calls[idx - 1] == "ensure"


def test_message_logger_throttles_replay_bytes(monkeypatch):
    logger = MessageLogger(
        session=SimpleNamespace(
            ensure_active=lambda: None,
            stop=lambda: None,
            session_count=1,
        ),
        data_logger=lambda *_args, **_kwargs: None,
        progress_reporter=NullProgressReporter(),
        logger=logging.getLogger(__name__),
        max_replay_bytes_per_second=100,
    )

    monotonic_times = iter([0.0, 0.0, 0.0, 0.5])
    sleeps: list[float] = []
    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.logger.time.monotonic",
        lambda: next(monotonic_times),
    )
    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.logger.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    logger._throttle_tokens = 0.0
    logger._throttle_last_update = 0.0
    logger._throttle_before_log(DataType.RGB_IMAGES, np.ones((20,), dtype=np.uint8))

    assert sleeps == []
    assert logger._throttle_tokens == 0.0


def test_message_logger_does_not_throttle_small_non_image_data(monkeypatch):
    logger = MessageLogger(
        session=SimpleNamespace(
            ensure_active=lambda: None,
            stop=lambda: None,
            session_count=1,
        ),
        data_logger=lambda *_args, **_kwargs: None,
        progress_reporter=NullProgressReporter(),
        logger=logging.getLogger(__name__),
        max_replay_bytes_per_second=1,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "neuracore.importer.utils.mcap.logger.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    logger._throttle_before_log(DataType.LANGUAGE, "open drawer")
    assert sleeps == []


def test_image_decoder_handles_base64_compressed_payload():
    import base64
    import io

    image = Image.fromarray(np.array([[1000, 2000]], dtype=np.uint16))
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        raw_png = buf.getvalue()

    message = {
        "data": base64.b64encode(raw_png).decode("ascii"),
        "format": "png",
    }
    decoder = ImageDecoder(logging.getLogger(__name__))

    decoded = decoder.coerce_message_data(
        DataType.DEPTH_IMAGES,
        message["data"],
        message,
    )
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (1, 2)


def test_mcap_importer_build_work_items(monkeypatch, tmp_path: Path):
    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")

    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
    )

    items = list(importer.build_work_items())
    assert len(items) == 1
    assert items[0].description == "episode_001.mcap"
    assert items[0].metadata["path"] == str(mcap_path)


def test_mcap_importer_import_item_runs_preprocess_and_logging(
    monkeypatch, tmp_path: Path
):
    pytest.importorskip("msgpack")

    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")

    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    calls: dict[str, object] = {}

    class _FakePreprocessor:
        def __init__(self, *args, **kwargs):
            calls["preprocessor_init"] = True
            assert callable(kwargs["on_message_error"])

        def preprocess_to_cache(self, cache_file: Path):
            calls["cache_file"] = cache_file
            cache_file.write_bytes(b"cache")
            return SimpleNamespace(
                message_count=3,
                event_count=2,
                cache_size_bytes=5,
                duration_seconds=0.1,
            )

    class _FakeMessageLogger:
        def __init__(self, *args, **kwargs):
            calls["logger_init"] = True
            assert callable(kwargs["on_event_error"])

        def log_from_cache(
            self,
            cache_file: Path,
            *,
            expected_total_messages: int | None = None,
        ):
            calls["logged_cache"] = cache_file
            calls["expected_total_messages"] = expected_total_messages
            return SimpleNamespace(
                message_count=2, session_count=1, duration_seconds=0.2
            )

    class _FakeRecordingSession:
        def __init__(self, *args, **kwargs):
            calls["session_init"] = True

    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessagePreprocessor", _FakePreprocessor
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessageLogger", _FakeMessageLogger
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.RecordingSession", _FakeRecordingSession
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
    )

    item = importer.build_work_items()[0]
    importer.import_item(item)

    assert calls["preprocessor_init"] is True
    assert calls["logger_init"] is True
    assert calls["session_init"] is True
    assert Path(calls["logged_cache"]) == Path(calls["cache_file"])
    assert calls["expected_total_messages"] == 2
    assert not Path(calls["cache_file"]).exists()


def test_mcap_importer_reuses_cache_and_skips_preprocessing(
    monkeypatch,
    tmp_path: Path,
):
    pytest.importorskip("msgpack")

    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")

    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    calls: dict[str, object] = {}

    class _FailIfPreprocessorRuns:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "Preprocessor should not run when reusable cache exists"
            )

    class _FakeMessageLogger:
        def __init__(self, *args, **kwargs):
            calls["logger_init"] = True

        def log_from_cache(
            self,
            cache_file: Path,
            *,
            expected_total_messages: int | None = None,
        ):
            calls["logged_cache"] = cache_file
            calls["expected_total_messages"] = expected_total_messages
            return SimpleNamespace(
                message_count=1, session_count=1, duration_seconds=0.1
            )

    class _FakeRecordingSession:
        def __init__(self, *args, **kwargs):
            calls["session_init"] = True

    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessagePreprocessor",
        _FailIfPreprocessorRuns,
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessageLogger", _FakeMessageLogger
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.RecordingSession", _FakeRecordingSession
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
    )

    cache_path = tmp_path / "reuse_cache.msgpack"
    monkeypatch.setattr(importer, "_get_cache_path", lambda _mcap: cache_path)

    with MessageCache(cache_path, mode="wb") as cache:
        cache.write_message(
            CachedMessage(
                data_type=DataType.CUSTOM_1D.value,
                name="value",
                timestamp=1.0,
                log_time_ns=1,
                transformed_data=np.array([1.0], dtype=np.float32),
                source_topic="/topic",
            )
        )

    metadata_path = importer._get_cache_metadata_path(cache_path)
    importer._write_cache_metadata(
        mcap_file=mcap_path,
        cache_path=cache_path,
        cache_metadata_path=metadata_path,
        message_count=1,
        event_count=1,
        cache_size_bytes=cache_path.stat().st_size,
    )

    item = importer.build_work_items()[0]
    importer.import_item(item)

    assert calls["logger_init"] is True
    assert calls["session_init"] is True
    assert calls["expected_total_messages"] == 1
    assert not cache_path.exists()
    assert not metadata_path.exists()


def test_mcap_importer_preserves_cache_when_logging_fails(monkeypatch, tmp_path: Path):
    pytest.importorskip("msgpack")

    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")

    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    class _FakePreprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def preprocess_to_cache(self, cache_file: Path):
            cache_file.write_bytes(b"cache")
            return SimpleNamespace(
                message_count=2,
                event_count=2,
                cache_size_bytes=5,
                duration_seconds=0.1,
            )

    class _FailingMessageLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_from_cache(
            self,
            cache_file: Path,
            *,
            expected_total_messages: int | None = None,
        ):
            raise RuntimeError("logging failed")

    class _FakeRecordingSession:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessagePreprocessor", _FakePreprocessor
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessageLogger", _FailingMessageLogger
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.RecordingSession", _FakeRecordingSession
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
    )

    cache_path = tmp_path / "preserve_cache.msgpack"
    monkeypatch.setattr(importer, "_get_cache_path", lambda _mcap: cache_path)
    metadata_path = importer._get_cache_metadata_path(cache_path)

    item = importer.build_work_items()[0]
    with pytest.raises(RuntimeError, match="logging failed"):
        importer.import_item(item)

    assert cache_path.exists()
    assert metadata_path.exists()


def test_mcap_importer_rebuilds_cache_once_if_missing_before_replay(
    monkeypatch,
    tmp_path: Path,
):
    pytest.importorskip("msgpack")

    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")

    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    calls = {"preprocess": 0, "log_from_cache": 0}

    class _FakePreprocessor:
        def __init__(self, *args, **kwargs):
            pass

        def preprocess_to_cache(self, cache_file: Path):
            calls["preprocess"] += 1
            cache_file.write_bytes(b"cache")
            return SimpleNamespace(
                message_count=1,
                event_count=1,
                cache_size_bytes=5,
                duration_seconds=0.1,
            )

    class _FlakyMessageLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_from_cache(
            self,
            cache_file: Path,
            *,
            expected_total_messages: int | None = None,
        ):
            calls["log_from_cache"] += 1
            if calls["log_from_cache"] == 1:
                cache_file.unlink()
                raise FileNotFoundError(
                    2,
                    "No such file or directory",
                    str(cache_file),
                )
            return SimpleNamespace(
                message_count=1, session_count=1, duration_seconds=0.1
            )

    class _FakeRecordingSession:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessagePreprocessor", _FakePreprocessor
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.MessageLogger", _FlakyMessageLogger
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap_importer.RecordingSession", _FakeRecordingSession
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
    )

    cache_path = tmp_path / "retry_cache.msgpack"
    monkeypatch.setattr(importer, "_get_cache_path", lambda _mcap: cache_path)
    metadata_path = importer._get_cache_metadata_path(cache_path)

    item = importer.build_work_items()[0]
    importer.import_item(item)

    assert calls["preprocess"] == 2
    assert calls["log_from_cache"] == 2
    assert not cache_path.exists()
    assert not metadata_path.exists()


def test_mcap_importer_phase_error_step_mode_enqueues_worker_error(
    monkeypatch,
    tmp_path: Path,
):
    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")
    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
        skip_on_error="step",
    )

    item = importer.build_work_items()[0]
    captured_logs: list[tuple[int, int | None, str]] = []
    captured_errors: list[object] = []
    importer._worker_id = 3
    importer._error_queue = SimpleNamespace(put=lambda err: captured_errors.append(err))
    monkeypatch.setattr(
        importer,
        "_log_worker_error",
        lambda worker_id, item_index, message: captured_logs.append(
            (worker_id, item_index, message)
        ),
    )

    handled = importer._handle_phase_error(
        item=item,
        phase="preprocess",
        unit_label="message",
        unit_index=7,
        topic="/topic",
        name=None,
        log_time_ns=55,
        exc=ValueError("bad payload"),
    )

    assert handled is True
    assert len(captured_errors) == 1
    assert captured_errors[0].worker_id == 3
    assert captured_errors[0].item_index == item.index
    assert "preprocess message 7" in captured_errors[0].message
    assert "topic=/topic" in captured_errors[0].message
    assert "log_time_ns=55" in captured_errors[0].message
    assert captured_logs


@pytest.mark.parametrize("mode", ["episode", "all"])
def test_mcap_importer_phase_error_non_step_mode_raises(
    monkeypatch,
    tmp_path: Path,
    mode: str,
):
    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")
    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
        skip_on_error=mode,
    )
    item = importer.build_work_items()[0]
    with pytest.raises(ImportError, match="logging event 2"):
        importer._handle_phase_error(
            item=item,
            phase="logging",
            unit_label="event",
            unit_index=2,
            topic="/topic",
            name="value",
            log_time_ns=99,
            exc=RuntimeError("boom"),
        )


def test_mcap_importer_progress_reporter_uses_base_emit_progress(
    monkeypatch,
    tmp_path: Path,
):
    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")
    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda _path: [mcap_path]),
    )

    def _fake_init_runtime(self):
        self._decoder_factories = []
        self._image_decoder = object()
        self._message_decoder = object()

    monkeypatch.setattr(
        MCAPDatasetImporter, "_init_runtime_components", _fake_init_runtime
    )

    importer = MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
    )
    item = importer.build_work_items()[0]
    emitted: list[tuple[int, int, int | None, str | None]] = []
    importer._worker_id = 0
    importer._progress_queue = object()
    monkeypatch.setattr(
        importer,
        "_emit_progress",
        lambda item_index, step, total_steps, episode_label: emitted.append(
            (item_index, step, total_steps, episode_label)
        ),
    )

    reporter = importer._create_progress_reporter(item, "episode_001.mcap")
    assert isinstance(reporter, EmitProgressReporter)
    assert reporter.report_every == importer.config.progress_emit_interval
    reporter.start_phase("preprocess", 10)
    reporter.update(3)
    reporter.finish_phase()

    assert emitted[0] == (item.index, 0, 10, "episode_001.mcap [preprocess]")
    assert emitted[-1] == (item.index, 3, 10, "episode_001.mcap [preprocess]")
