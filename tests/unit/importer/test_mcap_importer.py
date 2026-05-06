from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from neuracore_types import DataType
from neuracore_types.nc_data import DatasetImportConfig
from PIL import Image

from neuracore.core.utils.depth_utils import MAX_DEPTH
from neuracore.importer.core.base import ImportItem
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.mcap.mcap_importer import MCAPDatasetImporter
from neuracore.importer.mcap.utils import (
    JSONDecoderFactory,
    MCAPSourceEvent,
    RawPassthroughDecoderFactory,
    build_topic_map,
    clip_depth,
    convert_decoded_mcap_data,
    estimate_total_messages,
    get_mcap_topics,
    iter_mcap_source_events,
    list_decoder_factories,
    read_image_data,
    resolve_path,
    resolve_timestamp_seconds,
    split_topic_path,
    to_numpy,
    to_python_types,
    validate_requested_topics,
)


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


def _make_dataset_config(data_import_config: dict) -> DatasetImportConfig:
    return cast(
        DatasetImportConfig,
        SimpleNamespace(
            data_import_config=data_import_config,
            robot=SimpleNamespace(name="test_robot"),
            frequency=30,
        ),
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
    topic_map = build_topic_map(
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

    topics = get_mcap_topics(topic_map)
    assert topics == ["/camera/color", "/camera/color/image_cam1"]

    relative_cfg = topic_map["/camera/color"][0]
    absolute_cfg = topic_map["/camera/color/image_cam1"][0]

    assert [item.name for item in relative_cfg.import_config.mapping] == ["color_main"]
    assert absolute_cfg.mapping_item.name == "cam_left"
    assert absolute_cfg.item_base_path == ["data"]


def test_topic_mapper_requires_source_for_relative_items():
    with pytest.raises(ImportError, match="Relative mapping entries require"):
        build_topic_map(
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


def test_preprocessor_estimate_total_messages():
    summary = SimpleNamespace(
        statistics=SimpleNamespace(channel_message_counts={1: 10, 2: 5}),
        channels={
            1: SimpleNamespace(topic="/a"),
            2: SimpleNamespace(topic="/b"),
        },
    )
    assert estimate_total_messages(summary, ["/a", "/b"]) == 15


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
    image_data = read_image_data(
        DataType.DEPTH_IMAGES,
        message["data"],
        message,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(image_data, np.ndarray)
    assert image_data.shape == (1, 2)


def _make_importer(
    monkeypatch,
    tmp_path: Path,
    *,
    dry_run: bool = False,
    skip_on_error: str = "episode",
) -> MCAPDatasetImporter:
    """Build a minimally patched MCAPDatasetImporter for testing."""
    mcap_path = tmp_path / "episode_001.mcap"
    mcap_path.write_bytes(b"mcap")

    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        staticmethod(lambda dataset_dir: [mcap_path]),
    )
    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_init_runtime_components",
        lambda self: setattr(self, "_decoder_factories", []),
    )

    return MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=_make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic",
                mapping=[_make_mapping_item("value")],
            )
        }),
        dry_run=dry_run,
        skip_on_error=skip_on_error,
    )


def test_mcap_importer_build_work_items(monkeypatch, tmp_path: Path):
    importer = _make_importer(monkeypatch, tmp_path)
    items = list(importer.build_work_items())
    assert len(items) == 1
    assert items[0].description == "episode_001.mcap"
    assert items[0].metadata["path"].endswith("episode_001.mcap")


def test_mcap_importer_import_item_starts_and_stops_recording(
    monkeypatch, tmp_path: Path
):
    calls: list[str] = []

    monkeypatch.setattr(
        "neuracore.importer.mcap.mcap_importer.nc.start_recording",
        lambda robot_name, instance: calls.append("start"),
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap.mcap_importer.nc.stop_recording",
        lambda robot_name, instance, wait: calls.append("stop"),
    )

    importer = _make_importer(monkeypatch, tmp_path)
    monkeypatch.setattr(importer, "_stream_episode_file", lambda *_args, **_kwargs: 3)

    importer.import_item(importer.build_work_items()[0])

    assert calls == ["start", "stop"]


def test_mcap_importer_dry_run_skips_recording(monkeypatch, tmp_path: Path):
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("dry-run must not call start/stop recording")

    monkeypatch.setattr(
        "neuracore.importer.mcap.mcap_importer.nc.start_recording", _unexpected
    )
    monkeypatch.setattr(
        "neuracore.importer.mcap.mcap_importer.nc.stop_recording", _unexpected
    )

    importer = _make_importer(monkeypatch, tmp_path, dry_run=True)
    monkeypatch.setattr(importer, "_stream_episode_file", lambda *_args, **_kwargs: 0)

    importer.import_item(importer.build_work_items()[0])


def test_mcap_importer_record_step_logs_each_event(monkeypatch, tmp_path: Path):
    logged: list[tuple[DataType, str]] = []

    importer = _make_importer(monkeypatch, tmp_path)
    monkeypatch.setattr(
        importer,
        "_log_data",
        lambda data_type, source_data, item, format, timestamp: logged.append(
            (data_type, item.name)
        ),
    )

    topic_map = build_topic_map(
        _make_dataset_config({
            DataType.CUSTOM_1D: _make_import_config(
                source="/topic.values",
                mapping=[
                    _make_mapping_item("a"),
                    _make_mapping_item("b"),
                ],
            )
        })
    )
    importer.topic_map = topic_map

    importer._record_step(  # noqa: SLF001
        step={"/topic": {"values": {"a": 1.0, "b": 2.0}}},
        timestamp=0.5,
    )

    assert len(logged) == 2
    assert {name for _, name in logged} == {"a", "b"}


@pytest.mark.parametrize(
    "source,match",
    [
        ("", "Source must include a topic"),
        (".field", "topic segment is empty"),
    ],
)
def test_split_topic_path_raises(source, match):
    with pytest.raises(ImportError, match=match):
        split_topic_path(source)


def test_split_topic_path_no_subpath():
    topic, path = split_topic_path("/joint_states")
    assert topic == "/joint_states"
    assert path == []


def test_resolve_path_numeric_string_as_integer_dict_key():
    assert resolve_path({0: "zero"}, ["0"]) == "zero"


def test_resolve_path_attribute_access_on_object():
    assert resolve_path(SimpleNamespace(x=42), ["x"]) == 42


def test_resolve_path_index_on_list():
    assert resolve_path({"items": [10, 20, 30]}, ["items", "2"]) == 30


def test_resolve_path_missing_key_raises():
    with pytest.raises(ImportError, match="Key 'missing' not found"):
        resolve_path({"a": 1}, ["missing"])


@pytest.mark.parametrize(
    "log_ns,pub_ns,expected",
    [
        (2_000_000_000, 0, 2.0),
        (0, 1_000_000_000, 1.0),
    ],
)
def test_resolve_timestamp_seconds(log_ns, pub_ns, expected):
    assert resolve_timestamp_seconds(
        log_time_ns=log_ns, publish_time_ns=pub_ns
    ) == pytest.approx(expected)


def test_resolve_timestamp_falls_back_to_wall_clock():
    before = time.time()
    ts = resolve_timestamp_seconds(log_time_ns=0, publish_time_ns=0)
    assert before <= ts <= time.time()


def test_validate_requested_topics_raises_on_missing():
    summary = SimpleNamespace(channels={1: SimpleNamespace(topic="/present")})
    with pytest.raises(ImportError, match="/missing"):
        validate_requested_topics(summary, ["/missing"])


def test_validate_requested_topics_passes_when_all_present():
    summary = SimpleNamespace(
        channels={1: SimpleNamespace(topic="/a"), 2: SimpleNamespace(topic="/b")}
    )
    validate_requested_topics(summary, ["/a", "/b"])  # must not raise


@pytest.mark.parametrize("value", [None, True, 42, 3.14, "hello"])
def test_to_python_types_primitives_passthrough(value):
    assert to_python_types(value) == value


def test_to_python_types_numpy_scalar_becomes_python():
    result = to_python_types(np.float32(1.5))
    assert isinstance(result, float)
    assert result == pytest.approx(1.5)


def test_to_python_types_bytearray_becomes_bytes():
    assert to_python_types(bytearray(b"\x01\x02")) == b"\x01\x02"


def test_to_python_types_numpy_array_passthrough():
    arr = np.array([1, 2, 3])
    assert to_python_types(arr) is arr


def test_to_python_types_dict_all_primitive_same_object():
    d = {"a": 1, "b": "x"}
    assert to_python_types(d) is d


def test_to_python_types_dict_with_nested_recurses():
    result = to_python_types({"a": SimpleNamespace(x=1)})
    assert result == {"a": {"x": 1}}


def test_to_python_types_list_all_primitive_same_object():
    lst = [1, 2, 3]
    assert to_python_types(lst) is lst


def test_to_python_types_list_with_nested_recurses():
    result = to_python_types([SimpleNamespace(x=1)])
    assert result == [{"x": 1}]


def test_to_python_types_object_with_dict_attrs():
    result = to_python_types(SimpleNamespace(a=1, b="hello"))
    assert result == {"a": 1, "b": "hello"}


def test_to_python_types_unknown_falls_back_to_repr():
    class _Opaque:
        __slots__ = ()

        def __repr__(self):
            return "opaque"

    assert to_python_types(_Opaque()) == "opaque"


def test_to_numpy_list_becomes_ndarray():
    result = to_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


@pytest.mark.parametrize("value", [5, 2.5])
def test_to_numpy_scalar_becomes_float64(value):
    assert isinstance(to_numpy(value), np.float64)


def test_to_numpy_bool_passthrough():
    assert to_numpy(True) is True


def test_to_numpy_ndarray_passthrough():
    arr = np.array([1.0])
    assert to_numpy(arr) is arr


def test_to_numpy_tensor_with_numpy_method():
    class _FakeTensor:
        def numpy(self):
            return np.array([7.0])

    np.testing.assert_array_equal(to_numpy(_FakeTensor()), [7.0])


@pytest.mark.parametrize("value", ["not an array", 42, None, [1, 2, 3]])
def test_clip_depth_non_array_passthrough(value):
    assert clip_depth(value) == value


def test_read_image_data_non_image_type_passthrough():
    data = {"position": 1.0}
    assert (
        read_image_data(
            DataType.JOINT_POSITIONS, data, data, logger=logging.getLogger(__name__)
        )
        is data
    )


def test_read_image_data_decodes_raw_rgb8():
    h, w = 2, 3
    raw = np.zeros((h, w, 3), dtype=np.uint8)
    raw[0, 0] = [255, 0, 0]
    message = SimpleNamespace(
        height=h,
        width=w,
        encoding="rgb8",
        step=w * 3,
        is_bigendian=False,
        data=raw.tobytes(),
    )
    result = read_image_data(
        DataType.RGB_IMAGES, message.data, message, logger=logging.getLogger(__name__)
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (h, w, 3)


def test_read_image_data_decodes_bigendian_mono16():
    raw = np.array([[1000, 2000]], dtype=">u2")
    message = SimpleNamespace(
        height=1,
        width=2,
        encoding="mono16",
        step=4,
        is_bigendian=True,
        data=raw.tobytes(),
    )

    result = read_image_data(
        DataType.DEPTH_IMAGES,
        message.data,
        message,
        logger=logging.getLogger(__name__),
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([[1000, 2000]], dtype=np.float32))


def test_read_image_data_raises_on_unsupported_encoding():
    message = SimpleNamespace(
        height=1,
        width=1,
        encoding="yuv422",
        step=2,
        is_bigendian=False,
        data=b"\x00\x00",
    )
    with pytest.raises(ImportError, match="Unsupported image encoding"):
        read_image_data(
            DataType.RGB_IMAGES,
            message.data,
            message,
            logger=logging.getLogger(__name__),
        )


def test_convert_decoded_mcap_data_converts_non_protobuf():
    result = convert_decoded_mcap_data({"a": SimpleNamespace(x=1)})
    assert result == {"a": {"x": 1}}


def test_convert_decoded_mcap_data_plain_dict_passthrough():
    d = {"a": 1}
    assert convert_decoded_mcap_data(d) is d


def test_mcap_importer_getstate_clears_decoder_factories(monkeypatch, tmp_path):
    importer = _make_importer(monkeypatch, tmp_path)
    importer._decoder_factories = [RawPassthroughDecoderFactory()]
    state = importer.__getstate__()
    assert state["_decoder_factories"] is None


def test_mcap_importer_prepare_worker_re_initializes_factories(monkeypatch, tmp_path):
    from neuracore.importer.core.base import NeuracoreDatasetImporter

    importer = _make_importer(monkeypatch, tmp_path)
    importer._decoder_factories = None
    inited = []
    monkeypatch.setattr(
        NeuracoreDatasetImporter,
        "prepare_worker",
        lambda self, worker_id, chunk=None: None,
    )
    monkeypatch.setattr(importer, "_init_runtime_components", lambda: inited.append(1))
    importer.prepare_worker(0)
    assert inited


def test_mcap_importer_import_item_missing_file_raises(monkeypatch, tmp_path):
    importer = _make_importer(monkeypatch, tmp_path)
    bad_item = ImportItem(
        index=0, description="gone.mcap", metadata={"path": str(tmp_path / "gone.mcap")}
    )
    with pytest.raises(ImportError, match="MCAP file not found"):
        importer.import_item(bad_item)


def test_discover_mcap_files_single_mcap_file(tmp_path):
    f = tmp_path / "episode.mcap"
    f.write_bytes(b"mcap")
    assert MCAPDatasetImporter._discover_mcap_files(f) == [f]


def test_discover_mcap_files_rejects_non_mcap_file(tmp_path):
    f = tmp_path / "data.bag"
    f.write_bytes(b"bag")
    with pytest.raises(ImportError, match="Expected an MCAP file"):
        MCAPDatasetImporter._discover_mcap_files(f)


def test_discover_mcap_files_raises_on_nonexistent_path(tmp_path):
    with pytest.raises(ImportError, match="does not exist"):
        MCAPDatasetImporter._discover_mcap_files(tmp_path / "nonexistent")


def test_discover_mcap_files_raises_when_no_mcap_in_dir(tmp_path):
    (tmp_path / "data.txt").write_text("hello")
    with pytest.raises(ImportError, match="No MCAP files found"):
        MCAPDatasetImporter._discover_mcap_files(tmp_path)


def test_iter_mcap_source_events_yields_source_event():
    topic_map = build_topic_map(
        _make_dataset_config({
            DataType.JOINT_POSITIONS: _make_import_config(
                source="/joint.position",
                mapping=[_make_mapping_item("joint_1")],
            )
        })
    )

    decoded_data = {"position": [1, 2, 3]}
    events = list(
        iter_mcap_source_events(
            "/joint",
            decoded_data,
            topic_map=topic_map,
            logger=logging.getLogger(__name__),
            timestamp=1.0,
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], MCAPSourceEvent)
    assert events[0].data_type == DataType.JOINT_POSITIONS
    assert events[0].item.name == "joint_1"
    np.testing.assert_array_equal(events[0].source_data, np.array([1, 2, 3]))


def test_iter_mcap_source_events_unknown_topic_yields_nothing():
    topic_map = build_topic_map(
        _make_dataset_config({
            DataType.JOINT_POSITIONS: _make_import_config(
                source="/joint.position",
                mapping=[_make_mapping_item("j1")],
            )
        })
    )

    events = list(
        iter_mcap_source_events(
            "/unknown",
            {},
            topic_map=topic_map,
            logger=logging.getLogger(__name__),
            timestamp=0.0,
        )
    )

    assert events == []


def test_mcap_importer_log_transformed_clips_depth(monkeypatch, tmp_path):
    importer = _make_importer(monkeypatch, tmp_path)
    logged = []
    monkeypatch.setattr(
        type(importer).__mro__[1],
        "_log_transformed_data",
        lambda self, data_type, transformed_data, name, timestamp, **kw: logged.append(
            (data_type, transformed_data)
        ),
    )
    oversized = np.array([[MAX_DEPTH + 100.0]], dtype=np.float32)
    importer._log_transformed_data(  # noqa: SLF001
        data_type=DataType.DEPTH_IMAGES,
        transformed_data=oversized,
        name="depth",
        timestamp=0.0,
    )
    assert len(logged) == 1
    assert float(np.max(logged[0][1])) <= MAX_DEPTH
