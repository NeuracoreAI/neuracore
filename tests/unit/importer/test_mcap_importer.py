from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from neuracore_types import DataType
from neuracore_types.importer.mcap import StagedRecord, TopicImportConfig

from neuracore.importer.core.base import NeuracoreDatasetImporter
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.mcap_importer import MCAPDatasetImporter


def _make_mapping_item(
    name: str, source_name: str | None = None, index=None, index_range=None
):
    return SimpleNamespace(
        name=name,
        source_name=source_name,
        index=index,
        index_range=index_range,
    )


def _make_import_config(source: str, mapping: list[SimpleNamespace]):
    return SimpleNamespace(
        source=source,
        mapping=mapping,
        format=SimpleNamespace(language_type=None),
    )


def _make_importer_with_config(data_import_config: dict):
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer.dataset_config = SimpleNamespace(data_import_config=data_import_config)
    return importer


def test_build_topic_map_with_absolute_mapping_topics():
    importer = _make_importer_with_config({
        DataType.RGB_IMAGES: _make_import_config(
            source="/camera/color",
            mapping=[
                _make_mapping_item(
                    name="cam_left", source_name="/camera/color/image_cam1.data"
                ),
                _make_mapping_item(
                    name="cam_right", source_name="/camera/color/image_cam2.data"
                ),
            ],
        )
    })

    topic_map = importer._build_topic_map()

    assert sorted(topic_map.keys()) == [
        "/camera/color/image_cam1",
        "/camera/color/image_cam2",
    ]
    left_cfg = topic_map["/camera/color/image_cam1"][0]
    right_cfg = topic_map["/camera/color/image_cam2"][0]
    assert left_cfg.mapping_item.name == "cam_left"
    assert left_cfg.item_base_path == ["data"]
    assert left_cfg.source_path == []
    assert right_cfg.mapping_item.name == "cam_right"
    assert right_cfg.item_base_path == ["data"]
    assert right_cfg.source_path == []


def test_build_topic_map_with_absolute_mapping_topics_without_source():
    importer = _make_importer_with_config({
        DataType.RGB_IMAGES: _make_import_config(
            source="",
            mapping=[
                _make_mapping_item(
                    name="cam_left", source_name="/camera/color/image_cam1.data"
                ),
                _make_mapping_item(
                    name="cam_right", source_name="/camera/color/image_cam2.data"
                ),
            ],
        )
    })

    topic_map = importer._build_topic_map()

    assert sorted(topic_map.keys()) == [
        "/camera/color/image_cam1",
        "/camera/color/image_cam2",
    ]
    left_cfg = topic_map["/camera/color/image_cam1"][0]
    right_cfg = topic_map["/camera/color/image_cam2"][0]
    assert left_cfg.mapping_item.name == "cam_left"
    assert right_cfg.mapping_item.name == "cam_right"


def test_build_topic_map_without_absolute_mapping_topics_preserves_legacy():
    importer = _make_importer_with_config({
        DataType.RGB_IMAGES: _make_import_config(
            source="/camera/color/image.data",
            mapping=[_make_mapping_item(name="color_main", source_name="pixels")],
        )
    })

    topic_map = importer._build_topic_map()

    assert sorted(topic_map.keys()) == ["/camera/color/image"]
    cfg = topic_map["/camera/color/image"][0]
    assert cfg.mapping_item is None
    assert cfg.item_base_path is None
    assert cfg.source_path == ["data"]


def test_build_topic_map_with_mixed_absolute_and_relative_items():
    importer = _make_importer_with_config({
        DataType.RGB_IMAGES: _make_import_config(
            source="/camera/color",
            mapping=[
                _make_mapping_item(
                    name="cam_left", source_name="/camera/color/image_cam1.data"
                ),
                _make_mapping_item(name="color_main", source_name="image"),
            ],
        )
    })

    topic_map = importer._build_topic_map()

    assert sorted(topic_map.keys()) == ["/camera/color", "/camera/color/image_cam1"]
    relative_cfg = topic_map["/camera/color"][0]
    absolute_cfg = topic_map["/camera/color/image_cam1"][0]
    assert [item.name for item in relative_cfg.import_config.mapping] == ["color_main"]
    assert relative_cfg.source_path == []
    assert absolute_cfg.mapping_item.name == "cam_left"
    assert absolute_cfg.item_base_path == ["data"]


def test_build_topic_map_requires_source_for_relative_mappings():
    importer = _make_importer_with_config({
        DataType.RGB_IMAGES: _make_import_config(
            source="",
            mapping=[_make_mapping_item(name="color_main", source_name="image")],
        )
    })

    with pytest.raises(ImportError, match="Relative mapping entries require"):
        importer._build_topic_map()


def test_record_step_absolute_mapping_uses_full_message_context():
    mapping_item = _make_mapping_item(
        name="cam_left", source_name="/camera/color/image_cam1.data"
    )
    import_config = _make_import_config(source="/camera/color", mapping=[mapping_item])
    cfg = TopicImportConfig(
        data_type=DataType.RGB_IMAGES,
        import_config=import_config,
        source_path=[],
        mapping_item=mapping_item,
        item_base_path=["data"],
    )

    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._topic_map = {"/camera/color/image_cam1": [cfg]}

    calls = {}

    def _fake_coerce(data_type, data, message):
        calls["coerce"] = (data_type, data, message)
        return data

    def _fake_log_data(data_type, source_data, item, data_format, timestamp):
        calls["log"] = (data_type, source_data, item, data_format, timestamp)

    importer._coerce_message_data = _fake_coerce
    importer._to_numpy = lambda data: data
    importer._log_data = _fake_log_data

    message_data = {
        "data": b"\x01\x02",
        "height": 1,
        "width": 2,
        "encoding": "mono8",
    }
    importer._record_step({"/camera/color/image_cam1": message_data}, 123.0)

    assert calls["coerce"][1] == b"\x01\x02"
    assert calls["coerce"][2] is message_data
    assert calls["log"][1] == b"\x01\x02"
    assert calls["log"][2] is mapping_item
    assert calls["log"][4] == 123.0


def test_record_step_relative_source_path_uses_full_message_context():
    mapping_item = _make_mapping_item(name="left_cam")
    import_config = _make_import_config(
        source="/left_camera/image/compressed.data", mapping=[mapping_item]
    )
    cfg = TopicImportConfig(
        data_type=DataType.RGB_IMAGES,
        import_config=import_config,
        source_path=["data"],
    )

    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._topic_map = {"/left_camera/image/compressed": [cfg]}

    calls = {}

    def _fake_coerce(data_type, data, message):
        calls["coerce"] = (data_type, data, message)
        return data

    def _fake_log_data(data_type, source_data, item, data_format, timestamp):
        calls["log"] = (data_type, source_data, item, data_format, timestamp)

    importer._coerce_message_data = _fake_coerce
    importer._to_numpy = lambda data: data
    importer._log_data = _fake_log_data

    message_data = {
        "data": b"\x89PNG...",
        "format": "rgb8; jpeg compressed bgr8",
    }
    importer._record_step({"/left_camera/image/compressed": message_data}, 456.0)

    assert calls["coerce"][1] == b"\x89PNG..."
    assert calls["coerce"][2] is message_data
    assert calls["log"][1] == b"\x89PNG..."
    assert calls["log"][2] is mapping_item
    assert calls["log"][4] == 456.0


def test_validate_requested_topics_raises_when_missing():
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    summary = SimpleNamespace(
        channels={
            1: SimpleNamespace(topic="/camera/color/image"),
            2: SimpleNamespace(topic="/camera/depth/image"),
        }
    )

    with pytest.raises(ImportError, match="Configured topic\\(s\\) not present"):
        importer._validate_requested_topics(
            summary, ["/camera/color/image", "/camera/color/image_cam1"]
        )


def test_decode_message_falls_back_to_raw_payload_for_unknown_encoding():
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._decoder_cache = {}
    importer._decoder_factories = []
    importer._raw_channel_ids = set()
    importer.logger = logging.getLogger(__name__)

    schema = SimpleNamespace(id=42, name="CustomSchema", encoding="custom")
    channel = SimpleNamespace(
        topic="/custom/topic", message_encoding="unknown-encoding"
    )
    message = SimpleNamespace(
        channel_id=7, data=b"\x01\x02", log_time=100, publish_time=90
    )

    decoded = importer._decode_message(schema, channel, message)

    assert decoded["data"] == b"\x01\x02"
    assert decoded["topic"] == "/custom/topic"
    assert decoded["message_encoding"] == "unknown-encoding"
    assert decoded["schema"] == {"id": 42, "name": "CustomSchema", "encoding": "custom"}
    assert decoded["log_time_ns"] == 100
    assert decoded["publish_time_ns"] == 90

    next_message = SimpleNamespace(
        channel_id=7, data=b"\x03", log_time=110, publish_time=100
    )
    decoded_next = importer._decode_message(schema, channel, next_message)
    assert decoded_next["data"] == b"\x03"


def test_decode_message_falls_back_to_raw_payload_when_decoder_raises():
    def _bad_decoder(_data):
        raise ValueError("decode failed")

    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._decoder_cache = {}
    importer._decoder_factories = [
        SimpleNamespace(
            decoder_for=lambda _encoding, _schema: _bad_decoder,
        )
    ]
    importer._raw_channel_ids = set()
    importer.logger = logging.getLogger(__name__)

    schema = SimpleNamespace(id=11, name="BadSchema", encoding="custom")
    channel = SimpleNamespace(topic="/bad/topic", message_encoding="custom")
    message = SimpleNamespace(
        channel_id=12, data=b"\x09\x08", log_time=200, publish_time=190
    )

    decoded = importer._decode_message(schema, channel, message)

    assert decoded["data"] == b"\x09\x08"
    assert decoded["topic"] == "/bad/topic"
    assert 12 in importer._raw_channel_ids


def test_staged_record_round_trip(tmp_path):
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    staged_path = tmp_path / "staged.bin"

    expected = [
        StagedRecord("/topic/a", 1.25, 123, {"value": [1, 2, 3]}),
        StagedRecord("/topic/b", 2.50, 456, b"\x00\x01"),
    ]
    with staged_path.open("wb") as staged_file:
        for record in expected:
            importer._write_staged_record(
                staged_file=staged_file,
                topic=record.topic,
                timestamp=record.timestamp,
                log_time_ns=record.log_time_ns,
                payload=record.payload,
            )

    actual = list(importer._iter_staged_records(staged_path))
    assert actual == expected


def test_validate_decoder_support_warns_when_decoder_missing(caplog):
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._decoder_factories = []
    importer.logger = logging.getLogger(__name__)

    summary = SimpleNamespace(
        schemas={1: SimpleNamespace(id=1, name="SomeSchema", encoding="protobuf")},
        channels={
            1: SimpleNamespace(
                topic="/camera/color/image",
                message_encoding="protobuf",
                schema_id=1,
            )
        },
    )

    with caplog.at_level(logging.WARNING):
        importer._validate_decoder_support(summary, ["/camera/color/image"])

    assert any(
        "No decoder currently available for topic '/camera/color/image'"
        in record.getMessage()
        for record in caplog.records
    )


def test_to_plain_data_handles_circular_references():
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._to_plain_max_depth = 64
    importer._to_plain_max_repr_chars = 128

    payload = {"name": "root"}
    payload["self"] = payload

    converted = importer._to_plain_data(payload)
    assert converted["name"] == "root"
    assert converted["self"] == "<circular-reference:dict>"


def test_to_plain_data_truncates_large_repr():
    class _HugeRepr:
        def __repr__(self) -> str:
            return "x" * 500

    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    importer._to_plain_max_depth = 64
    importer._to_plain_max_repr_chars = 64

    converted = importer._to_plain_data(_HugeRepr())
    assert isinstance(converted, str)
    assert converted.startswith("x" * 64)
    assert "truncated" in converted


def test_decode_raw_image_rejects_oversized_buffer():
    importer = MCAPDatasetImporter.__new__(MCAPDatasetImporter)
    message = {
        "height": 1,
        "width": 2,
        "encoding": "mono8",
        "step": 2,
        "is_bigendian": 0,
    }

    with pytest.raises(ImportError, match="too large"):
        importer._decode_raw_image(DataType.DEPTH_IMAGES, b"\x01\x02\x03", message)


def test_init_honors_skip_on_error_all(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_super_init(self, **kwargs):
        captured.update(kwargs)
        self.logger = logging.getLogger(__name__)
        self.skip_on_error = kwargs["skip_on_error"]

    monkeypatch.setattr(NeuracoreDatasetImporter, "__init__", _fake_super_init)
    monkeypatch.setattr(
        MCAPDatasetImporter,
        "_discover_mcap_files",
        lambda self, _dataset_dir: [],
    )
    monkeypatch.setattr(MCAPDatasetImporter, "_build_topic_map", lambda self: {})
    monkeypatch.setattr(
        MCAPDatasetImporter, "_build_decoder_factories", lambda self: []
    )

    MCAPDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=tmp_path,
        dataset_config=SimpleNamespace(data_import_config={}),
        skip_on_error="all",
    )

    assert captured["max_workers"] == 1
    assert captured["skip_on_error"] == "all"
