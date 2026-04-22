"""Topic mapping utilities for MCAP imports.

This module resolves dataset config mappings into per-topic lookup structures so
message processing is O(1) at runtime. It preserves the existing absolute-topic
semantics where mapping items can override `source` with `/topic.subpath`.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

from neuracore_types import DataType
from neuracore_types.nc_data import DatasetImportConfig
from neuracore_types.nc_data.nc_data import MappingItem

from neuracore.importer.core.exceptions import ImportError

from .paths import split_topic_path


@dataclass(frozen=True, slots=True)
class TopicConfig:
    """Resolved topic configuration for one mapping entry group."""

    data_type: DataType
    import_config: Any
    source_path: list[str]
    mapping_item: MappingItem | None = None
    item_base_path: list[str] | None = None


class TopicMapper:
    """Build and query per-topic mapping configurations.

    Source paths may be absolute (prefixed with `/`) or relative to an import
    config's `source`. Absolute mapping items are split into dedicated topic
    configs so each message topic can be transformed independently.
    """

    def __init__(self, dataset_config: DatasetImportConfig) -> None:
        """Build topic lookup tables from dataset mapping configuration."""
        self._topic_map = self._build_topic_map(dataset_config)

    def get_configs_for_topic(self, topic: str) -> list[TopicConfig]:
        """Return mapping configs for a topic (empty if unmapped)."""
        return self._topic_map.get(topic, [])

    def get_all_topics(self) -> list[str]:
        """Return all configured topics in deterministic order."""
        return sorted(self._topic_map)

    def _build_topic_map(
        self, dataset_config: DatasetImportConfig
    ) -> dict[str, list[TopicConfig]]:
        topic_map: dict[str, list[TopicConfig]] = {}

        for data_type, import_config in dataset_config.data_import_config.items():
            source = (import_config.source or "").strip()
            mapping = list(import_config.mapping)

            absolute_items = [
                item
                for item in mapping
                if item.source_name and item.source_name.startswith("/")
            ]
            relative_items = [
                item
                for item in mapping
                if not (item.source_name and item.source_name.startswith("/"))
            ]

            if relative_items:
                if not source:
                    raise ImportError(
                        f"Missing source for data type '{data_type.value}'. "
                        "Relative mapping entries require a base source path."
                    )
                topic, subpath = split_topic_path(source)
                topic_map.setdefault(topic, []).append(
                    TopicConfig(
                        data_type=data_type,
                        import_config=self._copy_import_config_with_mapping(
                            import_config,
                            relative_items,
                        ),
                        source_path=subpath,
                    )
                )

            for item in absolute_items:
                item_topic, item_subpath = split_topic_path(item.source_name)
                topic_map.setdefault(item_topic, []).append(
                    TopicConfig(
                        data_type=data_type,
                        import_config=import_config,
                        source_path=[],
                        mapping_item=item,
                        item_base_path=item_subpath,
                    )
                )

        if not topic_map:
            raise ImportError("No data_import_config entries found for MCAP import.")

        return topic_map

    def _copy_import_config_with_mapping(
        self,
        import_config: Any,
        mapping: list[MappingItem],
    ) -> Any:
        """Clone an import config while replacing mapping entries."""
        mapping_copy = list(mapping)
        if hasattr(import_config, "model_copy"):
            return import_config.model_copy(update={"mapping": mapping_copy})
        cloned = copy(import_config)
        setattr(cloned, "mapping", mapping_copy)
        return cloned
