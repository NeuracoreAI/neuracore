"""Tests for SerializedSynchronizedPoints and SerializedSynchronizedEpisode."""

import pytest

from neuracore.core.data.serialized_synchronized_episode import (
    SerializedSynchronizedEpisode,
    SerializedSynchronizedPoints,
)


def test_round_trips_every_sync_point(synced_data):
    """Return sync points equal to the ones stored, in order."""
    stored = SerializedSynchronizedPoints(synced_data.observations)

    assert len(stored) == len(synced_data.observations)
    assert [point.model_dump(mode="json") for point in stored] == [
        point.model_dump(mode="json") for point in synced_data.observations
    ]


def test_supports_negative_indices(synced_data):
    """Index from the end like a list."""
    stored = SerializedSynchronizedPoints(synced_data.observations)

    assert stored[-1].model_dump(mode="json") == (
        synced_data.observations[-1].model_dump(mode="json")
    )


def test_rejects_out_of_range_index(synced_data):
    """Raise IndexError past the last sync point."""
    stored = SerializedSynchronizedPoints(synced_data.observations)

    with pytest.raises(IndexError):
        stored[len(stored)]


def test_returns_a_new_object_on_each_read(synced_data):
    """Keep stored data unchanged when a returned sync point is modified."""
    stored = SerializedSynchronizedPoints(synced_data.observations)

    stored[0].timestamp = -1.0

    assert stored[0].timestamp == synced_data.observations[0].timestamp


def test_episode_keeps_metadata_and_serializes_points(synced_data):
    """Copy the episode metadata and serialize its points."""
    episode = SerializedSynchronizedEpisode(synced_data)

    assert (episode.robot_id, episode.start_time, episode.end_time) == (
        synced_data.robot_id,
        synced_data.start_time,
        synced_data.end_time,
    )
    assert isinstance(episode.observations, SerializedSynchronizedPoints)
    assert len(episode.observations) == len(synced_data.observations)
