"""Tests for Neuracore pretrained model archive cache helpers."""

from unittest.mock import MagicMock, patch

from neuracore.ml.utils.pretrained_cache import (
    ensure_pretrained_model,
    hf_hub_cache_dir_name,
    is_hf_hub_repo_id,
)


def test_is_hf_hub_repo_id() -> None:
    assert is_hf_hub_repo_id("lerobot/pi0_base")
    assert is_hf_hub_repo_id("nvidia/GR00T-N1.6-3B")
    assert is_hf_hub_repo_id("distilbert-base-uncased")
    assert not is_hf_hub_repo_id("/tmp/checkpoint")
    assert not is_hf_hub_repo_id("model.safetensors")


def test_hf_hub_cache_dir_name() -> None:
    assert hf_hub_cache_dir_name("lerobot/pi0_base") == "models--lerobot--pi0_base"


@patch("neuracore.ml.utils.pretrained_cache._repo_cached_locally", return_value=True)
def test_ensure_pretrained_model_cache_hit(_cached: MagicMock) -> None:
    assert ensure_pretrained_model("lerobot/pi0_base") is True
    _cached.assert_called_once_with("lerobot/pi0_base")


@patch(
    "neuracore.ml.utils.pretrained_cache._download_and_extract_archive",
    return_value=True,
)
@patch(
    "neuracore.ml.utils.pretrained_cache._get_pretrained_download_url",
    return_value="https://signed.example/archive.tar.gz",
)
@patch("neuracore.ml.utils.pretrained_cache._repo_cached_locally", return_value=False)
def test_ensure_pretrained_model_downloads_archive(
    _cached: MagicMock,
    get_url: MagicMock,
    extract: MagicMock,
) -> None:
    assert ensure_pretrained_model("distilbert-base-uncased") is True
    get_url.assert_called_once_with("distilbert-base-uncased")
    extract.assert_called_once_with(
        "https://signed.example/archive.tar.gz",
        "distilbert-base-uncased",
    )


@patch(
    "neuracore.ml.utils.pretrained_cache._get_pretrained_download_url",
    return_value=None,
)
@patch("neuracore.ml.utils.pretrained_cache._repo_cached_locally", return_value=False)
def test_ensure_pretrained_model_skips_without_download_url(
    _cached: MagicMock,
    _get_url: MagicMock,
) -> None:
    assert ensure_pretrained_model("lerobot/pi05_base") is False


@patch(
    "neuracore.ml.utils.pretrained_cache._download_and_extract_archive",
    return_value=False,
)
@patch(
    "neuracore.ml.utils.pretrained_cache._get_pretrained_download_url",
    return_value="https://signed.example/archive.tar.gz",
)
@patch("neuracore.ml.utils.pretrained_cache._repo_cached_locally", return_value=False)
def test_ensure_pretrained_model_handles_extract_failure(
    _cached: MagicMock,
    _get_url: MagicMock,
    _extract: MagicMock,
) -> None:
    assert ensure_pretrained_model("lerobot/pi05_base") is False
