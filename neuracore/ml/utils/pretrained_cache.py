"""Pull individual Hugging Face pretrained models via Neuracore signed URLs."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
from huggingface_hub.constants import HF_HUB_CACHE

logger = logging.getLogger(__name__)


def is_hf_hub_repo_id(model_path: str | os.PathLike[str]) -> bool:
    """Return True if ``model_path`` looks like a Hugging Face repo id."""
    path = os.fspath(model_path)
    if not path or path.startswith((".", "/")):
        return False
    if os.path.isdir(path) or os.path.isfile(path):
        return False
    if "/" in path:
        return True
    return not path.endswith(
        (".safetensors", ".json", ".pt", ".bin", ".ckpt", ".tar.gz")
    )


def hf_hub_cache_dir_name(repo_id: str) -> str:
    """Return the Hugging Face hub cache directory name for a repo id."""
    return "models--" + repo_id.replace("/", "--")


def _repo_cached_locally(repo_id: str) -> bool:
    snapshots_dir = Path(HF_HUB_CACHE) / hf_hub_cache_dir_name(repo_id) / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    return any(child.is_dir() for child in snapshots_dir.iterdir())


def _get_pretrained_download_url(repo_id: str) -> str | None:
    try:
        import neuracore as nc
        from neuracore.core.auth import get_auth
        from neuracore.core.config.get_current_org import get_current_org
        from neuracore.core.const import API_URL
        from neuracore.core.utils.http_session import thread_local_session
    except ImportError:
        return None

    try:
        auth = get_auth()
        if not auth.is_authenticated:
            nc.login()
        org_id = get_current_org()
        session = thread_local_session()
        url = f"{API_URL}/org/{org_id}/pretrained-models/download_url"
        response = session.get(
            url,
            params={"repo_id": repo_id},
            headers=get_auth().get_headers(),
            timeout=60,
        )
        if response.status_code == 401:
            nc.login()
            response = session.get(
                url,
                params={"repo_id": repo_id},
                headers=get_auth().get_headers(),
                timeout=60,
            )
        if response.status_code != 200:
            logger.warning(
                "Pretrained model download URL request failed for %s: %s %s",
                repo_id,
                response.status_code,
                response.text,
            )
            return None
        download_url = response.json().get("url")
        if not isinstance(download_url, str) or not download_url:
            logger.warning("Pretrained model download URL missing for %s", repo_id)
            return None
        return download_url
    except Exception as exc:
        logger.warning(
            "Failed to request pretrained model download URL for %s: %s",
            repo_id,
            exc,
        )
        return None


def _download_and_extract_archive(download_url: str, repo_id: str) -> bool:
    archive_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            archive_path = tmp.name
            with requests.get(download_url, stream=True, timeout=300) as response:
                response.raise_for_status()
                shutil.copyfileobj(response.raw, tmp)

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(path=Path(HF_HUB_CACHE))

        return _repo_cached_locally(repo_id)
    except Exception as exc:
        logger.warning(
            "Failed to download or extract pretrained model archive for %s: %s",
            repo_id,
            exc,
        )
        return False
    finally:
        if archive_path and os.path.exists(archive_path):
            os.unlink(archive_path)


def ensure_pretrained_model(repo_id: str) -> bool:
    """Ensure a Hugging Face repo is present in the local hub cache.

    Requests a signed download URL from the Neuracore API, downloads the model
    archive, and extracts it into the standard Hugging Face hub cache. Falls
    back silently when unauthenticated or the download fails (callers may still
    use Hugging Face Hub).

    Args:
        repo_id: Hugging Face model id (e.g. ``lerobot/pi0_base``).

    Returns:
        True if the repo is available locally after this call (including cache hit).
    """
    logger.info("Ensuring pretrained model %s is cached locally", repo_id)
    if not is_hf_hub_repo_id(repo_id):
        return False

    if _repo_cached_locally(repo_id):
        logger.info("Pretrained model already cached locally: %s", repo_id)
        return True

    logger.info(
        "Pretrained model %s not in local cache; requesting Neuracore archive",
        repo_id,
    )
    download_url = _get_pretrained_download_url(repo_id)
    if not download_url:
        logger.info(
            "No Neuracore archive available for %s; falling back to Hugging Face Hub",
            repo_id,
        )
        return False

    logger.info("Downloading pretrained model archive for %s", repo_id)
    if _download_and_extract_archive(download_url, repo_id):
        logger.info("Loaded pretrained model %s from Neuracore archive", repo_id)
        return True

    logger.warning(
        "Pretrained model %s archive did not produce a local cache; "
        "falling back to Hugging Face Hub",
        repo_id,
    )
    return False
