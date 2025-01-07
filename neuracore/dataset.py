import logging

import requests

from neuracore.const import API_URL

from .auth import Auth, get_auth

logger = logging.getLogger(__name__)


class Dataset:

    def __init__(
        self, name: str, description: str | None = None, tags: list[str] | None = None
    ):
        datasets = self._get_datasets()
        for dataset in datasets:
            if dataset["name"] == name:
                self.id = dataset["id"]
                logger.info(f"Dataset '{name}' already exist.")
                return
        id = self._create_dataset(name, description, tags)
        self.id = id

    def _create_dataset(
        self, name: str, description: str | None = None, tags: list[str] | None = None
    ):
        auth: Auth = get_auth()
        response = requests.post(
            f"{API_URL}/datasets",
            headers=auth.get_headers(),
            json={
                "name": name,
                "description": description,
            },
        )
        response.raise_for_status()
        dataset_json = response.json()
        return dataset_json["id"]

    def _get_datasets(self):
        auth: Auth = get_auth()
        response = requests.get(f"{API_URL}/datasets", headers=auth.get_headers())
        response.raise_for_status()
        return response.json()
