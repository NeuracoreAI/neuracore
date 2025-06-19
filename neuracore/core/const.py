"""Configuration constants for Neuracore client behavior."""

import os

# Disabling these can help when running tests or if you just want to run a
# local endpoint
REMOTE_RECORDING_TRIGGER_ENABLED = (
    os.getenv("NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED", "True").lower() == "true"
)
LIVE_DATA_ENABLED = os.getenv("NEURACORE_LIVE_DATA_ENABLED", "True").lower() == "true"

API_URL = os.getenv("NEURACORE_API_URL", "https://api.neuracore.app/api")
MAX_DATA_STREAMS = 50


MAX_INPUT_ATTEMPTS = 3
CONFIRMATION_INPUT = {
    "yes",
    "y",
    "ok",
    "okay",
    "sure",
    "confirm",
    "agreed",
    "accept",
    "proceed",
    "go ahead",
    "yeah",
    "yep",
    "absolutely",
    "true",
    "continue",
    "do it",
}

REJECTION_INPUT = {
    "no",
    "n",
    "cancel",
    "decline",
    "disagree",
    "reject",
    "stop",
    "nope",
    "false",
    "not now",
    "abort",
    "never",
    "don't",
    "exit",
    "quit",
}
