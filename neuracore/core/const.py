"""Configuration constants for Neuracore client behavior."""

import os
from warnings import warn

API_URL = (os.getenv("NEURACORE_API_URL") or "https://api.neuracore.app/api").strip()

STANDARD_API_URLS = {
    "https://api.neuracore.app/api",
    "https://staging.api.neuracore.app/api",
    "http://localhost:8000/api",
}

if API_URL not in STANDARD_API_URLS:
    warn(f"API Base Url {API_URL} is non-standard, are you sure it is correct?")


API_URL = os.getenv("NEURACORE_API_URL", "https://api.neuracore.app/api")
MAX_DATA_STREAMS = 300
MAX_INPUT_ATTEMPTS = 3

STREAMING_MINIMUM_BACKOFF_TIME_S = 0.05
STREAMING_MAXIMUM_BACKOFF_TIME_S = 5

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
    "q",
}


# Disabling these can help when running tests or if you just want to run a
# local endpoint
REMOTE_RECORDING_TRIGGER_ENABLED = (
    os.getenv("NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED") or "yes"
).lower().strip() in CONFIRMATION_INPUT
PROVIDE_LIVE_DATA = (
    os.getenv("NEURACORE_PROVIDE_LIVE_DATA") or "yes"
).lower().strip() in CONFIRMATION_INPUT
CONSUME_LIVE_DATA = (
    os.getenv("NEURACORE_CONSUME_LIVE_DATA") or "yes"
).lower().strip() in CONFIRMATION_INPUT
