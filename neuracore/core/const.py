import os

LIVE_DATA_ENABLED = os.getenv("NEURACORE_LIVE_DATA_ENABLED", "True").lower() == "true"
API_URL = os.getenv("NEURACORE_API_URL", "https://api.neuracore.app/api")
MAX_DATA_STREAMS = 50
