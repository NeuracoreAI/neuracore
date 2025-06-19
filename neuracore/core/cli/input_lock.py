"""Global lock to simultaneous input requests."""

import threading

user_input_lock = threading.Lock()
