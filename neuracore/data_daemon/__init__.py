"""Python-side interface to the Neuracore data daemon.

The daemon itself is a native binary built from
[rust/data_daemon](../../rust/data_daemon); this package holds the SDK's glue
to it — the ``_data_bridge`` PyO3 extension, the bundled ``bin/data-daemon``
binary, and the thin modules that launch it and ship data to it.
"""
