def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Please update your 'hppfcl' imports to 'coal':DeprecationWarning",
    )
