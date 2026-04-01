"""Root conftest for loading test environment variables."""

from dotenv import load_dotenv

load_dotenv(".vscode/test.env", override=True)
