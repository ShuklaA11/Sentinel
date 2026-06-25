"""Package init — load repo-root .env for local dev runs.

No-op in CI (where the .env doesn't exist and secrets are real env vars) and no-op if
python-dotenv isn't installed.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass
