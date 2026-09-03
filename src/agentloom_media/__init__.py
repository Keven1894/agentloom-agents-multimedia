"""AgentLoom Multimedia Knowledge Ingestion Agent."""

from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

__version__ = "0.1.0"
