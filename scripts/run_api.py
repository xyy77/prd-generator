"""Start the FastAPI server for PRD Generator.

Usage:
    python scripts/run_api.py
    # or: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from src.config import settings


def main():
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
        timeout_keep_alive=900,  # 15 min for long multi-agent runs
    )


if __name__ == "__main__":
    main()
