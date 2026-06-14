"""
Entry point — used by Lightning AI `run` command.
Lightning AI exposes port 7860 by default.
"""
import uvicorn
from app.core.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
    )
