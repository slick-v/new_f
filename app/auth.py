from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import DEMO_API_KEYS

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)):
    # No keys configured = open access, for local development
    if DEMO_API_KEYS and api_key not in DEMO_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")