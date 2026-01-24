try:
    # pydantic v1 / v2 compatibility: BaseSettings moved to pydantic_settings in v2
    from pydantic import BaseSettings  # type: ignore
except Exception:
    from pydantic_settings import BaseSettings  # type: ignore
from functools import lru_cache


class Settings(BaseSettings):
    """Central configuration for policyflux.

    Values can be overridden via environment variables with prefix
    `POLICYFLUX_` (e.g. `POLICYFLUX_SEED`, `POLICYFLUX_LOG_LEVEL`).
    """

    seed: int = 42
    log_level: str = "INFO"

    class Config:
        env_prefix = "POLICYFLUX_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
