"""Configure a package-level logger for policyflux."""
import logging
from .config import get_settings


def configure_logging():
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


configure_logging()

# Export a package logger
logger = logging.getLogger("policyflux")
