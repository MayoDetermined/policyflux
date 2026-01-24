"""PolicyFlux package public surface.

Recommended imports:

- For core abstractions: `from policyflux.core import Bill, CongressModel, Layer`
- For models (heavy imports) import when needed: `import policyflux.models as models`

This module exposes lightweight helpers (config, logger, RNG) and
re-exports core abstractions. Models are intentionally not imported at
package import time to avoid circular import issues; import them on demand.
"""

from .core import *  # noqa: F401,F403
from .config import get_settings, Settings  # noqa: F401
from .logging_config import logger  # noqa: F401
from .pfrandom import set_seed, get_rng, random  # noqa: F401


def import_models():
	"""Import `policyflux.models` on demand and return the module.

	Use this to avoid importing model implementations at package import time.
	"""
	import importlib

	return importlib.import_module("policyflux.models")
