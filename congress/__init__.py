"""Congressional Dynamics Modeling - Core Module.

This package contains the core abstractions for congressional voting dynamics:
- CongressMan: Individual legislator agent with ideology, loyalty, and voting behavior
- TheCongress: System of agents with network influence and global context
"""

from congress.actors import CongressMan
from congress.congress import TheCongress

__all__ = ["CongressMan", "TheCongress"]
