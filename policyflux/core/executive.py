"""Abstract base for executive branch actors across political systems."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from enum import Enum

from .types import PolicySpace
from .bill_template import Bill


class ExecutiveType(Enum):
    """Types of executive systems."""
    PRESIDENTIAL = "presidential"
    PARLIAMENTARY = "parliamentary"
    SEMI_PRESIDENTIAL = "semi_presidential"
    

class ExecutiveActor(ABC):
    """Base class for all executive branch actors (Presidents, PMs, etc.)."""
    
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
    
    @abstractmethod
    def get_influence_on_bill(self, bill: Bill, **context) -> float:
        """Return influence score [0-1] on bill passage.
        
        Args:
            bill: The bill being voted on
            **context: Additional context (party strength, cohabitation, etc.)
            
        Returns:
            Float between 0-1 representing influence strength
        """
        pass
    
    @abstractmethod
    def can_veto_bill(self, bill: Bill) -> bool:
        """Check if this executive actor can veto the bill."""
        pass


class Executive(ABC):
    """Container for executive branch configuration."""
    
    def __init__(self, executive_type: ExecutiveType):
        self.executive_type = executive_type
    
    @abstractmethod
    def get_primary_actor(self) -> ExecutiveActor:
        """Return the main executive actor (President or PM)."""
        pass
    
    @abstractmethod
    def inject_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add executive-specific context for voting.
        
        This method is called before casting votes to inject executive
        influence parameters into the voting context.
        """
        pass
    
    @abstractmethod
    def process_bill_result(self, bill: Bill, votes_for: int, total_votes: int) -> int:
        """Process bill result through executive powers (veto, etc.).
        
        Args:
            bill: The bill that was voted on
            votes_for: Number of votes in favor
            total_votes: Total number of voters
            
        Returns:
            Final vote count after executive processing (may be 0 if vetoed)
        """
        pass