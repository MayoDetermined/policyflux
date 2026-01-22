from threading import Lock
from typing import Dict


class IdGenerator:
    """
    Thread-safe centralized ID generator for unique identifiers across entities.
    """
    
    _instance = None
    _lock = Lock()
    _counters: Dict[str, int] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._counters = {
            'actor': 0,
            'layer': 0,
            'bill': 0,
            'model': 0,
        }
        self._initialized = True
    
    def generate_actor_id(self) -> int:
        """Generate unique actor ID."""
        with self._lock:
            self._counters['actor'] += 1
            return self._counters['actor']
    
    def generate_layer_id(self) -> int:
        """Generate unique layer ID."""
        with self._lock:
            self._counters['layer'] += 1
            return self._counters['layer']
    
    def generate_bill_id(self) -> int:
        """Generate unique bill ID."""
        with self._lock:
            self._counters['bill'] += 1
            return self._counters['bill']
    
    def generate_model_id(self) -> int:
        """Generate unique model ID."""
        with self._lock:
            self._counters['model'] += 1
            return self._counters['model']
    
    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        with self._lock:
            for key in self._counters:
                self._counters[key] = 0


def get_id_generator() -> IdGenerator:
    """Get the singleton IdGenerator instance."""
    return IdGenerator()
