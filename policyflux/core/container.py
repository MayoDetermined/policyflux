from typing import Callable, TypeVar, Generic, Dict, Any

T = TypeVar('T')

class ServiceContainer:
    """Lightweight dependency injection container."""
    
    def __init__(self):
        self._factories: Dict[type, Callable[..., Any]] = {}
        self._singletons: Dict[type, Any] = {}
    
    def register_factory(self, interface: type, factory: Callable[..., T]) -> None:
        """Register a factory function for a service."""
        self._factories[interface] = factory
    
    def register_singleton(self, interface: type, instance: T) -> None:
        """Register a singleton instance."""
        self._singletons[interface] = instance
    
    def resolve(self, interface: type[T]) -> T:
        """Resolve a service instance."""
        if interface in self._singletons:
            return self._singletons[interface]
        
        if interface not in self._factories:
            raise ValueError(f"No factory registered for {interface}")
        
        return self._factories[interface](self)