"""Quick test to verify all integrations work correctly."""

import sys
sys.path.insert(0, '.')

from policyflux.integration import (
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
    build_executive,
    build_engine,
)
from policyflux.core.executive import ExecutiveType

def test_imports():
    """Test that all imports work."""
    print("[OK] All imports successful")

def test_presidential_config():
    """Test presidential config creation."""
    config = create_presidential_config(
        num_actors=10,
        iterations=5,
        president_approval=0.6
    )
    assert config.actors_config.executive_type == ExecutiveType.PRESIDENTIAL
    print("[OK] Presidential config created")

def test_parliamentary_config():
    """Test parliamentary config creation."""
    config = create_parliamentary_config(
        num_actors=10,
        iterations=5,
        pm_party_strength=0.55
    )
    assert config.actors_config.executive_type == ExecutiveType.PARLIAMENTARY
    assert config.layer_config.include_government_agenda == True
    print("[OK] Parliamentary config created (government agenda auto-enabled)")

def test_semi_presidential_config():
    """Test semi-presidential config creation."""
    config = create_semi_presidential_config(
        num_actors=10,
        iterations=5,
        president_approval=0.5,
        pm_party_strength=0.55
    )
    assert config.actors_config.executive_type == ExecutiveType.SEMI_PRESIDENTIAL
    print("[OK] Semi-presidential config created")

def test_executive_building():
    """Test executive building."""
    # Presidential
    config = create_presidential_config(num_actors=10, iterations=5)
    exec1 = build_executive(config)
    assert exec1 is not None
    assert exec1.executive_type == ExecutiveType.PRESIDENTIAL
    print("[OK] Presidential executive built")

    # Parliamentary
    config = create_parliamentary_config(num_actors=10, iterations=5)
    exec2 = build_executive(config)
    assert exec2 is not None
    assert exec2.executive_type == ExecutiveType.PARLIAMENTARY
    print("[OK] Parliamentary executive built")

    # Semi-Presidential
    config = create_semi_presidential_config(num_actors=10, iterations=5)
    exec3 = build_executive(config)
    assert exec3 is not None
    assert exec3.executive_type == ExecutiveType.SEMI_PRESIDENTIAL
    print("[OK] Semi-presidential executive built")

def test_engine_building():
    """Test that engines can be built."""
    config = create_presidential_config(num_actors=5, iterations=2)
    engine = build_engine(config)
    assert engine is not None
    print("[OK] Engine built successfully")

def test_layer_registry():
    """Test that government agenda layer is registered."""
    from policyflux.integration import LAYER_REGISTRY
    assert "government_agenda" in LAYER_REGISTRY
    print("[OK] GovernmentAgendaLayer registered")

def main():
    print("=" * 60)
    print("PolicyFlux Integration Tests")
    print("=" * 60)

    try:
        test_imports()
        test_presidential_config()
        test_parliamentary_config()
        test_semi_presidential_config()
        test_executive_building()
        test_engine_building()
        test_layer_registry()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nKey features verified:")
        print("  * All executive types can be configured")
        print("  * Executive systems build correctly")
        print("  * Parliamentary system auto-enables government agenda layer")
        print("  * Engine building works end-to-end")
        print("  * Layer registry includes government agenda")
        print("\nProject fully integrated and ready to use!")
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
