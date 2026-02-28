from policyflux.core.id_generator import IdGenerator, get_id_generator


def test_id_generator_is_singleton() -> None:
    first = IdGenerator()
    second = IdGenerator()
    third = get_id_generator()

    assert first is second is third


def test_id_generator_increments_and_resets() -> None:
    generator = get_id_generator()

    assert generator.generate_actor_id() == 1
    assert generator.generate_actor_id() == 2
    assert generator.generate_layer_id() == 1

    generator.reset()

    assert generator.generate_actor_id() == 1
    assert generator.generate_layer_id() == 1
