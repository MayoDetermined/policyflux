from policyflux.layers.media_pressure import MediaPressureLayer


def test_media_pressure_construction() -> None:
    layer = MediaPressureLayer(pressure=0.3)
    assert layer.name == "MediaPressure"


def test_media_pressure_call_in_range() -> None:
    layer = MediaPressureLayer(pressure=0.5)
    result = layer.call([0.5, 0.5], base_prob=0.5)
    assert 0.0 <= result <= 1.0


def test_media_pressure_compile_no_raise() -> None:
    layer = MediaPressureLayer(pressure=0.0)
    layer.compile()
