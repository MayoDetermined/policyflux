import pandas as pd
import pytest

from policyflux.exceptions import OptionalDependencyError
from policyflux.layers import ideal_point as ip


def _install_fake_torch_nn(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDevice:
        def __init__(self, device_type: str = "cpu") -> None:
            self.type = device_type

    class _FakeTensor:
        def __init__(self, data, device_type: str = "cpu") -> None:
            self.data = data
            self.device = _FakeDevice(device_type)

        def cpu(self):
            self.device = _FakeDevice("cpu")
            return self

        def to(self, dtype=None):
            return self

    class _FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class _FakeTorch:
        float32 = "float32"

        @staticmethod
        def tensor(data, dtype=None):
            return _FakeTensor(data)

        @staticmethod
        def sigmoid(x):
            return x

        @staticmethod
        def no_grad():
            return _FakeNoGrad()

        @staticmethod
        def cat(tensors, dim=1):
            return _FakeTensor(("cat", [t.data for t in tensors], dim))

    class _FakeLinear:
        def __init__(self, in_dim: int, out_dim: int) -> None:
            self.in_dim = in_dim
            self.out_dim = out_dim

        def __call__(self, x):
            return x

    class _FakePassThrough:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, x):
            return x

    class _FakeSequential:
        def __init__(self, *layers) -> None:
            self.layers = layers

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class _FakeNN:
        Linear = _FakeLinear
        ReLU = _FakePassThrough
        Dropout = _FakePassThrough
        Sigmoid = _FakePassThrough
        Sequential = _FakeSequential

    monkeypatch.setattr(ip, "HAS_TORCH", True)
    monkeypatch.setattr(ip, "torch", _FakeTorch)
    monkeypatch.setattr(ip, "nn", _FakeNN)


def test_ideal_point_encoder_df_raises_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ip, "HAS_TORCH", False)

    with pytest.raises(OptionalDependencyError):
        ip.IdealPointEncoderDF(output_dim=2, dataset=pd.DataFrame({"a": [1], "b": [2]}))


def test_ideal_point_encoder_df_forward_and_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_nn(monkeypatch)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    encoder = ip.IdealPointEncoderDF(output_dim=2, dataset=df)

    forwarded = encoder.forward("x")
    encoded = encoder.encode(df)

    assert forwarded == "x"
    assert hasattr(encoded, "data")


def test_ideal_point_encoder_df_encode_mismatch_and_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_nn(monkeypatch)
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    encoder = ip.IdealPointEncoderDF(output_dim=2, dataset=df)

    with pytest.raises(ValueError):
        encoder.encode(pd.DataFrame({"a": [1.0]}))

    monkeypatch.setattr(ip, "HAS_TORCH", False)
    with pytest.raises(OptionalDependencyError):
        encoder.forward("x")
    with pytest.raises(OptionalDependencyError):
        encoder.encode(df)


def test_ideal_point_text_encoder_raises_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ip, "HAS_TORCH", False)

    with pytest.raises(OptionalDependencyError):
        ip.IdealPointTextEncoder(output_dim=2, corpus=["a b c"])


def test_ideal_point_text_encoder_raises_when_embeddings_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_nn(monkeypatch)
    monkeypatch.setattr(ip, "HAS_SENTENCE_TRANSFORMERS", False)

    with pytest.raises(OptionalDependencyError):
        ip.IdealPointTextEncoder(output_dim=2, corpus=["alpha beta gamma"], use_embeddings=True)


def test_ideal_point_text_encoder_encode_df_and_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_nn(monkeypatch)
    encoder = ip.IdealPointTextEncoder(
        output_dim=2,
        corpus=["hello world", "world policy"],
        use_embeddings=False,
        hidden_dims=[4],
    )

    single = encoder.encode("hello world")
    multiple = encoder.encode(["hello", "world"])
    assert hasattr(single, "data")
    assert hasattr(multiple, "data")

    df = pd.DataFrame({"text": ["alpha beta", "beta gamma"]})
    encoded_df = encoder.encode_df(df, text_column="text")
    assert hasattr(encoded_df, "data")

    with pytest.raises(ValueError):
        encoder.encode_df(df, text_column="missing")

    class _FakeOptimizer:
        def __init__(self) -> None:
            self.zeroed = False
            self.stepped = False

        def zero_grad(self) -> None:
            self.zeroed = True

        def step(self) -> None:
            self.stepped = True

    class _FakeLoss:
        def __init__(self) -> None:
            self.backprop = False

        def backward(self) -> None:
            self.backprop = True

        def item(self) -> float:
            return 0.123

    fake_loss = _FakeLoss()

    def _criterion(predictions, targets):
        return fake_loss

    optimizer = _FakeOptimizer()
    loss_value = encoder.train_step(
        texts=["alpha beta", "beta gamma"],
        targets="targets",
        optimizer=optimizer,
        criterion=_criterion,
    )

    assert loss_value == pytest.approx(0.123)
    assert optimizer.zeroed is True
    assert optimizer.stepped is True
    assert fake_loss.backprop is True


def test_ideal_point_text_encoder_embeddings_extract_and_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_nn(monkeypatch)
    encoder = ip.IdealPointTextEncoder(
        output_dim=2,
        corpus=["alpha beta", "beta gamma"],
        use_embeddings=False,
    )

    class _FakeEmbeddingModel:
        def encode(self, texts, convert_to_tensor=True, show_progress_bar=False):
            return ip.torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    encoder.use_embeddings = True
    encoder.embedding_model = _FakeEmbeddingModel()
    features = encoder._extract_features(["a b", "b c"])

    assert hasattr(features, "data")
    assert features.data[0] == "cat"

    monkeypatch.setattr(ip, "HAS_TORCH", False)
    with pytest.raises(OptionalDependencyError):
        encoder.forward("x")
    with pytest.raises(OptionalDependencyError):
        encoder._extract_features(["a"])
    with pytest.raises(OptionalDependencyError):
        encoder.encode(["a"])
    with pytest.raises(OptionalDependencyError):
        encoder.train_step(["a"], "t", optimizer=object(), criterion=object())
