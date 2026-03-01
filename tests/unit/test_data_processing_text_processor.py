import pytest

from policyflux.data_processing import text_processor as tp
from policyflux.exceptions import OptionalDependencyError


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTorch:
        long = "long"

        @staticmethod
        def tensor(data, dtype=None):
            return {"data": list(data), "dtype": dtype}

    def _fake_pad_sequence(batch, batch_first=True, padding_value=0):
        max_len = max((len(item["data"]) for item in batch), default=0)
        padded = [item["data"] + [padding_value] * (max_len - len(item["data"])) for item in batch]
        return {
            "data": padded,
            "batch_first": batch_first,
            "padding_value": padding_value,
        }

    monkeypatch.setattr(tp, "HAS_TORCH", True)
    monkeypatch.setattr(tp, "torch", _FakeTorch)
    monkeypatch.setattr(tp, "pad_sequence", _fake_pad_sequence)


def test_vectorizer_init_raises_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tp, "HAS_TORCH", False)

    with pytest.raises(OptionalDependencyError):
        tp.SimpleTextVectorizer(["a b c"])


def test_tokenize_build_vocab_and_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    vectorizer = tp.SimpleTextVectorizer(["Hello, world!", "hello policyflux"])

    assert vectorizer.tokenize("Hi, TEST!") == ["hi", "test"]

    vectorizer.build_vocab(min_freq=2)
    assert vectorizer.word_to_idx["<pad>"] == 0
    assert vectorizer.word_to_idx["<unk>"] == 1
    assert "hello" in vectorizer.word_to_idx
    assert "world" not in vectorizer.word_to_idx

    indices = vectorizer.text_pipeline("hello unknown token")
    assert indices[0] == vectorizer.word_to_idx["hello"]
    assert indices[1] == 1
    assert indices[2] == 1


def test_collect_batch_uses_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    vectorizer = tp.SimpleTextVectorizer(["alpha beta", "alpha"])
    vectorizer.build_vocab()

    batch = vectorizer.collect_batch(["alpha beta", "alpha"])

    assert batch["batch_first"] is True
    assert batch["padding_value"] == 0
    assert len(batch["data"]) == 2
    assert len(batch["data"][0]) == len(batch["data"][1])


def test_collect_batch_raises_when_torch_unavailable_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)
    vectorizer = tp.SimpleTextVectorizer(["text"])

    monkeypatch.setattr(tp, "HAS_TORCH", False)
    monkeypatch.setattr(tp, "pad_sequence", None)

    with pytest.raises(OptionalDependencyError):
        vectorizer.collect_batch(["text"])


def test_fit_process_and_vectorize(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    vectorizer = tp.SimpleTextVectorizer(["seed text"])

    vectorizer.fit(["A A B", "B C"])
    assert "a" in vectorizer.word_to_idx
    assert "b" in vectorizer.word_to_idx

    processed = vectorizer.process(["A B", "C"])
    assert isinstance(processed, list)
    assert len(processed) == 2

    fresh = tp.SimpleTextVectorizer(["x y z"])
    fresh.word_to_idx = {}
    vec = fresh.vectorize("x")
    assert isinstance(vec, dict)
    assert vec["dtype"] == "long"


def test_vectorize_raises_when_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    vectorizer = tp.SimpleTextVectorizer(["x"])

    monkeypatch.setattr(tp, "HAS_TORCH", False)
    with pytest.raises(OptionalDependencyError):
        vectorizer.vectorize("x")
