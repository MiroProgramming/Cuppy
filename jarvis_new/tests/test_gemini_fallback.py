from types import SimpleNamespace

import pytest
from google.genai import errors

from gemini_fallback import DEFAULT_MODELS, generate_with_fallback, model_chain


class FakeModels:
    def __init__(self, failures):
        self.failures = failures  # model -> error code
        self.tried = []

    async def generate_content(self, model, contents, config=None):
        self.tried.append(model)
        if model in self.failures:
            code = self.failures[model]
            cls = errors.ServerError if code >= 500 else errors.ClientError
            raise cls(code, {"error": {"code": code, "message": "busy"}})
        return f"answer from {model}"


def fake_client(failures):
    models = FakeModels(failures)
    return SimpleNamespace(aio=SimpleNamespace(models=models)), models


async def test_falls_back_when_busy():
    client, models = fake_client({"a": 503, "b": 429})
    result = await generate_with_fallback(client, ["a", "b", "c"], "hi", retry_delay=0)
    assert result == "answer from c"
    assert models.tried == ["a", "b", "c"]


async def test_raises_when_all_fail():
    client, _ = fake_client({"a": 503, "b": 503})
    with pytest.raises(errors.ServerError):
        await generate_with_fallback(client, ["a", "b"], "hi", retry_delay=0)


async def test_bad_request_is_not_retried():
    client, models = fake_client({"a": 400})
    with pytest.raises(errors.ClientError):
        await generate_with_fallback(client, ["a", "b"], "hi", retry_delay=0)
    assert models.tried == ["a"]


def test_model_chain(monkeypatch):
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    assert model_chain("CHAT_MODEL") == DEFAULT_MODELS
    monkeypatch.setenv("CHAT_MODEL", "gemini-2.5-flash")
    chain = model_chain("CHAT_MODEL")
    assert chain[0] == "gemini-2.5-flash" and chain.count("gemini-2.5-flash") == 1
