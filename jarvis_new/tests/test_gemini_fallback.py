from types import SimpleNamespace

import pytest
from google.genai import errors

from gemini_fallback import (
    DEFAULT_MODELS,
    generate_with_fallback,
    model_chain,
    order_models,
    reset_health,
)


@pytest.fixture(autouse=True)
def clean_health():
    reset_health()
    yield
    reset_health()


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
    result = await generate_with_fallback(client, ["a", "b", "c"], "hi")
    assert result == "answer from c"
    assert models.tried == ["a", "b", "c"]


async def test_raises_when_all_fail():
    client, _ = fake_client({"a": 503, "b": 503})
    with pytest.raises(errors.ServerError):
        await generate_with_fallback(client, ["a", "b"], "hi")


async def test_bad_request_is_not_retried():
    client, models = fake_client({"a": 400})
    with pytest.raises(errors.ClientError):
        await generate_with_fallback(client, ["a", "b"], "hi")
    assert models.tried == ["a"]


def test_model_chain(monkeypatch):
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    assert model_chain("CHAT_MODEL") == DEFAULT_MODELS
    monkeypatch.setenv("CHAT_MODEL", "gemini-2.5-flash")
    chain = model_chain("CHAT_MODEL")
    assert chain[0] == "gemini-2.5-flash" and chain.count("gemini-2.5-flash") == 1


async def test_hanging_model_times_out_and_falls_back():
    import asyncio

    class SlowModels(FakeModels):
        async def generate_content(self, model, contents, config=None):
            if model == "slow":
                self.tried.append(model)
                await asyncio.sleep(10)
            return await super().generate_content(model, contents, config)

    models = SlowModels({})
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    result = await generate_with_fallback(client, ["slow", "fast"], "hi", timeout=0.05)
    assert result == "answer from fast"


async def test_failed_models_are_skipped_next_time():
    client, models = fake_client({"busy": 503, "gone": 404})
    chain = ["busy", "gone", "ok"]
    assert await generate_with_fallback(client, chain, "hi") == "answer from ok"
    assert models.tried == ["busy", "gone", "ok"]

    # Next reply goes straight to the model that works.
    models.tried.clear()
    assert await generate_with_fallback(client, chain, "hi") == "answer from ok"
    assert models.tried == ["ok"]
    assert order_models(chain)[-2:] == ["busy", "gone"]


async def test_recovered_model_is_used_again_after_cooldown():
    client, models = fake_client({"busy": 503})
    chain = ["busy", "ok"]
    await generate_with_fallback(client, chain, "hi")
    models.failures.clear()
    # Pretend the cooldown has passed.
    later = order_models(chain, now=10**12)
    assert later == ["ok", "busy"]  # last good still first, but busy is back in play
