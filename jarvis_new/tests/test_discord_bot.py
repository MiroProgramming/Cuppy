import time
from types import SimpleNamespace

import discord
import pytest

from discord_bot import (
    ChatTools,
    CuppyChat,
    call_tool,
    html_to_text,
    is_owner_dm,
    parse_search_results,
)
from watches import WatchStore

DDG_PAGE = """
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.dubizzle.com.eg%2Fad%2F42&amp;rut=abc">
     Australian <b>Shepherd</b> puppy - Cairo</a>
  <a class="result__snippet" href="x">Blue merle, 3 months, 15,000 EGP</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/guide">Aussie care guide</a>
</div>
"""


def test_parse_search_results():
    results = parse_search_results(DDG_PAGE)
    assert results[0] == {
        "title": "Australian Shepherd puppy - Cairo",
        "url": "https://www.dubizzle.com.eg/ad/42",
        "snippet": "Blue merle, 3 months, 15,000 EGP",
    }
    assert results[1]["url"] == "https://example.com/guide"
    assert results[1]["snippet"] == ""


def test_html_to_text_skips_scripts():
    title, text = html_to_text(
        "<html><head><title>Pups</title><style>x{}</style></head>"
        "<body><script>evil()</script><h1>Hello</h1><p>World</p></body></html>"
    )
    assert title == "Pups"
    assert text == "Hello World"


async def test_call_tool_reports_errors():
    async def boom():
        raise ValueError("nope")

    assert await call_tool({"boom": boom}, "boom", {}) == {"error": "nope"}
    assert "Unknown tool" in (await call_tool({}, "missing", {}))["error"]


async def test_watch_tools(tmp_path):
    tools = ChatTools(WatchStore(tmp_path))
    created = await tools.create_watch("pups", "pups egypt", 1)
    assert created["every_minutes"] == 60
    assert len((await tools.list_watches())["watches"]) == 1
    assert (await tools.cancel_watch(created["id"]))["cancelled"]


def test_only_owner_dms_are_answered():
    bot = object()
    dm = discord.DMChannel.__new__(discord.DMChannel)
    owner = SimpleNamespace(id=1)
    stranger = SimpleNamespace(id=2)
    assert is_owner_dm(SimpleNamespace(author=owner, channel=dm), 1, bot)
    assert not is_owner_dm(SimpleNamespace(author=stranger, channel=dm), 1, bot)
    assert not is_owner_dm(SimpleNamespace(author=owner, channel=object()), 1, bot)


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    async def send_message(self, message):
        self.sent.append(message)
        return self.responses.pop(0)


async def test_reply_runs_tools_then_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    tools = ChatTools(WatchStore(tmp_path))
    chat = CuppyChat(tools)
    fake = FakeChat(
        [
            SimpleNamespace(
                function_calls=[
                    SimpleNamespace(
                        name="create_watch",
                        args={
                            "description": "Aussie pups Egypt",
                            "search_query": "aussie egypt",
                        },
                    )
                ],
                text=None,
            ),
            SimpleNamespace(function_calls=None, text="Watching for pups, Sir."),
        ]
    )
    chat._chat = fake
    chat._last_message = time.monotonic()

    assert (
        await chat.reply("keep an eye out for aussie puppies")
        == "Watching for pups, Sir."
    )
    assert [w.description for w in tools.store.list()] == ["Aussie pups Egypt"]
    function_response = fake.sent[1][0].function_response
    assert function_response.name == "create_watch"
    assert function_response.response["created"] is True
