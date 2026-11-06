from types import SimpleNamespace

import discord

from discord_bot import (
    ChatTools,
    CuppyChat,
    call_tool,
    html_to_text,
    is_owner_dm,
    parse_search_results,
)
from shared_memory import ConversationLog
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


def make_response(text=None, calls=None):
    from google.genai import types

    content = types.Content(
        role="model",
        parts=[
            types.Part.from_function_call(name=c["name"], args=c["args"])
            for c in calls or []
        ]
        or [types.Part.from_text(text=text or "")],
    )
    return SimpleNamespace(
        function_calls=[SimpleNamespace(**c) for c in calls] if calls else None,
        text=text,
        candidates=[SimpleNamespace(content=content)],
    )


async def test_reply_runs_tools_then_answers_and_shares_history(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    tools = ChatTools(WatchStore(tmp_path))
    log = ConversationLog(tmp_path)
    log.append(
        "assistant", "Good evening, Sir", "voice"
    )  # history may start with Cuppy
    log.append("user", "My budget is 20,000 pounds", "voice")
    chat = CuppyChat(tools, log)

    responses = [
        make_response(
            calls=[
                {
                    "name": "create_watch",
                    "args": {
                        "description": "Aussie pups Egypt",
                        "search_query": "aussie egypt",
                    },
                }
            ]
        ),
        make_response(text="Watching for pups under your budget, Sir."),
    ]
    sent = []

    async def fake_generate(contents):
        sent.append(list(contents))
        return responses.pop(0)

    chat._generate = fake_generate

    answer = await chat.reply("keep an eye out for aussie puppies")
    assert answer == "Watching for pups under your budget, Sir."
    assert [w.description for w in tools.store.list()] == ["Aussie pups Egypt"]

    first = sent[0]
    assert first[0].role == "user"  # leading Cuppy line dropped
    assert "voice call] My budget is 20,000 pounds" in first[0].parts[0].text
    assert first[0].parts[1].text == "keep an eye out for aussie puppies"
    tool_reply = sent[1][-1].parts[0].function_response
    assert tool_reply.name == "create_watch" and tool_reply.response["created"]

    assert [(e["channel"], e["text"]) for e in log.recent()][-2:] == [
        ("discord", "keep an eye out for aussie puppies"),
        ("discord", "Watching for pups under your budget, Sir."),
    ]


async def test_reset_clears_shared_history(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    log = ConversationLog(tmp_path)
    log.append("user", "secret", "voice")
    chat = CuppyChat(ChatTools(WatchStore(tmp_path)), log)
    chat.reset()
    assert len(chat.build_history("hi")) == 1
