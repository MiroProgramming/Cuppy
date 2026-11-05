import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from discord_tools import DiscordClient, DiscordError, split_message

USER_ID = "111"
BOT_ID = "999"


@pytest.fixture
async def fake_discord():
    state = {"sent": [], "files": [], "auth": []}

    async def create_dm(request):
        state["auth"].append(request.headers.get("Authorization"))
        body = await request.json()
        assert body == {"recipient_id": USER_ID}
        return web.json_response({"id": "dm-1"})

    async def post_message(request):
        if request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            async for part in reader:
                if part.name == "payload_json":
                    state["sent"].append((await part.json())["content"])
                else:
                    state["files"].append(await part.read())
        else:
            state["sent"].append((await request.json())["content"])
        return web.json_response({"id": str(len(state["sent"]))})

    async def get_messages(request):
        return web.json_response(
            [
                {"author": {"id": USER_ID}, "content": "newest", "timestamp": "t3"},
                {"author": {"id": BOT_ID}, "content": "bot reply", "timestamp": "t2"},
                {"author": {"id": USER_ID}, "content": "older", "timestamp": "t1"},
            ]
        )

    async def forbidden(request):
        return web.Response(status=403, text="Cannot send messages to this user")

    app = web.Application()
    app.router.add_post("/users/@me/channels", create_dm)
    app.router.add_post("/channels/dm-1/messages", post_message)
    app.router.add_get("/channels/dm-1/messages", get_messages)
    app.router.add_post("/channels/blocked/messages", forbidden)

    server = TestServer(app)
    await server.start_server()
    yield server, state
    await server.close()


def make_client(server) -> DiscordClient:
    return DiscordClient("token-abc", USER_ID, api_base=str(server.make_url("")))


def test_split_message_respects_limit():
    text = "word " * 1000
    chunks = split_message(text, limit=2000)
    assert len(chunks) == 3
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_message_rejects_empty():
    with pytest.raises(ValueError):
        split_message("   ")


async def test_send_message_opens_dm_and_sends(fake_discord):
    server, state = fake_discord
    client = make_client(server)
    try:
        await client.send_message("Hello, Sir")
        await client.send_message("Second")
    finally:
        await client.close()

    assert state["sent"] == ["Hello, Sir", "Second"]
    assert state["auth"] == ["Bot token-abc"]  # DM channel created once, then cached


async def test_send_long_message_is_chunked(fake_discord):
    server, state = fake_discord
    client = make_client(server)
    try:
        ids = await client.send_message("x " * 1500)
    finally:
        await client.close()
    assert len(ids) == 2
    assert all(len(text) <= 2000 for text in state["sent"])


async def test_send_with_screenshot_attaches_file(fake_discord):
    server, state = fake_discord
    client = make_client(server)
    try:
        await client.send_message("Here you go", file_bytes=b"\x89PNGfake")
    finally:
        await client.close()
    assert state["sent"] == ["Here you go"]
    assert state["files"] == [b"\x89PNGfake"]


async def test_read_recent_messages_only_returns_users_messages(fake_discord):
    server, _ = fake_discord
    client = make_client(server)
    try:
        messages = await client.read_recent_messages(5)
    finally:
        await client.close()
    assert [m["text"] for m in messages] == ["older", "newest"]


async def test_forbidden_gives_friendly_error(fake_discord):
    server, _ = fake_discord
    client = make_client(server)
    client._dm_channel_id = "blocked"
    try:
        with pytest.raises(DiscordError, match="allow direct messages"):
            await client.send_message("hi")
    finally:
        await client.close()


def test_from_env_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_USER_ID", raising=False)
    assert DiscordClient.from_env() is None
