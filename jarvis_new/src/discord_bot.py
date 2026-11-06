"""Cuppy on Discord: reads your DMs and replies over text, any time.

Run it and leave it running:

    uv run src/discord_bot.py

It also runs your background watches, so you don't need watcher.py as well.
Only messages from DISCORD_USER_ID are answered; everyone else is ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from typing import Any

import aiohttp
from dotenv import load_dotenv

from discord_tools import DiscordClient, split_message
from shared_memory import ConversationLog, tag_for_model
from watcher import clean_link, search_url, watch_loop
from watches import WatchStore

logger = logging.getLogger("cuppy-discord")

DEFAULT_MODEL = "gemini-flash-latest"
MAX_TOOL_ROUNDS = 6
MAX_PAGE_CHARS = 8000
RESET_WORDS = {"reset", "!reset", "new chat", "/reset"}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )
}

TEXT_INSTRUCTIONS = """\
You are Cuppy, a helpful and sarcastic British AI butler, chatting with your owner
over Discord direct messages. Address him as "Sir" where it fits.

- Reply like a text message: short and to the point, a few sentences at most unless
  he asks for detail. Simple markdown, bullet lists, and full links are fine.
- If he asks "Cuppy, you there?", reply exactly: "At your service, Sir".
- Use web_search for anything current or that you are not sure about, and
  read_webpage to look inside a result before answering. Include links to what
  you found. Never invent links, prices, or facts.
- Use create_watch when he wants you to keep checking for something and tell him
  when it appears. If he did not say how often, check every two hours. Use
  list_watches and cancel_watch to manage them.
- If a tool fails, say so briefly and suggest what to try instead.
- You also talk with him by voice. Earlier messages are tagged with when and where
  they happened (voice call or Discord); treat it all as one conversation and
  continue naturally. Do not add tags to your own replies.
"""

ToolFn = Callable[..., Awaitable[Any]]


class _TextExtractor(HTMLParser):
    SKIP = frozenset({"script", "style", "noscript", "svg", "head"})

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(page: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(page)
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return parser.title.strip(), text


_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?:.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
    re.S,
)


def parse_search_results(page: str, limit: int = 8) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for match in _RESULT_RE.finditer(page):
        url = clean_link(html.unescape(match["href"]), search_url(""))
        if not url:
            continue
        strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()  # noqa: E731
        results.append(
            {
                "title": strip(match["title"]),
                "url": url,
                "snippet": strip(match["snippet"]),
            }
        )
        if len(results) >= limit:
            break
    return results


class ChatTools:
    """Tools Cuppy can use while chatting over text."""

    def __init__(self, store: WatchStore, session: aiohttp.ClientSession | None = None):
        self.store = store
        self._session = session

    async def _get(self, url: str) -> str:
        if self._session is None:
            self._session = aiohttp.ClientSession(headers=BROWSER_HEADERS)
        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=20), allow_redirects=True
        ) as response:
            response.raise_for_status()
            return await response.text(errors="replace")

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def web_search(self, query: str) -> dict:
        page = await self._get(search_url(query))
        results = parse_search_results(page)
        return (
            {"results": results} if results else {"results": [], "note": "No results."}
        )

    async def read_webpage(self, url: str) -> dict:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        title, text = html_to_text(await self._get(url))
        return {"url": url, "title": title, "text": text[:MAX_PAGE_CHARS]}

    async def create_watch(
        self, description: str, search_query: str, interval_hours: float = 2
    ) -> dict:
        watch = self.store.add(description, search_query, round(interval_hours * 60))
        return {
            "created": True,
            "id": watch.id,
            "every_minutes": watch.interval_minutes,
        }

    async def list_watches(self) -> dict:
        return {
            "watches": [
                {
                    "id": w.id,
                    "description": w.description,
                    "every_minutes": w.interval_minutes,
                }
                for w in self.store.list()
            ]
        }

    async def cancel_watch(self, watch_id: str) -> dict:
        return {"cancelled": self.store.remove(watch_id)}

    def registry(self) -> dict[str, ToolFn]:
        return {
            "web_search": self.web_search,
            "read_webpage": self.read_webpage,
            "create_watch": self.create_watch,
            "list_watches": self.list_watches,
            "cancel_watch": self.cancel_watch,
        }


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOL_DECLARATIONS = [
    {
        "name": "web_search",
        "description": "Search the web. Returns titles, links, and snippets.",
        "parameters_json_schema": _schema(
            {"query": {"type": "string", "description": "What to search for."}},
            ["query"],
        ),
    },
    {
        "name": "read_webpage",
        "description": "Read the text of a web page.",
        "parameters_json_schema": _schema(
            {"url": {"type": "string", "description": "Full URL of the page."}}, ["url"]
        ),
    },
    {
        "name": "create_watch",
        "description": (
            "Keep checking for something in the background and message the user "
            "on Discord when new matches appear."
        ),
        "parameters_json_schema": _schema(
            {
                "description": {
                    "type": "string",
                    "description": "What counts as a match, including location and details.",
                },
                "search_query": {
                    "type": "string",
                    "description": "Web search query that finds fresh results.",
                },
                "interval_hours": {
                    "type": "number",
                    "description": "How often to check, in hours. Default 2.",
                },
            },
            ["description", "search_query"],
        ),
    },
    {
        "name": "list_watches",
        "description": "List the background watches that are running.",
        "parameters_json_schema": _schema({}, []),
    },
    {
        "name": "cancel_watch",
        "description": "Stop a background watch by its id. Call list_watches first.",
        "parameters_json_schema": _schema(
            {"watch_id": {"type": "string", "description": "The watch id."}},
            ["watch_id"],
        ),
    },
]


async def call_tool(registry: dict[str, ToolFn], name: str, args: dict | None) -> dict:
    fn = registry.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}."}
    try:
        result = await fn(**(args or {}))
        return result if isinstance(result, dict) else {"result": result}
    except Exception as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return {"error": str(exc) or exc.__class__.__name__}


class CuppyChat:
    """Gemini text chat with tool use, sharing its history with voice Cuppy."""

    def __init__(
        self,
        tools: ChatTools,
        log: ConversationLog | None = None,
        model: str | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client()
        self._model = model or os.getenv("CHAT_MODEL", DEFAULT_MODEL)
        self._registry = tools.registry()
        self.log = log or ConversationLog()
        self._config = types.GenerateContentConfig(
            system_instruction=TEXT_INSTRUCTIONS,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    def reset(self) -> None:
        self.log.reset()

    def build_history(self, text: str) -> list:
        """Shared recent history (voice and Discord) plus the new message."""
        types = self._types
        turns: list[tuple[str, str]] = [
            ("user" if e["role"] == "user" else "model", tag_for_model(e))
            for e in self.log.recent()
        ]
        turns.append(("user", text))
        while turns and turns[0][0] != "user":
            turns.pop(0)  # the conversation must start with the user
        contents: list = []
        for role, message in turns:
            if contents and contents[-1].role == role:
                contents[-1].parts.append(types.Part.from_text(text=message))
            else:
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=message)])
                )
        return contents

    async def _generate(self, contents: list):
        return await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=self._config
        )

    async def reply(self, text: str) -> str:
        contents = self.build_history(text)
        response = await self._generate(contents)
        for _ in range(MAX_TOOL_ROUNDS):
            calls = response.function_calls or []
            if not calls:
                break
            contents.append(response.candidates[0].content)
            parts = []
            for call in calls:
                logger.info(
                    "Tool call: %s %s", call.name, json.dumps(call.args or {})[:200]
                )
                result = await call_tool(self._registry, call.name, call.args)
                parts.append(
                    self._types.Part.from_function_response(
                        name=call.name, response=result
                    )
                )
            contents.append(self._types.Content(role="user", parts=parts))
            response = await self._generate(contents)
        answer = (response.text or "").strip()
        answer = answer or "My apologies, Sir, I seem to be lost for words."
        self.log.append("user", text, "discord")
        self.log.append("assistant", answer, "discord")
        return answer


def is_owner_dm(message: Any, owner_id: int, bot_user: Any) -> bool:
    """Only answer direct messages from the owner."""
    import discord

    return (
        message.author.id == owner_id
        and message.author != bot_user
        and isinstance(message.channel, discord.DMChannel)
    )


def build_client(
    owner_id: int, chat: CuppyChat, notify: DiscordClient, run_watches: bool
):
    import discord

    intents = discord.Intents.default()
    intents.dm_messages = True
    intents.message_content = True  # not needed for DMs, harmless if enabled

    class CuppyBot(discord.Client):
        async def setup_hook(self) -> None:
            if run_watches:
                self.loop.create_task(watch_loop(notify))

        async def on_ready(self) -> None:
            logger.info(
                "Cuppy is online on Discord as %s. Leave this window open.", self.user
            )

        async def on_message(self, message: discord.Message) -> None:
            if not is_owner_dm(message, owner_id, self.user):
                return
            text = message.content.strip()
            if not text:
                return
            if text.lower() in RESET_WORDS:
                chat.reset()
                await message.channel.send(
                    "Fresh start, Sir. I have forgotten our conversation, "
                    "on Discord and by voice."
                )
                return
            async with message.channel.typing():
                try:
                    answer = await chat.reply(text)
                except Exception:
                    logger.exception("Chat failed")
                    answer = "Apologies, Sir, something went wrong on my end. Try again shortly."
            for chunk in split_message(answer):
                await message.channel.send(chunk)

    return CuppyBot(intents=intents)


async def main() -> None:
    load_dotenv(".env.local")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    owner = os.getenv("DISCORD_USER_ID", "").strip()
    if not token or not owner.isdigit():
        raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_USER_ID in .env.local")
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY in .env.local")

    tools = ChatTools(WatchStore())
    chat = CuppyChat(tools)
    notify = DiscordClient(token, owner)
    run_watches = os.getenv("BOT_RUNS_WATCHES", "1") != "0"
    client = build_client(int(owner), chat, notify, run_watches)
    try:
        await client.start(token)
    finally:
        await tools.close()
        await notify.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
