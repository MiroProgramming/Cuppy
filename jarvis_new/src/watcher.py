"""Background watcher: re-runs Cuppy's saved watches and DMs new matches on Discord.

Run it alongside the agent and leave it running:

    uv run src/watcher.py

It needs GOOGLE_API_KEY, DISCORD_BOT_TOKEN and DISCORD_USER_ID in .env.local.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

from dotenv import load_dotenv

from discord_tools import DiscordClient, DiscordError
from watches import Watch, WatchState, WatchStore

logger = logging.getLogger("cuppy-watcher")

POLL_SECONDS = 60
MAX_CANDIDATES = 150
MAX_RESULTS_PER_MESSAGE = 5
DEFAULT_MODEL = "gemini-2.5-flash"

Link = dict[str, str]
Fetcher = Callable[[str], Awaitable[list[Link]]]
Judge = Callable[[str, list[Link]], Awaitable[list[dict[str, str]]]]


def search_url(query: str) -> str:
    return f"https://html.duckduckgo.com/html/?{urlencode({'q': query})}"


def clean_link(href: str, base: str) -> str | None:
    """Make a link absolute and unwrap DuckDuckGo redirect links."""
    url = urljoin(base, href.strip())
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target is None:
            return None
        url, parsed = target, urlparse(target)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc.endswith("duckduckgo.com"):
        return None
    return url.split("#", 1)[0]


def dedupe_links(links: list[Link]) -> list[Link]:
    seen: set[str] = set()
    unique: list[Link] = []
    for link in links:
        if link["url"] in seen or not link["text"].strip():
            continue
        seen.add(link["url"])
        unique.append(link)
    return unique[:MAX_CANDIDATES]


class PlaywrightFetcher:
    """Opens pages in a headless browser and returns the links on them."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None

    async def __aenter__(self) -> PlaywrightFetcher:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def __call__(self, url: str) -> list[Link]:
        page = await self._browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(2_000)  # let listing pages render
            raw = await page.eval_on_selector_all(
                "a[href]",
                """anchors => anchors.map(a => {
                    const box = a.closest('article, li, [class*="listing"], [class*="card"], [class*="result"]') || a;
                    return {
                        href: a.getAttribute('href') || '',
                        text: (box.innerText || a.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 300),
                    };
                })""",
            )
        finally:
            await page.close()

        links: list[Link] = []
        for item in raw:
            cleaned = clean_link(item["href"], url)
            if cleaned:
                links.append({"url": cleaned, "text": item["text"]})
        return links


class GeminiJudge:
    """Asks Gemini which of the collected links are real matches."""

    def __init__(self, model: str | None = None) -> None:
        from google import genai

        self._client = genai.Client()
        self._model = model or os.getenv("WATCHER_MODEL", DEFAULT_MODEL)

    async def __call__(
        self, description: str, candidates: list[Link]
    ) -> list[dict[str, str]]:
        from google.genai import types

        prompt = (
            "You are filtering web links for a user who wants to be told about: "
            f"{description}\n\n"
            "Below are links collected from search results and listing pages, each "
            "with nearby text. Return ONLY links that are a single specific listing, "
            "post, or ad that clearly matches the request. Exclude category pages, "
            "search pages, articles, guides, breed information, and anything that "
            "does not match the location or other details. If nothing matches, "
            "return an empty list.\n\n" + json.dumps(candidates, ensure_ascii=False)
        )
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "price": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["url", "title"],
            },
        }
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            ),
        )
        try:
            matches = json.loads(response.text or "[]")
        except json.JSONDecodeError:
            logger.warning("Gemini returned invalid JSON: %s", response.text)
            return []
        allowed = {c["url"] for c in candidates}
        # Never trust a URL the model made up.
        return [m for m in matches if m.get("url") in allowed]


def format_message(watch: Watch, matches: list[dict[str, str]]) -> str:
    lines = [f"Found something new for: **{watch.description}**", ""]
    for match in matches[:MAX_RESULTS_PER_MESSAGE]:
        details = " · ".join(
            part for part in (match.get("price"), match.get("location")) if part
        )
        lines.append(
            f"• {match.get('title', 'Listing')}" + (f" ({details})" if details else "")
        )
        lines.append(f"  {match['url']}")
    extra = len(matches) - MAX_RESULTS_PER_MESSAGE
    if extra > 0:
        lines.append(f"…and {extra} more.")
    return "\n".join(lines)


async def run_watch(
    watch: Watch,
    state: WatchState,
    fetch: Fetcher,
    judge: Judge,
    discord: Any,
) -> list[dict[str, str]]:
    """Check one watch once. Returns the new matches that were sent."""
    pages = [*watch.urls]
    if watch.search_query:
        pages.append(search_url(watch.search_query))

    links: list[Link] = []
    for url in pages:
        try:
            links.extend(await fetch(url))
        except Exception:
            logger.exception("Could not load %s", url)

    seen = state.seen(watch.id)
    candidates = [link for link in dedupe_links(links) if link["url"] not in seen]
    matches = await judge(watch.description, candidates) if candidates else []

    if matches:
        await discord.send_message(format_message(watch, matches))
    # Record every candidate as seen, so non-matches are not re-judged each time.
    state.record_run(watch, [c["url"] for c in candidates])
    return matches


async def main() -> None:
    load_dotenv(".env.local")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    discord = DiscordClient.from_env()
    if discord is None:
        raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_USER_ID in .env.local")
    store = WatchStore()
    judge = GeminiJudge()
    logger.info("Cuppy watcher started. Leave this window open.")

    try:
        while True:
            watches = store.list()
            state = WatchState()
            state.prune({w.id for w in watches})
            due = [w for w in watches if state.is_due(w)]
            if due:
                async with PlaywrightFetcher() as fetch:
                    for watch in due:
                        try:
                            found = await run_watch(watch, state, fetch, judge, discord)
                            logger.info("%s: %d new", watch.description, len(found))
                        except DiscordError as exc:
                            logger.error("Discord error: %s", exc)
                            state.record_run(watch, [])  # retry next interval
                        except Exception:
                            logger.exception("Watch %s failed", watch.id)
                            state.record_run(watch, [])
            await asyncio.sleep(POLL_SECONDS)
    finally:
        await discord.close()


if __name__ == "__main__":
    asyncio.run(main())
