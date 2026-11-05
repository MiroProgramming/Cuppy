"""Discord messaging tools so Jarvis can DM its owner.

Uses the Discord REST API directly through aiohttp (already installed with
livekit-agents), so no extra dependency or gateway connection is needed.

Required environment variables:
    DISCORD_BOT_TOKEN   Bot token from the Discord Developer Portal.
    DISCORD_USER_ID     Your own Discord user ID (enable Developer Mode,
                        right-click your name, "Copy User ID").
"""

from __future__ import annotations

import json
import logging
import os

import aiohttp
from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from browser import BrowserError, BrowserManager

logger = logging.getLogger("discord-tools")

DISCORD_API = "https://discord.com/api/v10"
DISCORD_MAX_CHARS = 2000


class DiscordError(Exception):
    """Raised when a Discord API call fails."""


def split_message(text: str, limit: int = DISCORD_MAX_CHARS) -> list[str]:
    """Split text into chunks that fit Discord's message length limit."""
    text = text.strip()
    if not text:
        raise ValueError("The message cannot be empty.")

    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


class DiscordClient:
    """Minimal async client that sends and reads DMs between the bot and one user."""

    def __init__(
        self,
        token: str,
        user_id: str,
        *,
        api_base: str = DISCORD_API,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._token = token
        self._user_id = user_id
        self._api_base = api_base.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._dm_channel_id: str | None = None

    @classmethod
    def from_env(cls) -> DiscordClient | None:
        token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
        user_id = os.getenv("DISCORD_USER_ID", "").strip()
        if not token or not user_id:
            logger.info(
                "Discord tools disabled: DISCORD_BOT_TOKEN or DISCORD_USER_ID not set"
            )
            return None
        return cls(token, user_id)

    @property
    def user_id(self) -> str:
        return self._user_id

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def send_message(
        self,
        content: str,
        *,
        file_bytes: bytes | None = None,
        filename: str = "screenshot.png",
    ) -> list[str]:
        """Send a DM to the user. Long text is split into several messages.

        Returns the IDs of the messages that were sent.
        """
        channel_id = await self._get_dm_channel()
        chunks = split_message(content)
        sent: list[str] = []

        for index, chunk in enumerate(chunks):
            is_last = index == len(chunks) - 1
            if file_bytes is not None and is_last:
                form = aiohttp.FormData()
                form.add_field(
                    "payload_json",
                    json.dumps({"content": chunk}),
                    content_type="application/json",
                )
                form.add_field(
                    "files[0]", file_bytes, filename=filename, content_type="image/png"
                )
                data = await self._request(
                    "POST", f"/channels/{channel_id}/messages", data=form
                )
            else:
                data = await self._request(
                    "POST", f"/channels/{channel_id}/messages", json={"content": chunk}
                )
            sent.append(str(data["id"]))
        return sent

    async def read_recent_messages(self, limit: int = 5) -> list[dict[str, str]]:
        """Return the user's most recent DMs to the bot, oldest first."""
        limit = max(1, min(limit, 20))
        channel_id = await self._get_dm_channel()
        # Fetch extra so the bot's own messages don't crowd out the user's.
        data = await self._request(
            "GET", f"/channels/{channel_id}/messages", params={"limit": str(limit * 3)}
        )
        from_user = [
            {
                "text": message.get("content", ""),
                "sent_at": message.get("timestamp", ""),
            }
            for message in data
            if str(message.get("author", {}).get("id")) == self._user_id
        ]
        return list(reversed(from_user[:limit]))

    async def _get_dm_channel(self) -> str:
        if self._dm_channel_id is None:
            data = await self._request(
                "POST", "/users/@me/channels", json={"recipient_id": self._user_id}
            )
            self._dm_channel_id = str(data["id"])
        return self._dm_channel_id

    async def _request(self, method: str, path: str, **kwargs):
        if self._session is None:
            self._session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bot {self._token}",
            "User-Agent": "DiscordBot (jarvis-voice-butler, 1.0)",
        }
        try:
            async with self._session.request(
                method,
                f"{self._api_base}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
                **kwargs,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning(
                        "Discord API %s %s failed: %s %s",
                        method,
                        path,
                        response.status,
                        body,
                    )
                    raise DiscordError(_friendly_error(response.status))
                return await response.json()
        except aiohttp.ClientError as exc:
            raise DiscordError("I could not reach Discord.") from exc


def _friendly_error(status: int) -> str:
    if status == 401:
        return "Discord rejected the bot token."
    if status == 403:
        return (
            "Discord would not let me message you. Make sure you share a server "
            "with the bot and allow direct messages from server members."
        )
    if status == 429:
        return "Discord is rate limiting me. Try again in a moment."
    return f"Discord returned an error ({status})."


class DiscordTools:
    """LiveKit function tools that let Jarvis message the user on Discord."""

    def __init__(
        self, client: DiscordClient, browser: BrowserManager | None = None
    ) -> None:
        self.client = client
        self.browser = browser

    @property
    def tools(self) -> list:
        tools = [self.send_discord_message, self.read_discord_messages]
        if self.browser is not None:
            tools.append(self.send_discord_screenshot)
        return tools

    @function_tool()
    async def send_discord_message(self, context: RunContext, message: str) -> str:
        """Send the user a direct message on Discord.

        Use this when the user asks you to message, text, ping, or send them something
        on Discord, or to send them a link, summary, list, or result so they can read
        it later. Write the message in clear, readable text; markdown and full links
        are fine here, unlike in speech.

        Args:
            message: The full text to send to the user.
        """
        try:
            sent = await self.client.send_message(message)
        except (DiscordError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return f"Sent to the user on Discord in {len(sent)} message(s)."

    @function_tool()
    async def send_discord_screenshot(self, context: RunContext, caption: str) -> str:
        """Send the user a screenshot of the current browser page on Discord.

        Use this when the user asks you to send them what is on screen, a picture of
        the page, or a screenshot.

        Args:
            caption: A short description to send alongside the screenshot.
        """
        if self.browser is None:
            raise ToolError("No browser is available to capture.")
        try:
            image, url = await self.browser.screenshot_png()
            text = f"{caption.strip() or 'Screenshot'}\n{url}"
            await self.client.send_message(text, file_bytes=image)
        except (BrowserError, DiscordError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return "Screenshot sent to the user on Discord."

    @function_tool()
    async def read_discord_messages(self, context: RunContext, limit: int = 5) -> dict:
        """Read the most recent messages the user sent you on Discord.

        Use this when the user asks you to check, read, or act on what they sent you
        on Discord.

        Args:
            limit: How many recent messages to read, from 1 to 20.
        """
        try:
            messages = await self.client.read_recent_messages(limit)
        except DiscordError as exc:
            raise ToolError(str(exc)) from exc
        if not messages:
            return {
                "messages": [],
                "note": "The user has not sent any Discord messages.",
            }
        return {"messages": messages}
