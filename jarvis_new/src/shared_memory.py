"""One conversation shared by voice Cuppy and Discord Cuppy.

Both programs append what was said to ``data/conversation.jsonl`` and read the
recent part back, so whatever you tell one of them, the other knows.
Saying or sending "reset" starts a fresh conversation for both.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from watches import DEFAULT_DATA_DIR

DEFAULT_LIMIT = 40
DEFAULT_MAX_AGE = timedelta(days=7)
MAX_TEXT_CHARS = 1500
CHANNEL_LABELS = {"voice": "voice call", "discord": "Discord"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationLog:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.path = Path(data_dir) / "conversation.jsonl"

    def append(
        self, role: str, text: str, channel: str, when: datetime | None = None
    ) -> None:
        text = (text or "").strip()
        if role not in ("user", "assistant") or not text:
            return
        self._write(
            {"role": role, "text": text[:MAX_TEXT_CHARS], "channel": channel}, when
        )

    def reset(self, when: datetime | None = None) -> None:
        self._write({"role": "reset"}, when)

    def recent(
        self,
        limit: int = DEFAULT_LIMIT,
        max_age: timedelta = DEFAULT_MAX_AGE,
        now: datetime | None = None,
    ) -> list[dict]:
        cutoff = (now or _now()) - max_age
        entries: list[dict] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written line from the other program
            if entry.get("role") == "reset":
                entries = []
            elif datetime.fromisoformat(entry["at"]) >= cutoff:
                entries.append(entry)
        return entries[-limit:]

    def _write(self, entry: dict, when: datetime | None) -> None:
        entry["at"] = (when or _now()).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _local_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%a %H:%M")


def format_for_prompt(entries: list[dict]) -> str:
    """Render recent history as a block for the voice agent's instructions."""
    if not entries:
        return ""
    lines = [
        "# Recent conversation with the user",
        "",
        "This is what you and the user said recently, over voice calls and Discord "
        "messages. It is the same ongoing conversation, so use it naturally as "
        "context. Do not read it back unless asked.",
        "",
    ]
    for entry in entries:
        who = "User" if entry["role"] == "user" else "Cuppy"
        where = CHANNEL_LABELS.get(entry.get("channel", ""), entry.get("channel", ""))
        lines.append(f"[{_local_time(entry['at'])}, {where}] {who}: {entry['text']}")
    return "\n".join(lines)


def tag_for_model(entry: dict) -> str:
    """Prefix a past message with when and where it happened, for the text chat."""
    where = CHANNEL_LABELS.get(entry.get("channel", ""), entry.get("channel", ""))
    return f"[{_local_time(entry['at'])}, {where}] {entry['text']}"
