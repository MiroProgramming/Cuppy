"""Reminders that Cuppy delivers as Discord messages at the right time.

Voice Cuppy and Discord Cuppy both create and cancel reminders by appending
lines to ``data/reminders.jsonl`` (appending is safe with two programs). Only
the Discord bot delivers them, and it alone records what it has sent in
``data/reminders_sent.json``. Reminders are therefore delivered while
``discord_bot.py`` is running; any missed while it was off are sent late,
with a note, when it starts again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from watches import DEFAULT_DATA_DIR

logger = logging.getLogger("cuppy-reminders")

CHECK_SECONDS = 15
LATE_GRACE = timedelta(minutes=2)
MAX_LATE = timedelta(days=1)
REPEATS = {"none": None, "daily": timedelta(days=1), "weekly": timedelta(weeks=1)}


def local_now() -> datetime:
    return datetime.now().astimezone()


def describe_now() -> str:
    now = local_now()
    return now.strftime("%A %d %B %Y, %H:%M") + f" (UTC{now.strftime('%z')[:3]})"


def parse_when(
    at: str | None = None,
    in_minutes: float | None = None,
    now: datetime | None = None,
) -> datetime:
    """Turn a local date/time string or a delay into an aware datetime."""
    now = now or local_now()
    if in_minutes:
        if in_minutes <= 0:
            raise ValueError("The delay must be in the future.")
        return now + timedelta(minutes=in_minutes)
    if not at:
        raise ValueError("Say when: a time, or how many minutes from now.")
    try:
        when = datetime.fromisoformat(at.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Use a date and time like 2026-11-07T15:30, or minutes from now."
        ) from exc
    if when.tzinfo is None:
        when = when.replace(tzinfo=now.tzinfo)
    if when <= now:
        raise ValueError("That time has already passed.")
    return when


def friendly_time(when: datetime, now: datetime | None = None) -> str:
    now = now or local_now()
    when = when.astimezone(now.tzinfo)
    if when.date() == now.date():
        day = "today"
    elif when.date() == (now + timedelta(days=1)).date():
        day = "tomorrow"
    else:
        day = when.strftime("%A %d %B")
    return f"{day} at {when.strftime('%H:%M')}"


@dataclass
class Reminder:
    id: str
    message: str
    due: datetime
    repeat: str = "none"

    def occurrence_key(self) -> str:
        return f"{self.id}@{self.due.astimezone(timezone.utc).isoformat()}"


class ReminderStore:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.path = Path(data_dir) / "reminders.jsonl"
        self.sent_path = Path(data_dir) / "reminders_sent.json"

    # Writing (both programs)

    def add(self, message: str, due: datetime, repeat: str = "none") -> Reminder:
        message = message.strip()
        if not message:
            raise ValueError("What should I remind you about?")
        repeat = (repeat or "none").lower()
        if repeat not in REPEATS:
            raise ValueError("Repeat must be none, daily, or weekly.")
        reminder = Reminder(uuid.uuid4().hex[:8], message, due, repeat)
        self._append(
            {
                "type": "add",
                "id": reminder.id,
                "message": message,
                "due": due.isoformat(),
                "repeat": repeat,
            }
        )
        return reminder

    def cancel(self, reminder_id: str) -> bool:
        if reminder_id not in {r.id for r in self.active()}:
            return False
        self._append({"type": "cancel", "id": reminder_id})
        return True

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Reading

    def _all(self) -> list[Reminder]:
        reminders: dict[str, Reminder] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "add":
                reminders[entry["id"]] = Reminder(
                    entry["id"],
                    entry["message"],
                    datetime.fromisoformat(entry["due"]),
                    entry.get("repeat", "none"),
                )
            elif entry.get("type") == "cancel":
                reminders.pop(entry.get("id"), None)
        return list(reminders.values())

    def _sent(self) -> dict[str, str]:
        try:
            return json.loads(self.sent_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _next_occurrence(
        self, reminder: Reminder, sent: dict[str, str]
    ) -> Reminder | None:
        """The next unsent occurrence, rolling repeating reminders forward."""
        step = REPEATS.get(reminder.repeat)
        current = reminder
        while current.occurrence_key() in sent:
            if step is None:
                return None
            current = Reminder(
                current.id, current.message, current.due + step, current.repeat
            )
        return current

    def active(self) -> list[Reminder]:
        sent = self._sent()
        upcoming = [self._next_occurrence(r, sent) for r in self._all()]
        return sorted((r for r in upcoming if r), key=lambda r: r.due)

    # Delivering (Discord bot only)

    def due(self, now: datetime | None = None) -> list[Reminder]:
        now = now or local_now()
        return [r for r in self.active() if r.due <= now]

    def mark_sent(self, reminder: Reminder, now: datetime | None = None) -> None:
        sent = self._sent()
        sent[reminder.occurrence_key()] = (now or local_now()).isoformat()
        self.sent_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.sent_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sent, indent=2), encoding="utf-8")
        os.replace(tmp, self.sent_path)


def reminder_text(reminder: Reminder, now: datetime | None = None) -> str:
    now = now or local_now()
    text = f"⏰ **Reminder, Sir:** {reminder.message}"
    late = now - reminder.due
    if late > LATE_GRACE:
        text += f"\n(This was due {friendly_time(reminder.due, now)}; I was offline.)"
    if reminder.repeat != "none":
        text += f"\n(Repeats {reminder.repeat}.)"
    return text


async def deliver_due(store: ReminderStore, discord: Any, log: Any = None) -> int:
    """Send every due reminder once. Returns how many were sent."""
    now = local_now()
    count = 0
    for reminder in store.due(now):
        if now - reminder.due > MAX_LATE and reminder.repeat == "none":
            store.mark_sent(reminder, now)  # too stale to be useful
            continue
        text = reminder_text(reminder, now)
        await discord.send_message(text)
        store.mark_sent(reminder, now)
        if log is not None:
            log.append("assistant", f"(Sent a reminder) {reminder.message}", "discord")
        count += 1
    return count


async def reminder_loop(
    discord: Any, store: ReminderStore | None = None, log: Any = None
):
    store = store or ReminderStore()
    while True:
        try:
            await deliver_due(store, discord, log)
        except Exception:
            logger.exception("Could not deliver reminders; will retry")
        await asyncio.sleep(CHECK_SECONDS)


# Shared tool logic, used by both voice and Discord Cuppy


def tool_set_reminder(
    store: ReminderStore,
    message: str,
    at: str | None = None,
    in_minutes: float | None = None,
    repeat: str = "none",
) -> dict:
    due = parse_when(at, in_minutes)
    reminder = store.add(message, due, repeat)
    return {
        "set": True,
        "id": reminder.id,
        "when": friendly_time(reminder.due),
        "repeat": reminder.repeat,
        "note": "It will arrive as a Discord message while the Discord bot is running.",
    }


def tool_list_reminders(store: ReminderStore) -> dict:
    return {
        "now": describe_now(),
        "reminders": [
            {
                "id": r.id,
                "message": r.message,
                "when": friendly_time(r.due),
                "repeat": r.repeat,
            }
            for r in store.active()
        ],
    }


def tool_cancel_reminder(store: ReminderStore, reminder_id: str) -> dict:
    return {"cancelled": store.cancel(reminder_id)}
