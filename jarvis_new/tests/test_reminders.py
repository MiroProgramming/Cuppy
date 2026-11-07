import json
from datetime import datetime, timedelta, timezone

import pytest

from reminders import (
    ReminderStore,
    deliver_due,
    friendly_time,
    local_now,
    parse_when,
    reminder_text,
    tool_list_reminders,
    tool_set_reminder,
)

CAIRO = timezone(timedelta(hours=2))
NOW = datetime(2026, 11, 7, 10, 0, tzinfo=CAIRO)


def test_parse_when():
    assert parse_when(in_minutes=20, now=NOW) == NOW + timedelta(minutes=20)
    assert parse_when("2026-11-07T15:30", now=NOW) == datetime(
        2026, 11, 7, 15, 30, tzinfo=CAIRO
    )
    with pytest.raises(ValueError, match="already passed"):
        parse_when("2026-11-07T09:00", now=NOW)
    with pytest.raises(ValueError):
        parse_when("half three", now=NOW)
    with pytest.raises(ValueError):
        parse_when(now=NOW)


def test_friendly_time():
    assert friendly_time(NOW + timedelta(hours=5), NOW) == "today at 15:00"
    assert friendly_time(NOW + timedelta(days=1), NOW) == "tomorrow at 10:00"


class FakeDiscord:
    def __init__(self):
        self.sent = []

    async def send_message(self, text):
        self.sent.append(text)


async def test_due_reminder_is_sent_once(tmp_path):
    store = ReminderStore(tmp_path)
    store.add("Feed the dog", local_now() - timedelta(seconds=5))
    store.add("Later thing", local_now() + timedelta(hours=1))
    discord = FakeDiscord()

    assert await deliver_due(store, discord) == 1
    assert "Feed the dog" in discord.sent[0]
    assert "offline" not in discord.sent[0]
    assert await deliver_due(store, discord) == 0
    assert [r.message for r in store.active()] == ["Later thing"]


async def test_late_reminder_says_so_and_stale_is_dropped(tmp_path):
    store = ReminderStore(tmp_path)
    store.add("Missed an hour ago", local_now() - timedelta(hours=1))
    store.add("Missed last week", local_now() - timedelta(days=7))
    discord = FakeDiscord()
    assert await deliver_due(store, discord) == 1
    assert "offline" in discord.sent[0]
    assert store.active() == []


async def test_daily_reminder_rolls_forward(tmp_path):
    store = ReminderStore(tmp_path)
    due = local_now() - timedelta(seconds=1)
    store.add("Walk", due, "daily")
    discord = FakeDiscord()
    assert await deliver_due(store, discord) == 1
    assert "Repeats daily" in discord.sent[0]
    (next_one,) = store.active()
    assert next_one.due == due + timedelta(days=1)


def test_cancel_and_tools(tmp_path):
    store = ReminderStore(tmp_path)
    result = tool_set_reminder(store, "Call mum", in_minutes=30)
    assert result["set"] and result["when"].startswith(("today", "tomorrow"))
    listed = tool_list_reminders(store)
    assert listed["reminders"][0]["message"] == "Call mum"
    assert store.cancel(result["id"])
    assert not store.cancel(result["id"])
    assert tool_list_reminders(store)["reminders"] == []


def test_store_ignores_half_written_lines(tmp_path):
    store = ReminderStore(tmp_path)
    store.add("Real", local_now() + timedelta(hours=1))
    with store.path.open("a") as f:
        f.write('{"type": "add", "id"')
    assert [r.message for r in store.active()] == ["Real"]
    assert json.loads(store.path.read_text().splitlines()[0])["type"] == "add"


def test_reminder_text_on_time():
    store_free = reminder_text
    from reminders import Reminder

    text = store_free(
        Reminder("x", "Stretch", NOW, "none"), NOW + timedelta(seconds=30)
    )
    assert text == "⏰ **Reminder, Sir:** Stretch"
