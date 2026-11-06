from datetime import timedelta

from livekit.agents.llm import ChatMessage

from shared_memory import ConversationLog, _now, format_for_prompt


def test_append_and_recent(tmp_path):
    log = ConversationLog(tmp_path)
    log.append("user", "Find me a puppy", "voice")
    log.append("assistant", "On it, Sir", "voice")
    log.append("system", "ignored", "voice")
    log.append("user", "   ", "discord")
    assert [(e["role"], e["text"]) for e in log.recent()] == [
        ("user", "Find me a puppy"),
        ("assistant", "On it, Sir"),
    ]


def test_reset_hides_earlier_messages(tmp_path):
    log = ConversationLog(tmp_path)
    log.append("user", "old", "voice")
    log.reset()
    log.append("user", "new", "discord")
    assert [e["text"] for e in log.recent()] == ["new"]


def test_old_messages_and_limit(tmp_path):
    log = ConversationLog(tmp_path)
    log.append("user", "ancient", "voice", when=_now() - timedelta(days=30))
    for i in range(5):
        log.append("user", f"m{i}", "discord")
    assert [e["text"] for e in log.recent(limit=3)] == ["m2", "m3", "m4"]


def test_half_written_line_is_skipped(tmp_path):
    log = ConversationLog(tmp_path)
    log.append("user", "hello", "voice")
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"role": "user", "te')
    assert [e["text"] for e in log.recent()] == ["hello"]


def test_format_for_prompt(tmp_path):
    log = ConversationLog(tmp_path)
    assert format_for_prompt([]) == ""
    log.append("user", "My dog is called Biscuit", "discord")
    text = format_for_prompt(log.recent())
    assert "Discord] User: My dog is called Biscuit" in text


def test_voice_agent_saves_and_loads_shared_history(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    from agent import Assistant, save_chat_item

    log = ConversationLog(tmp_path)
    save_chat_item(log, ChatMessage(role="user", content=["Remember Biscuit"]))
    save_chat_item(log, ChatMessage(role="assistant", content=["Noted, Sir"]))
    save_chat_item(log, ChatMessage(role="system", content=["hidden"]))
    assert [e["channel"] for e in log.recent()] == ["voice", "voice"]

    assistant = Assistant(conversation=log)
    assert "voice call] User: Remember Biscuit" in assistant.instructions
    assert "Recent conversation" not in Assistant().instructions
