import random
from datetime import timedelta

import pytest

from watcher import clean_link, dedupe_links, format_message, run_watch, search_url
from watches import WatchState, WatchStore, utcnow


@pytest.fixture
def store(tmp_path):
    return WatchStore(tmp_path)


def test_add_list_remove(store):
    watch = store.add("Aussie puppies in Egypt", "australian shepherd puppy egypt", 120)
    assert [w.id for w in store.list()] == [watch.id]
    assert store.remove(watch.id)
    assert store.list() == []
    assert not store.remove("missing")


def test_interval_has_a_floor(store):
    assert store.add("x", "y", 5).interval_minutes == 30


def test_add_requires_query_or_page(store):
    with pytest.raises(ValueError):
        store.add("x", "", 60)
    assert store.add("x", "", 60, ["https://example.com/search"]).urls


def test_due_and_jitter(tmp_path, store):
    watch = store.add("x", "y", 120)
    state = WatchState(tmp_path)
    now = utcnow()
    assert state.is_due(watch, now)
    state.record_run(watch, ["https://a"], now, random.Random(1))
    assert not state.is_due(watch, now + timedelta(minutes=89))
    assert state.is_due(watch, now + timedelta(minutes=151))
    # State survives a restart.
    assert WatchState(tmp_path).seen(watch.id) == {"https://a"}


def test_prune_drops_cancelled_watches(tmp_path, store):
    watch = store.add("x", "y", 60)
    state = WatchState(tmp_path)
    state.record_run(watch, ["https://a"])
    state.prune(set())
    assert WatchState(tmp_path).seen(watch.id) == set()


def test_clean_link_unwraps_duckduckgo():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.dubizzle.com.eg%2Fad%2F123&rut=x"
    assert clean_link(href, search_url("q")) == "https://www.dubizzle.com.eg/ad/123"
    assert clean_link("/html/?q=next", search_url("q")) is None
    assert clean_link("javascript:void(0)", "https://site.com") is None
    assert clean_link("/ad/5#top", "https://site.com/s") == "https://site.com/ad/5"


def test_dedupe_links():
    links = [
        {"url": "https://a", "text": "A"},
        {"url": "https://a", "text": "A again"},
        {"url": "https://b", "text": "  "},
    ]
    assert dedupe_links(links) == [{"url": "https://a", "text": "A"}]


class FakeDiscord:
    def __init__(self):
        self.sent = []

    async def send_message(self, text):
        self.sent.append(text)


async def test_run_watch_sends_only_new_matches(tmp_path, store):
    watch = store.add("Aussie puppies in Egypt", "aussie puppies egypt", 60)
    state = WatchState(tmp_path)
    discord = FakeDiscord()
    pages = {
        search_url(watch.search_query): [
            {
                "url": "https://shop/ad/1",
                "text": "Australian Shepherd puppy, Cairo, 15000 EGP",
            },
            {"url": "https://blog/breed-guide", "text": "All about Aussies"},
        ]
    }
    judged = []

    async def fetch(url):
        return pages[url]

    async def judge(description, candidates):
        judged.append([c["url"] for c in candidates])
        return [
            {
                "url": "https://shop/ad/1",
                "title": "Aussie puppy",
                "price": "15000 EGP",
                "location": "Cairo",
            }
            for c in candidates
            if c["url"] == "https://shop/ad/1"
        ]

    found = await run_watch(watch, state, fetch, judge, discord)
    assert [m["url"] for m in found] == ["https://shop/ad/1"]
    assert len(discord.sent) == 1 and "https://shop/ad/1" in discord.sent[0]

    # Second run: nothing new, so no message and nothing re-judged.
    found = await run_watch(watch, state, fetch, judge, discord)
    assert found == [] and len(discord.sent) == 1
    assert judged[1:] == []

    # A new listing appears.
    pages[search_url(watch.search_query)].append(
        {"url": "https://shop/ad/2", "text": "Another Aussie puppy, Giza"}
    )

    async def judge2(description, candidates):
        return [{"url": c["url"], "title": "New pup"} for c in candidates]

    found = await run_watch(watch, state, fetch, judge2, discord)
    assert [m["url"] for m in found] == ["https://shop/ad/2"]
    assert len(discord.sent) == 2


def test_format_message_caps_results(store):
    watch = store.add("pups", "pups", 60)
    matches = [{"url": f"https://x/{i}", "title": f"Pup {i}"} for i in range(8)]
    text = format_message(watch, matches)
    assert "https://x/4" in text and "https://x/5" not in text
    assert "3 more" in text
