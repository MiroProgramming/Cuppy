"""Saved "watches": recurring searches Cuppy keeps running after a call ends.

The voice agent creates and cancels watches (writes ``watches.json``).
The background watcher (``watcher.py``) only reads that file and keeps its own
progress, such as listings already sent, in ``watch_state.json``. Keeping the
writers separate means the two processes never overwrite each other's changes.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIN_INTERVAL_MINUTES = 30
MAX_SEEN_PER_WATCH = 2000


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Watch:
    id: str
    description: str
    search_query: str
    interval_minutes: int
    urls: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utcnow().isoformat())

    @classmethod
    def from_dict(cls, data: dict) -> Watch:
        return cls(
            id=data["id"],
            description=data["description"],
            search_query=data.get("search_query", ""),
            interval_minutes=int(data.get("interval_minutes", 120)),
            urls=list(data.get("urls", [])),
            created_at=data.get("created_at", utcnow().isoformat()),
        )


class WatchStore:
    """The list of active watches. Written by the voice agent only."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.path = Path(data_dir) / "watches.json"

    def list(self) -> list[Watch]:
        return [Watch.from_dict(item) for item in _read_json(self.path, [])]

    def add(
        self,
        description: str,
        search_query: str,
        interval_minutes: int,
        urls: list[str] | None = None,
    ) -> Watch:
        description = description.strip()
        search_query = search_query.strip()
        if not description:
            raise ValueError("Describe what to look for.")
        if not search_query and not urls:
            raise ValueError("A search query or a page to check is required.")
        watch = Watch(
            id=uuid.uuid4().hex[:8],
            description=description,
            search_query=search_query,
            interval_minutes=max(MIN_INTERVAL_MINUTES, int(interval_minutes)),
            urls=[u for u in (urls or []) if u.startswith(("http://", "https://"))],
        )
        watches = self.list()
        watches.append(watch)
        _write_json(self.path, [asdict(w) for w in watches])
        return watch

    def remove(self, watch_id: str) -> bool:
        watches = self.list()
        kept = [w for w in watches if w.id != watch_id]
        if len(kept) == len(watches):
            return False
        _write_json(self.path, [asdict(w) for w in kept])
        return True


class WatchState:
    """Per-watch progress. Written by the background watcher only."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.path = Path(data_dir) / "watch_state.json"
        self._data: dict[str, dict] = _read_json(self.path, {})

    def is_due(self, watch: Watch, now: datetime | None = None) -> bool:
        now = now or utcnow()
        next_run = self._data.get(watch.id, {}).get("next_run")
        return next_run is None or datetime.fromisoformat(next_run) <= now

    def seen(self, watch_id: str) -> set[str]:
        return set(self._data.get(watch_id, {}).get("seen", []))

    def record_run(
        self,
        watch: Watch,
        new_urls: list[str],
        now: datetime | None = None,
        rng: random.Random | None = None,
    ) -> None:
        now = now or utcnow()
        rng = rng or random.Random()
        entry = self._data.setdefault(watch.id, {"seen": []})
        entry["seen"] = (entry["seen"] + new_urls)[-MAX_SEEN_PER_WATCH:]
        entry["last_run"] = now.isoformat()
        # Jitter by up to a quarter of the interval so checks land at varied times.
        jitter = rng.uniform(-0.25, 0.25) * watch.interval_minutes
        entry["next_run"] = (
            now + timedelta(minutes=watch.interval_minutes + jitter)
        ).isoformat()
        self.save()

    def prune(self, active_ids: set[str]) -> None:
        stale = [key for key in self._data if key not in active_ids]
        for key in stale:
            del self._data[key]
        if stale:
            self.save()

    def save(self) -> None:
        _write_json(self.path, self._data)


class WatchTools:
    """Voice-agent tools for creating, listing, and cancelling watches."""

    def __init__(self, store: WatchStore, browser=None) -> None:
        self.store = store
        self.browser = browser

    @property
    def tools(self) -> list:
        return [self.create_watch, self.list_watches, self.cancel_watch]

    @function_tool()
    async def create_watch(
        self,
        context: RunContext,
        description: str,
        search_query: str,
        interval_hours: float = 2,
        watch_current_page: bool = False,
    ) -> str:
        """Start a recurring background search that keeps running after the call ends.

        Use this when the user asks you to keep checking, keep an eye out, watch for,
        or notify them about something new, for example new listings, prices, or
        posts. When something new matches, the user gets a Discord message.

        Args:
            description: What counts as a match, in plain words, including details
                such as location, budget, or condition. Example: "Australian Shepherd
                puppies for sale in Egypt".
            search_query: A web search query that finds fresh results. Include the
                location and words like "for sale". Example: "Australian Shepherd
                puppies for sale Egypt".
            interval_hours: How often to check, in hours. At least half an hour.
            watch_current_page: True if the browser is showing a search results or
                listings page that should also be re-checked each time.
        """
        urls: list[str] = []
        if watch_current_page and self.browser is not None:
            try:
                urls.append((await self.browser.read_page(max_chars=1))["url"])
            except Exception as exc:
                raise ToolError("I could not read the current page.") from exc
        try:
            watch = self.store.add(
                description,
                search_query,
                round(interval_hours * 60),
                urls,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return (
            f"Watch created. Checking about every {watch.interval_minutes} minutes "
            "while the watcher program is running on the user's computer."
        )

    @function_tool()
    async def list_watches(self, context: RunContext) -> dict:
        """List the background searches currently being watched."""
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

    @function_tool()
    async def cancel_watch(self, context: RunContext, watch_id: str) -> str:
        """Stop a background search. Call list_watches first to find its id.

        Args:
            watch_id: The id of the watch to stop.
        """
        if not self.store.remove(watch_id):
            raise ToolError("No watch with that id exists.")
        return "Watch cancelled."
