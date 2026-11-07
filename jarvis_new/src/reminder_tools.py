"""Voice Cuppy's reminder tools. Reminders arrive as Discord messages."""

from __future__ import annotations

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

from reminders import (
    ReminderStore,
    tool_cancel_reminder,
    tool_list_reminders,
    tool_set_reminder,
)


class ReminderTools:
    def __init__(self, store: ReminderStore | None = None) -> None:
        self.store = store or ReminderStore()

    @property
    def tools(self) -> list:
        return [self.set_reminder, self.list_reminders, self.cancel_reminder]

    @function_tool()
    async def set_reminder(
        self,
        context: RunContext,
        message: str,
        at: str | None = None,
        in_minutes: float | None = None,
        repeat: str = "none",
    ) -> dict:
        """Remind the user about something later, by Discord message.

        Use this when the user asks to be reminded of something, or asks you to
        tell or message them at a certain time. Give either `at` or `in_minutes`.

        Args:
            message: What to remind them about, written as the reminder itself,
                for example "Take the chicken out of the freezer".
            at: Local date and time, in the form 2026-11-07T15:30. Work it out
                from the current date and time in your instructions.
            in_minutes: Minutes from now, for requests like "in 20 minutes".
            repeat: "none", "daily", or "weekly".
        """
        try:
            return tool_set_reminder(self.store, message, at, in_minutes, repeat)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @function_tool()
    async def list_reminders(self, context: RunContext) -> dict:
        """List upcoming reminders, along with the current date and time."""
        return tool_list_reminders(self.store)

    @function_tool()
    async def cancel_reminder(self, context: RunContext, reminder_id: str) -> dict:
        """Cancel a reminder. Call list_reminders first to find its id.

        Args:
            reminder_id: The id of the reminder to cancel.
        """
        result = tool_cancel_reminder(self.store, reminder_id)
        if not result["cancelled"]:
            raise ToolError("No upcoming reminder has that id.")
        return result
