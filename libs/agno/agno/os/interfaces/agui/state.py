import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agno.utils.log import log_warning


@dataclass
class StreamState:
    """Per-stream state for AG-UI event translation.

    Tracks message lifecycle, tool calls, reasoning sessions, and state deltas.
    All handlers receive this object and mutate it as events flow through.

    Text Message Lifecycle:
        CLOSED (initial)      OPEN                   CLOSED
        text_message_id=""    text_message_id=X      text_message_id=X (persists!)
        text_message_open=F   text_message_open=T    text_message_open=F

    The text_message_id persists after close so tool calls can parent to it.

    Team runs interleave events from the leader and its members, so every message
    also records the key of the agent/team that produced it. A message is only
    reused while the producer stays the same.
    """

    # Text message tracking
    text_message_id: str = ""
    text_message_open: bool = False
    text_message_source: str = ""

    # Run ids of team events seen so far. A run whose parent is one of these is a
    # team member; parent_run_id on its own also covers workflow steps and context
    # provider sub-agents, which are not members.
    team_run_ids: Set[str] = field(default_factory=set)

    # Tool call tracking
    active_tool_call_ids: Set[str] = field(default_factory=set)
    ended_tool_call_ids: Set[str] = field(default_factory=set)
    pending_tool_calls_parent_id: str = ""
    pending_tool_calls_parent_source: str = ""

    # Reasoning tracking
    reasoning_message_id: Optional[str] = None
    reasoning_step_count: int = 0

    # State delta tracking
    _last_snapshot: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # Run context
    thread_id: str = ""
    run_id: str = ""
    run_state: Optional[Dict[str, Any]] = None

    def open_text_message(self, source: str = "") -> str:
        self.text_message_id = str(uuid.uuid4())
        self.text_message_open = True
        self.text_message_source = source
        return self.text_message_id

    def close_text_message(self) -> None:
        # ID persists for tool call parenting — only flag changes
        self.text_message_open = False

    def start_tool_call(self, tool_call_id: str) -> None:
        self.active_tool_call_ids.add(tool_call_id)

    def end_tool_call(self, tool_call_id: str) -> None:
        self.active_tool_call_ids.discard(tool_call_id)
        self.ended_tool_call_ids.add(tool_call_id)

    def get_parent_message_id_for_tool_call(self, source: str = "") -> str:
        """Parent message for a tool call, or "" when the caller must create one.

        A candidate only counts when it belongs to the same producer: parenting a
        member's tool call to the team leader's message misattributes the call.
        """
        # pending_tool_calls_parent_id used for sequential tools after message close
        if self.pending_tool_calls_parent_id:
            if self.pending_tool_calls_parent_source == source:
                return self.pending_tool_calls_parent_id
            return ""
        # text_message_id persists after close
        if self.text_message_source == source:
            return self.text_message_id
        return ""

    def set_pending_tool_calls_parent_id(self, parent_id: str, source: str = "") -> None:
        self.pending_tool_calls_parent_id = parent_id
        self.pending_tool_calls_parent_source = source

    def clear_pending_tool_calls_parent_id(self) -> None:
        self.pending_tool_calls_parent_id = ""
        self.pending_tool_calls_parent_source = ""

    def start_reasoning(self) -> str:
        self.reasoning_message_id = str(uuid.uuid4())
        self.reasoning_step_count = 0
        return self.reasoning_message_id

    def ensure_reasoning_started(self) -> Tuple[str, bool]:
        if self.reasoning_message_id is not None:
            return self.reasoning_message_id, False
        return self.start_reasoning(), True

    def next_reasoning_step(self) -> int:
        self.reasoning_step_count += 1
        return self.reasoning_step_count

    def end_reasoning(self) -> None:
        self.reasoning_message_id = None
        self.reasoning_step_count = 0

    def set_state_snapshot(self, state: Dict[str, Any]) -> None:
        self._last_snapshot = copy.deepcopy(state)

    def compute_state_delta(self, current_state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if self._last_snapshot is None:
            return None
        try:
            import jsonpatch

            patch = jsonpatch.make_patch(self._last_snapshot, current_state)
            ops = patch.patch
            if not ops:
                return None
            return ops
        except Exception as e:
            log_warning(f"Failed to compute state delta: {e}")
            return None
