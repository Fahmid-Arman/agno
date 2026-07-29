"""Regression: AG-UI must not flatten a team's members and its leader into one message.

Team runs interleave leader events (team_id/team_name) with member events
(agent_id/agent_name). The adapter used to keep a single text message for the whole
stream, so the last member's text and the leader's final answer were concatenated
into one assistant message with no way to tell them apart. Each producer now gets
its own message, and member messages carry the agent name in TEXT_MESSAGE_START.
"""

import json
from typing import Any, AsyncIterator, Iterator, List

import pytest

pytest.importorskip("ag_ui", reason="ag_ui not installed")

from ag_ui.core import EventType

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.base import Model
from agno.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from agno.os.interfaces.agui.stream import async_stream_agno_response_as_agui_events
from agno.run.agent import RunCompletedEvent as AgentRunCompleted
from agno.run.agent import RunContentEvent as AgentRunContent
from agno.run.agent import RunStartedEvent as AgentRunStarted
from agno.run.agent import ToolCallStartedEvent as AgentToolCallStarted
from agno.run.team import RunCompletedEvent as TeamRunCompleted
from agno.run.team import RunContentEvent as TeamRunContent
from agno.run.team import RunStartedEvent as TeamRunStarted
from agno.run.team import ToolCallStartedEvent as TeamToolCallStarted
from agno.team import Team

TEAM_RUN_ID = "team-run-1"
TEAM_ID = "research-team"
TEAM_NAME = "Research Team"


async def _collect(events: List[Any]) -> List[Any]:
    async def gen():
        for event in events:
            yield event

    return [event async for event in async_stream_agno_response_as_agui_events(gen(), "thread-1", TEAM_RUN_ID)]


def _messages(agui_events: List[Any]) -> List[dict]:
    """Fold the stream into one record per text message, in emission order."""
    by_id: dict = {}
    order: List[str] = []
    for event in agui_events:
        if event.type == EventType.TEXT_MESSAGE_START:
            by_id[event.message_id] = {"id": event.message_id, "name": event.name, "text": ""}
            order.append(event.message_id)
        elif event.type == EventType.TEXT_MESSAGE_CONTENT:
            by_id[event.message_id]["text"] += event.delta
    return [by_id[message_id] for message_id in order]


def _assert_well_formed(agui_events: List[Any]) -> None:
    """AG-UI allows only one open text message at a time; content must land inside one."""
    open_id = None
    for event in agui_events:
        if event.type == EventType.TEXT_MESSAGE_START:
            assert open_id is None, f"TEXT_MESSAGE_START for {event.message_id} while {open_id} is still open"
            open_id = event.message_id
        elif event.type == EventType.TEXT_MESSAGE_CONTENT:
            assert event.message_id == open_id, f"content for {event.message_id} but {open_id} is the open message"
        elif event.type == EventType.TEXT_MESSAGE_END:
            assert event.message_id == open_id, f"END for {event.message_id} but {open_id} is the open message"
            open_id = None
    assert open_id is None, f"message {open_id} was never closed"


def _team_run(*events: Any) -> List[Any]:
    """Wrap events in the team lifecycle a real stream always has.

    The leading TeamRunStarted is what identifies TEAM_RUN_ID as a team run, which
    is how the members nested under it are recognized as members.
    """
    return [
        TeamRunStarted(team_id=TEAM_ID, team_name=TEAM_NAME, run_id=TEAM_RUN_ID),
        *events,
        TeamRunCompleted(team_id=TEAM_ID, team_name=TEAM_NAME, run_id=TEAM_RUN_ID),
    ]


def _member_content(agent_id: str, agent_name: str, run_id: str, text: str) -> AgentRunContent:
    return AgentRunContent(
        agent_id=agent_id, agent_name=agent_name, run_id=run_id, parent_run_id=TEAM_RUN_ID, content=text
    )


def _leader_content(text: str) -> TeamRunContent:
    return TeamRunContent(team_id=TEAM_ID, team_name=TEAM_NAME, run_id=TEAM_RUN_ID, content=text)


def _member_tool(agent_id: str, agent_name: str, run_id: str, tool_call_id: str) -> AgentToolCallStarted:
    return AgentToolCallStarted(
        agent_id=agent_id,
        agent_name=agent_name,
        run_id=run_id,
        parent_run_id=TEAM_RUN_ID,
        tool=ToolExecution(tool_call_id=tool_call_id, tool_name=f"tool_{tool_call_id}", tool_args={}),
    )


async def test_member_text_is_not_merged_into_leader_message():
    """The reported bug: a member's answer and the leader's answer shared one message."""
    agui_events = await _collect(
        _team_run(
            _member_content("review-agent", "Review Agent", "member-run-1", "I reviewed the result."),
            _leader_content("Here is the final brief."),
        )
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    assert [m["text"] for m in messages] == ["I reviewed the result.", "Here is the final brief."]
    assert messages[0]["id"] != messages[1]["id"]


async def test_member_message_carries_agent_name_and_leader_message_does_not():
    agui_events = await _collect(
        _team_run(
            _member_content("research-agent", "Research Agent", "member-run-1", "I found three documents."),
            _leader_content("Here is the final brief."),
        )
    )

    messages = _messages(agui_events)
    assert messages[0]["name"] == "Research Agent"
    # Leader/team message behaviour is unchanged: no name is emitted.
    assert messages[1]["name"] is None


async def test_consecutive_members_get_separate_named_messages():
    agui_events = await _collect(
        _team_run(
            _member_content("research-agent", "Research Agent", "member-run-1", "I found three documents."),
            _member_content("review-agent", "Review Agent", "member-run-2", "I reviewed them."),
        )
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    assert [m["name"] for m in messages] == ["Research Agent", "Review Agent"]
    assert [m["text"] for m in messages] == ["I found three documents.", "I reviewed them."]
    assert messages[0]["id"] != messages[1]["id"]


async def test_deltas_from_one_member_stay_in_one_message():
    """Splitting is per producer, not per chunk — a member's own deltas must not fragment."""
    agui_events = await _collect(
        _team_run(
            _member_content("research-agent", "Research Agent", "member-run-1", "I found "),
            _member_content("research-agent", "Research Agent", "member-run-1", "three documents."),
        )
    )

    messages = _messages(agui_events)
    assert len(messages) == 1
    assert messages[0]["text"] == "I found three documents."


async def test_interleaved_members_produce_well_formed_messages():
    """Parallel delegation merges member streams, so deltas can alternate mid-answer."""
    agui_events = await _collect(
        _team_run(
            _member_content("research-agent", "Research Agent", "member-run-1", "I found "),
            _member_content("review-agent", "Review Agent", "member-run-2", "I reviewed "),
            _member_content("research-agent", "Research Agent", "member-run-1", "three documents."),
            _member_content("review-agent", "Review Agent", "member-run-2", "them."),
        )
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    # Each switch opens a new message; no text is ever attributed to the wrong agent.
    assert [(m["name"], m["text"]) for m in messages] == [
        ("Research Agent", "I found "),
        ("Review Agent", "I reviewed "),
        ("Research Agent", "three documents."),
        ("Review Agent", "them."),
    ]


async def test_same_member_invoked_twice_gets_one_message_per_run():
    """Two concurrent runs of the same agent are two answers, not one.

    A leader can emit two delegate_task_to_member calls for the same member in a
    single turn. Those run as concurrent tasks whose events are merged into one
    queue, so they share agent_id and interleave; only run_id separates them.
    """
    agui_events = await _collect(
        _team_run(
            _member_content("research-agent", "Research Agent", "member-run-1", "On topic A: "),
            _member_content("research-agent", "Research Agent", "member-run-2", "On topic B: "),
            _member_content("research-agent", "Research Agent", "member-run-1", "found two papers."),
            _member_content("research-agent", "Research Agent", "member-run-2", "found nothing."),
        )
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    # Both runs are the same agent, so both are labelled -- but they stay apart.
    assert [m["name"] for m in messages] == ["Research Agent"] * 4
    assert [m["text"] for m in messages] == [
        "On topic A: ",
        "On topic B: ",
        "found two papers.",
        "found nothing.",
    ]
    assert len({m["id"] for m in messages}) == 4, "concurrent runs of one agent must not share a message"


async def test_member_tool_call_parents_to_a_member_message():
    """A member's tool call must not be attributed to the leader's message."""
    leader_tool = ToolExecution(tool_call_id="deleg-1", tool_name="delegate_task_to_member", tool_args={})
    member_tool = ToolExecution(tool_call_id="search-1", tool_name="web_search", tool_args={"q": "x"})

    agui_events = await _collect(
        _team_run(
            TeamToolCallStarted(team_id=TEAM_ID, team_name=TEAM_NAME, run_id=TEAM_RUN_ID, tool=leader_tool),
            AgentRunStarted(
                agent_id="research-agent",
                agent_name="Research Agent",
                run_id="member-run-1",
                parent_run_id=TEAM_RUN_ID,
            ),
            AgentToolCallStarted(
                agent_id="research-agent",
                agent_name="Research Agent",
                run_id="member-run-1",
                parent_run_id=TEAM_RUN_ID,
                tool=member_tool,
            ),
        )
    )

    _assert_well_formed(agui_events)
    parents = {e.tool_call_id: e.parent_message_id for e in agui_events if e.type == EventType.TOOL_CALL_START}
    names = {m["id"]: m["name"] for m in _messages(agui_events)}

    assert parents["deleg-1"] != parents["search-1"], "member tool call reused the leader's parent message"
    assert names[parents["deleg-1"]] is None, "leader tool call should parent to an unnamed leader message"
    assert names[parents["search-1"]] == "Research Agent"


async def test_interleaved_member_tool_calls_never_borrow_another_members_parent():
    """StreamState keeps one pending-parent slot, shared by every producer.

    With parallel members that slot is repeatedly overwritten and cleared across
    producers, including when a member issues a tool call after another member has
    already streamed. A tool call must still never parent to a message belonging to
    a different member: the lookup is source-guarded, so a mismatched slot yields a
    fresh correctly-named parent instead of the wrong one.
    """
    owner = {"t1": "Research Agent", "t2": "Review Agent", "t3": "Research Agent"}

    agui_events = await _collect(
        _team_run(
            _member_content("research-agent", "Research Agent", "run-a", "A: searching. "),
            _member_tool("research-agent", "Research Agent", "run-a", "t1"),
            _member_content("review-agent", "Review Agent", "run-b", "B: reviewing. "),
            _member_tool("review-agent", "Review Agent", "run-b", "t2"),
            # Back to the first member, after the second one has taken the slot.
            _member_tool("research-agent", "Research Agent", "run-a", "t3"),
            _member_content("research-agent", "Research Agent", "run-a", "A: done."),
        )
    )

    _assert_well_formed(agui_events)
    names = {m["id"]: m["name"] for m in _messages(agui_events)}

    parented = {e.tool_call_id: names[e.parent_message_id] for e in agui_events if e.type == EventType.TOOL_CALL_START}
    assert parented == owner, "a tool call was parented to another member's message"


async def test_standalone_agent_stream_is_unchanged():
    """Backward compatibility: a non-team agent produces one unnamed message as before."""
    agui_events = await _collect(
        [
            AgentRunContent(agent_id="solo", agent_name="Solo Agent", run_id="run-1", content="Hello "),
            AgentRunContent(agent_id="solo", agent_name="Solo Agent", run_id="run-1", content="there."),
            AgentRunCompleted(agent_id="solo", agent_name="Solo Agent", run_id="run-1"),
        ]
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    assert len(messages) == 1
    assert messages[0]["text"] == "Hello there."
    assert messages[0]["name"] is None


async def test_context_provider_sub_agent_is_not_treated_as_a_team_member():
    """parent_run_id alone must not imply team membership.

    Context providers stream a sub-agent's events with parent_run_id set to the
    parent agent's run (agno/context/provider.py), and stream_sub_agent_events
    defaults to True. There is no team here, so the stream must look as it always
    has: one unnamed message.
    """
    agui_events = await _collect(
        [
            AgentRunContent(agent_id="solo", agent_name="Solo Agent", run_id="run-1", content="Checking. "),
            AgentRunContent(
                agent_id="kb-agent",
                agent_name="Knowledge Agent",
                run_id="sub-run-1",
                parent_run_id="run-1",
                content="Found it. ",
            ),
            AgentRunContent(agent_id="solo", agent_name="Solo Agent", run_id="run-1", content="Done."),
            AgentRunCompleted(agent_id="solo", agent_name="Solo Agent", run_id="run-1"),
        ]
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    assert len(messages) == 1, "a sub-agent must not open a message of its own outside a team run"
    assert messages[0]["text"] == "Checking. Found it. Done."
    assert messages[0]["name"] is None


async def test_workflow_step_agent_is_not_treated_as_a_team_member():
    """Workflow steps also set parent_run_id, and carry no team run to belong to."""
    agui_events = await _collect(
        [
            AgentRunContent(
                agent_id="step-agent",
                agent_name="Step Agent",
                run_id="step-run-1",
                parent_run_id="workflow-run-1",
                workflow_id="wf",
                workflow_run_id="workflow-run-1",
                step_id="step-1",
                step_name="research",
                content="Step output.",
            ),
            AgentRunCompleted(agent_id="step-agent", agent_name="Step Agent", run_id="step-run-1"),
        ]
    )

    _assert_well_formed(agui_events)
    messages = _messages(agui_events)

    assert len(messages) == 1
    assert messages[0]["name"] is None


# --- End-to-end over a real Team run (offline scripted model) ----------------------


class _ScriptedModel(Model):
    """Emits scripted turns offline: ('tool', name, args, id) or ('content', text)."""

    def __init__(self, model_id: str, script: List[tuple]):
        super().__init__(id=model_id, name=model_id, provider="test")
        self._script = list(script)
        self._i = 0

    def _next(self) -> ModelResponse:
        turn = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if turn[0] == "tool":
            _, name, args, tool_call_id = turn
            response = ModelResponse(role="assistant")
            response.tool_calls = [
                {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            ]
            return response
        response = ModelResponse(content=turn[1], role="assistant")
        response.event = ModelResponseEvent.assistant_response.value
        return response

    def invoke(self, *a, **k):
        return self._next()

    async def ainvoke(self, *a, **k):
        return self._next()

    def invoke_stream(self, *a, **k) -> Iterator[ModelResponse]:
        yield self._next()

    async def ainvoke_stream(self, *a, **k) -> AsyncIterator[ModelResponse]:
        yield self._next()

    def parse_provider_response(self, response: Any, **k) -> ModelResponse:
        return response if isinstance(response, ModelResponse) else ModelResponse()

    def parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response if isinstance(response, ModelResponse) else ModelResponse()

    def _parse_provider_response(self, response: Any, **k) -> ModelResponse:
        return response if isinstance(response, ModelResponse) else ModelResponse()

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response if isinstance(response, ModelResponse) else ModelResponse()


async def test_real_team_run_attributes_each_member(tmp_path):
    """Same guarantee, driven through real team plumbing rather than hand-built events."""
    db = SqliteDb(db_file=str(tmp_path / "team_identity.db"))
    researcher = Agent(
        name="Research Agent",
        id="research-agent",
        db=db,
        telemetry=False,
        model=_ScriptedModel("m-research", [("content", "I found three relevant documents.")]),
    )
    reviewer = Agent(
        name="Review Agent",
        id="review-agent",
        db=db,
        telemetry=False,
        model=_ScriptedModel("m-review", [("content", "I reviewed the result and it checks out.")]),
    )
    team = Team(
        name="Research Team",
        id="research-team",
        db=db,
        telemetry=False,
        members=[researcher, reviewer],
        model=_ScriptedModel(
            "m-leader",
            [
                ("tool", "delegate_task_to_member", {"member_id": "research-agent", "task": "research"}, "tc-1"),
                ("tool", "delegate_task_to_member", {"member_id": "review-agent", "task": "review"}, "tc-2"),
                ("content", "Here is the final brief."),
            ],
        ),
        show_members_responses=True,
    )

    async def gen():
        async for event in team.arun("research this", session_id="s1", stream=True, stream_events=True):
            yield event

    agui_events = [event async for event in async_stream_agno_response_as_agui_events(gen(), "thread-1", "agui-run-1")]

    _assert_well_formed(agui_events)
    with_text = [m for m in _messages(agui_events) if m["text"]]

    assert [(m["name"], m["text"]) for m in with_text] == [
        ("Research Agent", "I found three relevant documents."),
        ("Review Agent", "I reviewed the result and it checks out."),
        (None, "Here is the final brief."),
    ]
