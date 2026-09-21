r"""Model output holding an unpaired surrogate must not abort the run.

``json.loads`` turns an unpaired ``\udXXX`` escape in a model response into a lone
surrogate, which UTF-8 cannot encode. The tool-invocation fingerprint hashes the
UTF-8 bytes of the call, so such a call used to raise ``UnicodeEncodeError`` out of
the run loop before the tool was ever invoked.
"""

from __future__ import annotations

import pytest

from agents import Agent, Runner, function_tool
from agents._tool_invocation import tool_invocation_identity
from agents.testing import ScriptedModel, assistant_message, function_call

pytestmark = pytest.mark.asyncio

LONE_SURROGATE = "\ud83d"


@function_tool
def lookup(city: str) -> str:
    return "sunny"


async def test_fingerprint_accepts_lone_surrogate_arguments() -> None:
    identity = tool_invocation_identity(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"city": "Tok' + LONE_SURROGATE + '"}',
        }
    )

    assert identity is not None


async def test_fingerprint_is_stable_for_encodable_arguments() -> None:
    call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": '{"city": "Tokyo"}',
    }

    assert tool_invocation_identity(call) == tool_invocation_identity(dict(call))


async def test_run_completes_with_lone_surrogate_tool_arguments() -> None:
    model = ScriptedModel(
        [
            [function_call("lookup", {"city": "Tok" + LONE_SURROGATE}, call_id="call_1")],
            [assistant_message("done")],
        ]
    )
    agent = Agent(name="weather", model=model, tools=[lookup])

    result = await Runner.run(agent, "weather?")

    assert result.final_output == "done"


async def test_run_completes_with_lone_surrogate_handoff_arguments() -> None:
    target = Agent(name="target", model=ScriptedModel([[assistant_message("from target")]]))
    model = ScriptedModel(
        [[function_call("transfer_to_target", {"note": LONE_SURROGATE}, call_id="call_1")]]
    )
    agent = Agent(name="source", model=model, handoffs=[target])

    result = await Runner.run(agent, "hand off")

    assert result.final_output == "from target"
