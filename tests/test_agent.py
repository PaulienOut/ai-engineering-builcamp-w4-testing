import os

import pytest


from sql_agent import SQLAgentConfig, create_agent, run_agent, sql_tools_instance
from utils import collect_tools, run_agent_test


@pytest.fixture(scope="module")
def agent():
    tools = sql_tools_instance
    agent_config = SQLAgentConfig()

    agent = create_agent(agent_config, tools)
    print(f"Created agent with name: {agent.name}")
    return agent


@pytest.mark.asyncio
async def test_agent_counts_trips_more_than_5_passengers(agent):
    """Ensure the agent returns a typed SQLResult with the exact trip count of 22413."""
  
    prompt = 'How many trips had more than 5 passengers?'

    result = await run_agent_test(agent, prompt)

    assert result.output is not None
    assert isinstance(result.output.sql_query, str)
    assert result.output.sql_query.strip() != ''
    assert isinstance(result.output.result_text, str)
    assert '22413' in result.output.result_text, (
        f'Expected the result text to contain the exact count 22413, got: {result.output.result_text}'
    )


@pytest.mark.asyncio
async def test_right_tools_called(agent):
    """
    Ensure that get_schema is the first tool called, 
    and that run_sql is called within the process
    """

    prompt = 'What is the most common payment type?'

    result = await run_agent_test(agent, prompt)

    messages = result.new_messages()
    tool_calls = collect_tools(messages)

    first_call = tool_calls[0]
    print(f"second tool call: {tool_calls[1].name}")
    assert first_call.name == 'get_schema', f'Expected first tool call to be get_schema, got {first_call.name}'
