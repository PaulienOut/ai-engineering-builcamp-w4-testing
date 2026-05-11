import pytest 



from sql_agent import SQLAgentConfig, create_agent, run_agent, sql_tools_instance


from tests.utils import run_agent_test
from tests.judge import assert_criteria


@pytest.fixture(scope="module")
def agent():
    tools = sql_tools_instance
    agent_config = SQLAgentConfig()

    agent = create_agent(agent_config, tools)

    return agent

@pytest.mark.asyncio
async def test_agent_highest_fare_hour(agent):
    user_prompt = 'Which hour of the day has the highest average fare amount?'
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "the SQL query correctly calculates average fare by hour of day",
        "the result identifies a specific hour as having the highest average fare",
        "the result includes the actual average fare amount"
    ])
