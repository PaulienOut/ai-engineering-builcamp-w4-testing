from typing import Any, Dict
from dataclasses import dataclass
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.run import AgentRun
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai._agent_graph import UserPromptNode, ModelRequestNode, CallToolsNode
from jaxn import JSONParserHandler, StreamingJSONParser

import sql_tools
from pydantic import BaseModel


DEFAULT_INSTRUCTIONS = """
You are a SQL query assistant. When a user asks for data from the trips database:

1. ALWAYS start by calling get_schema() to understand the table structure and available columns
2. Based on the schema, construct and run appropriate SQL queries using run_sql()
3. Return the results in a SQLResult with the query that was executed, the formatted result text, and the number of rows returned

Be helpful and accurate with your SQL queries.
""".strip()


class SQLResult(BaseModel):
    """Result from SQL query execution."""
    sql_query: str
    result_text: str
    row_count: int


@dataclass
class SQLAgentConfig:
    model: str = 'openai:gpt-4o-mini'
    name: str = 'sql_agent'
    instructions: str = DEFAULT_INSTRUCTIONS


def create_agent(
    config: SQLAgentConfig,
    sql_tools_instance: sql_tools.SQLTools
) -> Agent:
    tools = [sql_tools_instance.get_schema, sql_tools_instance.run_sql]

    sql_agent = Agent(
        name=config.name,
        model=config.model,
        instructions=config.instructions,
        tools=tools,
        output_type=SQLResult
    )
    return sql_agent


class NamedCallback:
    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return
        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


async def run_agent(
        agent: Agent,
        user_prompt: str,
        message_history=None
    ) -> AgentRunResult:
    callback = NamedCallback(agent)

    if message_history is None:
        message_history = []

    result = await agent.run(
        user_prompt,
        event_stream_handler=callback,
        message_history=message_history,
        output_type=SQLResult
    )

    return result

class SQLResponseHandler(JSONParserHandler):
    def on_object_field_end(self, path: str, field_name: str, value: Any = None) -> None:
        if path == '' and field_name == 'sql_query':
            print('\nsql query:', value)
        elif path == '' and field_name == 'row_count':
            print('row count:', value)

    def on_array_item_end(self, path: str, field_name: str, item: Dict[str, Any] = None) -> None:
        pass  # No arrays in SQLResult


async def run_agent_stream(
    agent: Agent,
    user_prompt: str,
    message_history=None
):
    runner = AgentStreamRunner(agent, SQLResponseHandler())
    return await runner.run(user_prompt, message_history)


def run_agent(agent: Agent, user_prompt: str, message_history=None):
    import asyncio
    return asyncio.run(run_agent_stream(agent, user_prompt, message_history))


class AgentStreamRunner:
    def __init__(self, agent: Agent, handler: JSONParserHandler):
        self.agent = agent
        self.handler = handler

    async def run(self, user_prompt: str, message_history=None):
        if message_history is None:
            message_history = []
        async with self.agent.iter(
            user_prompt,
            message_history=message_history,
            output_type=SQLResult
        ) as agent_run:
            async for node in agent_run:
                if isinstance(node, UserPromptNode):
                    await self.process_user_node(node, agent_run)
                elif isinstance(node, ModelRequestNode):
                    await self.process_model_request_node(node, agent_run)
                elif isinstance(node, CallToolsNode):
                    await self.process_call_tools_node(node, agent_run)
            return agent_run.result

    async def process_user_node(self, node: UserPromptNode, agent_run: AgentRun):
        print(f"USER PROMPT ({self.agent.name}): {node.user_prompt}")

    async def process_model_request_node(self, node: ModelRequestNode, agent_run: AgentRun):
        args_so_far = ""
        parser = StreamingJSONParser(self.handler)
        async with node.stream(agent_run.ctx) as stream:
            async for response in stream.stream_responses():
                for part in response.parts:
                    if part.part_kind != 'tool-call':
                        continue
                    if part.tool_name != 'final_result':
                        continue
                    args_new = part.args
                    args_new_chunk = args_new[len(args_so_far):]
                    args_so_far = args_new
                    parser.parse_incremental(args_new_chunk)

    async def process_call_tools_node(self, node: CallToolsNode, agent_run: AgentRun):
        async with node.stream(agent_run.ctx) as events:
            async for event in events:
                if not isinstance(event, FunctionToolCallEvent):
                    continue
                tool_name = event.part.tool_name
                args = event.part.args
                print(f"TOOL CALL ({self.agent.name}): {tool_name}({args})")


# Initialize SQLTools instance
sql_tools_instance = sql_tools.SQLTools()
