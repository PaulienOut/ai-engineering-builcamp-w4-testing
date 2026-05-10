from pydantic import BaseModel
from pydantic_ai import Agent

import sql_tools


class SQLResult(BaseModel):
    """Result from SQL query execution."""
    sql_query: str
    result_text: str
    row_count: int


# Initialize SQLTools instance
tools = sql_tools.SQLTools()

# Create agent with instructions
agent = Agent(
    model="gpt-4o-mini",
    result_type=SQLResult,
    tools=[tools.get_schema, tools.run_sql],
    system_prompt="""You are a SQL query assistant. When a user asks for data from the trips database:

1. ALWAYS start by calling get_schema() to understand the table structure and available columns
2. Based on the schema, construct and run appropriate SQL queries using run_sql()
3. Return the results in a SQLResult with the query that was executed, the formatted result text, and the number of rows returned

Be helpful and accurate with your SQL queries.""",
)
