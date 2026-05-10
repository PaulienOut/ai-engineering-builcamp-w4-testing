import dotenv
dotenv.load_dotenv()

from pydantic_ai import Agent, RunUsage
from sql_agent import (
    SQLAgentConfig,
    create_agent,
    run_agent,
    run_agent_stream,
    DEFAULT_INSTRUCTIONS,
    sql_tools_instance
)

def print_messages(messages):
    for m in messages:
        print(m.kind)
        for p in m.parts:
            part_kind = p.part_kind
            if part_kind == 'user-prompt':
                print('  USER:', p.content)
            if part_kind == 'tool-call':
                print('  TOOL CALL:', p.tool_name, p.args)
            if part_kind == 'tool-return':
                print('  TOOL RETURN:', p.tool_name, p.content)
            if part_kind == 'text':
                print('  TEXT:', p.content)

if __name__ == '__main__':
    user_prompt = "What's the average trip distance for rides with 2 passengers?"
    
    config = SQLAgentConfig()
    agent = create_agent(config, sql_tools_instance)
    
    print("Running agent...")
    result = run_agent(agent, user_prompt)
    print("Result:", result)
    print("Usage:", result.usage())
    
    print("\nMessages:")
    print_messages(result.all_messages())
