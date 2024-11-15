from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage   
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(first_int: int, second_int:int) -> int:
    """Multiply two integers together """
    return first_int * second_int


@tool
def add(first_int: int, second_int:int) -> int:
    """add two integers together """
    return first_int + second_int

@tool
def minus(first_int: int, second_int:int) -> int:
    """add two integers together """
    return first_int - second_int

@tool
def cube(first_int: int, second_int:int) -> int:
    """ cube """
    return first_int ** second_int  

# Initilize the OpenAIChat model

llm= ChatOpenAI(
    temperature=0, 
    model_name="gpt-4",
    max_tokens=4000,
)

tools= [multiply, add, cube]

llm_with_tool = llm.bind_tools(tools)

def call_tool(msg:AIMessage) -> Runnable:
    tool_map={tool.name:tool  for tool in tools}
    tool_calls=msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"]=tool_map[tool_call["name"]].invoke(tool_call["args"])
        return tool_calls

chain= llm_with_tool | call_tool

print(chain.invoke("what is 5 plus 11 "))


