from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate

load_dotenv()

# Define the math tools
@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together"""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers"""
    return first_int + second_int

@tool
def exponentize(base: int, exponent: int) -> int:
    """Raise base to the power of exponent"""
    return base ** exponent

# Initialize the language model
llm = OpenAI(
    temperature=0,
    max_tokens=4000
)

# Create a proper prompt template including required variables
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "intermediate_steps"],
    template=(
        "You are a math assistant. Solve the following expression: {input}\n\n"
        "Available tools: {tool_names}\n"
        "Tools: {tools}\n"
        "Scratchpad: {agent_scratchpad}\n"
        "Intermediate Steps: {intermediate_steps}"
    )
)

# Create the agent with the tools and prompt
agent = create_react_agent(
    llm=llm,
    tools=[multiply, add, exponentize],
    prompt=prompt_template
)

# User input for calculation
input_data = {
    "input": "Compute 3 to the power of 5, add 12 and 3, multiply the results, and then square the final answer.",
    "agent_scratchpad": "",  # Initialize the scratchpad as an empty string
    "tools": "multiply, add, exponentize",  # Comma-separated tool names
    "tool_names": "multiply, add, exponentize",  # Pass tool names as well
    "intermediate_steps": []  # Initialize as an empty list
}

# Invoke the agent and handle the output
try:
    result = agent.invoke(input_data)  # Pass the input data including intermediate_steps
    print("Result:", result)
except KeyError as e:
    print(f"KeyError: {e}. Please check the input data and ensure all required keys are provided.")
except Exception as e:
    print(f"An error occurred: {e}")