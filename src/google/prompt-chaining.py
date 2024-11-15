## Prompt chaining is a technique that involves breaking down a complex task into a series of smaller, 
# interconnected prompts, where the output of one prompt serves as the input for the next, guiding the 
# LLM through a structured reasoning process.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv(dotenv_path="../../.env", override=True)


## Defining a function to interact with Gemini
def get_completion(prompt, model="gemini-pro"):
    try:
        # Initialize an LLM application 
        llm = ChatGoogleGenerativeAI(model = model)
        return llm.invoke(prompt)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
## Chaining multiple prompts
def prompt_chain(initial_prompt, follow_up_prompts):
    result = get_completion(initial_prompt)
    if result is None:
        return "Initial prompt failed."
    print(f"Initial output: {result}\n")
    for i, prompt in enumerate(follow_up_prompts, 1):
        full_prompt = f"{prompt}\n\nPrevious output: {result}"
        result = get_completion(full_prompt)
        if result is None:
            return f"Prompt {i} failed."
        print(f"Step {i} output: {result}\n")
    return result

## Usages of multi prompt
initial_prompt = "Summarize the key trends in global temperature changes over the past century."
follow_up_prompts = [
    "Based on the trends identified, list the major scientific studies that discuss the causes of these changes.",
    "Summarize the findings of the listed studies, focusing on the impact of climate change on marine ecosystems.",
    "Propose three strategies to mitigate the impact of climate change on marine ecosystems based on the summarized findings."
]
final_result = prompt_chain(initial_prompt, follow_up_prompts)
print("Final result:", final_result)

