## Huggingface Endpoints Example with Prompt Template.
#  the latest code where all the new library has been used with no excpection in output
from langchain_huggingface.llms import HuggingFaceEndpoint, HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../../.env", override=True)

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    max_new_tokens=128,
    temperature=0.5,
    top_k=1,
    top_p=0.95,
    typical_p=0.95,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)
#llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain =  prompt | llm
print(llm_chain.invoke(question))
