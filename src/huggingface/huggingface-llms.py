from langchain_huggingface.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../../.env", override=True)

llm_huggingface = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,  # Removed from model_kwargs
    max_new_tokens=64,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)

output = llm_huggingface.invoke("Write a short poem on Artificial Intelligence")
print(output)