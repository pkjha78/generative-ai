
from langchain_community.llms import huggingface_hub
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../../.env", override=True)

llm_huggingface =  huggingface_hub.HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={ "temperature": 0, "max_length":64 } ,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)

output=llm_huggingface.invoke("Write a short poem on Artificial Intelligence")
print(output)