from langchain_core.prompts import  PromptTemplate
from langchain_openai import AzureOpenAI
from dotenv import load_dotenv
import os


def main():
    load_dotenv(dotenv_path="../../.env", override=True)
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
    print("Azure OpenAI Endpoint "+ endpoint)

    prompt= PromptTemplate.from_template(" Tell me key achivements of {name} in 4 bulleted points")
    llm=AzureOpenAI(deployment_name="gpt-35-turbo", max_tokens=2048)
    chain=prompt|llm
    response=chain.invoke({"name":"Sachin Tendulkar"})
    print(response)
    
if __name__ == "__main__":
    main()