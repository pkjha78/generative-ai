from langchain_core.prompts import  ChatPromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

def main():
    prompt= ChatPromptTemplate.from_template(" Tell me key achivements of {name} in 4 bulleted points")
    llm=OpenAI()
    chain=prompt|llm
    response=chain.invoke({"name":"Virat Kohli"})
    print(response)
    
if __name__ == "__main__":
    main()