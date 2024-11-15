from langchain_core.prompts import  PromptTemplate
from langchain_openai import OpenAI
import streamlit as st
from dotenv import load_dotenv


load_dotenv(dotenv_path="../../.env", override=True)


def llm_chain():
    prompt= PromptTemplate.from_template(" Tell me key achivements of {name} in 4 bulleted points")
    llm=OpenAI()
    chain=prompt|llm
    response=chain.invoke({"name":"Mahatma Gandhi"})
    print(response)

# refactor above main function to use streamlit

def main():
    st.title("Prompt Template Demo")
    user_input = st.text_input("Please enter name: ")
    prompt= PromptTemplate.from_template(" Tell me key achivements of {name} in 4 bulleted points")
    llm=OpenAI()
    chain=prompt|llm
    response=chain.invoke({"name": user_input})
    st.write(response)
    
if __name__ == "__main__":
    main()