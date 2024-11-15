from langchain_core.prompts import PromptTemplate
import sys
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path="../../.env", override=True)

model = "gemini-pro"
llm = GoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "How much is 2+2?"
print(chain.invoke({"question": question}))
