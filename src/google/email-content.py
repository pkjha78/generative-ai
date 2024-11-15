from langchain_google_genai import GoogleGenerativeAI
from getpass import getpass

from dotenv import load_dotenv
import os


load_dotenv(dotenv_path="../../.env", override=True)

api_key = os.getenv("GOOGLE_API_KEY")

# Setting up a mode
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)

prompts = ["Generate a professional email for a job application."]

# Assuming 'generate' is the correct method
response = llm.generate(prompts=prompts)
print(response)