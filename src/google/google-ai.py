from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path="../../.env", override=True)

model = "gemini-pro" # Note: gemini-pro is an alias for gemini-1.0-pro.

# Other new Models of Gemini Pro
model_pro_001 = "gemini-1.0-pro-001"
model_pro_002 = "gemini-1.5-pro-002"

llm = GoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))

print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)