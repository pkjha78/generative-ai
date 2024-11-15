import sys
import os
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv


load_dotenv(dotenv_path="../../.env", override=True)

model = "gemini-pro"
api_key = os.getenv("GOOGLE_API_KEY")
# llm = GoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))
llm = GoogleGenerativeAI(
    model=model,
    google_api_key=api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

for chunk in llm.stream("Tell me a short poem about snow"):
    sys.stdout.write(chunk)
    sys.stdout.flush()

