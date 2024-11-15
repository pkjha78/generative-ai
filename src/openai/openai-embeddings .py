from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-large",
    input="The food was delicious and the waiter..."
)

print(response)