from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(dotenv_path="../../.env", override=True)

client = OpenAI()

response = client.images.generate(
    prompt="I want dog on moon (chandrayan)",
    size="1024x1024"
)

print(response.data[0].url)