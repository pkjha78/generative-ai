from dotenv import load_dotenv
from openai import OpenAI

#load_dotenv()
load_dotenv(dotenv_path="../../.env", override=True)


client = OpenAI()

prompt= "What is GenAI, explain in simple term?"

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print(completion.choices[0].message.content)