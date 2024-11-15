from dotenv import load_dotenv
from openai import OpenAI

#load_dotenv(".env",override=True)
load_dotenv(dotenv_path="../../.env", override=True)

openai_client=OpenAI()


available_models=openai_client.models.list()

print(available_models)

#or model in available_models['data']:
#    print(model['id'])



