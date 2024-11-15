from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

prompt= "What is GenAI ,explain in simple term?"

print(llm.invoke(prompt))


