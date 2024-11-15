# LangChain prompt templates

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv(dotenv_path="../../.env", override=True)

# Initialize an LLM application 
llm = ChatGoogleGenerativeAI(model = "gemini-pro") 


## consider the following prompt:
prompt = """ 
Instruction: Answer the question based on the context below. If you cannot answer the question with the given context, answer with "I don't know.". 
Context: Codecademy is an interactive online learning platform offering courses in various programming languages and tech skills. It provides a hands-on, project-based approach to learning, allowing users to write and execute code directly in the browser. The platform covers topics such as web development, data science, computer science, and machine learning. Codecademy features a mix of free and paid content, with the Pro membership granting access to advanced courses, quizzes, and real-world projects. The site also includes community forums, career advice, and a personalized learning path to help users achieve their specific goals. 
Query: How many users does Codecademy have? 
""" 

## create a prompt template
prompt_template = """ 
Instruction: Answer the question based on the context below. If you cannot answer the question with the given context, answer with "I don't know.". 
Context: Codecademy is an interactive online learning platform offering courses in various programming languages and tech skills. It provides a hands-on, project-based approach to learning, allowing users to write and execute code directly in the browser. The platform covers topics such as web development, data science, computer science, and machine learning. Codecademy features a mix of free and paid content, with the Pro membership granting access to advanced courses, quizzes, and real-world projects. The site also includes community forums, career advice, and a personalized learning path to help users achieve their specific goals. 
Query: {input_query} 
""" 

## How to Create Prompt Templates in LangChain?
# The from_template() function
# The PromptTemplate() function

## Generate Prompt Templates Using the from_template() Function

# Create a prompt template 
prompt_template = PromptTemplate.from_template(template="Suggest one name for a restaurant in {country} that serves {cuisine} food.") 

# Create a prompt using the prompt template 
prompt = prompt_template.format(cuisine="Mexican", country="USA") 
print("The prompt is:",prompt) 

# Generate results using the LLM application 
result = llm.invoke(prompt) 
print("The output is:", result.content)


## Generate Prompt Templates Using the PromptTemplate() Function

# Create a prompt template 
prompt_template = PromptTemplate(
  input_variables=["country", "cuisine"], 
  template = "Suggest one name for a restaurant in {country} that serves {cuisine} food." ) 

# Create a prompt using the prompt template 
prompt = prompt_template.format(cuisine = "Chinese", country = "India") 
print("The prompt is: ", prompt) 

# Generate results using the LLM application 
result = llm.invoke(prompt) 
print("The output is:", result.content)