from langchain_google_genai import ChatGoogleGenerativeAI
#from IPython.display import display, Markdown
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv(dotenv_path="../../.env", override=True)

#def display_code_as_markdown(code):
#    display(Markdown(f"```python\n{code}\n```"))
   


# Initialize an LLM application 
llm = ChatGoogleGenerativeAI(model = "gemini-pro") 


template = """
Act as a computer science instructor.
Given:
- Coding challenge: {challenge}
- Coding language: {language}
Provide the student with step-by-step instructions to solve the coding challenge using pseudo-code as in the example below, be succinct, keep this part short. Then provide the answer in code as exemplified in the example below under code.
Example:
Pseudo-code:
1. Create an empty result array.
2. Iterate over input array.
3. For each element in input array:
a. Count the number of times the element appears in the input array.
b. If the count is even, add the element to the result array.
4. Return the result array.

{language} code:
def even_elements(arr):
result = []
for element in arr:
count = arr.count(element)
if count % 2 == 0:
result.append(element)
return result

Pseudo-code:
"""
prompt_template = PromptTemplate.from_template(template=template)

# Create a prompt using the prompt template 
prompt = prompt_template.format(language="python", challenge="Write a function that returns the greatest common factor between num1 and num2") 
print("The prompt is:", prompt)


# Generate results using the LLM application 
result = llm.invoke(prompt)
print("The output is:", result.content)


#chain = prompt_template | llm

#llm_output = chain.invoke({"challenge": "Write a function that returns the greatest common factor between num1 and num2", "language": "javascript"})

#display_code_as_markdown(llm_output)

#print(llm_output)