## Introduction to Generative AI with Python
from transformers import pipeline

# Load the Question-Answering Pipeline
question_answerer = pipeline("question-answering")

# Define the Question and Context
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris."

# Ask the Question
answer = question_answerer(question=question, context=context)
print(answer['answer'])
