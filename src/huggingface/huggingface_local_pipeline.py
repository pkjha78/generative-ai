from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 128},
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
# Create Chain: With the model loaded into memory, you can compose it with a prompt to form a chain.
chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))

## They can also be loaded by passing in an existing transformers pipeline directly
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
hf = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))