from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
hf = HuggingFacePipeline(pipeline=pipe)

template = """
Write an email with the subject: {subject}.

Content:
{Brief description of the email's purpose}

Additional instructions:
{Specific guidelines, tone, or style requirements}

"""

prompt = PromptTemplate.from_template(template)

chain = prompt | hf

print(chain.invoke({
    "subject": "Meeting Invitation: Project Kickoff",
    "Brief description of the email's purpose": "Please join us for a project kickoff meeting...",
    "Specific guidelines, tone, or style requirements": "Keep the tone formal and professional. Include meeting details."
}))