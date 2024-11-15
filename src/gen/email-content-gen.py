from transformers import pipeline
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env", override=True)

model_tag = "postbot/gpt2-medium-emailgen"
generator = pipeline(
              'text-generation', 
              model=model_tag, 
            )
            
prompt = """
Hello, 

Following up on the bubblegum shipment."""

prompt1 = """
Write an email with the subject: "Meeting Invitation: Project Kickoff."

Content:
Please join us for a project kickoff meeting to discuss the goals, timeline, and roles for the upcoming project.

Additional instructions:
Keep the tone formal and professional. Include the meeting details, such as date, time, and meeting link.
"""

result = generator(
    prompt,
    max_length=64,
    do_sample=False,
    early_stopping=True,
) # generate
print(result[0]['generated_text'])
