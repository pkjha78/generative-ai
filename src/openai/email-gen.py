from openai import OpenAI
from dotenv import load_dotenv

def generate_text(prompt):
    """Generates text based on the given prompt.

    Args:
        prompt: The prompt to generate text from.

    Returns:
        The generated text.
    """

    #openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
        
    return response.choices[0].message.content

if __name__ == "__main__":
    load_dotenv(dotenv_path="../../.env", override=True)
    prompt = input("Enter a prompt: ")
    generated_text = generate_text(prompt)
    print(generated_text)