from transformers import AutoTokenizer

def count_tokens(text, model_name="gpt2"):
    """Counts the number of tokens in a given text.

    Args:
        text (str): The text to count tokens in.
        model_name (str, optional): The name of the model to use for tokenization. Defaults to "gpt2".

    Returns:
        int: The number of tokens in the text.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_text = tokenizer.tokenize(text)
    return len(tokenized_text)

# Example usage:
prompt = "What is the meaning of life?"
token_count = count_tokens(prompt)
print(f"Token count: {token_count}")