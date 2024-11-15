from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load the pre-trained model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample text data
texts = ["Kids", "Chrome Dino Pin"]

# Generate embeddings
embeddings = model.encode_plus(texts, tokenizer=tokenizer)['embeddings']

# ... (rest of your code for processing embeddings and performing similarity search)
query_text = "Kids"
query_emb = model.encode_plus(query_text, tokenizer=tokenizer)['embeddings'][0]
print(query_emb)