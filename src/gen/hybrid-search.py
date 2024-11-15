# generate an unique id for this session
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
#from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
#    HybridQuery,
#)

UID = datetime.now().strftime("%m%d%H%M")

print(UID)



CSV_URL = "https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/sample-apps/photo-discovery/ag-web/google_merch_shop_items.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(CSV_URL)
print(df["title"])

# Sample Text Data
corpus = df.title.tolist()

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and Transform
vectorizer.fit_transform(corpus)

def get_sparse_embedding(text):
    # Transform Text into TF-IDF Sparse Vector
    tfidf_vector = vectorizer.transform([text])

    # Create Sparse Embedding for the New Text
    values = []
    dims = []
    for i, tfidf_value in enumerate(tfidf_vector.data):
        values.append(float(tfidf_value))
        dims.append(int(tfidf_vector.indices[i]))
    return {"values": values, "dimensions": dims}

text_text = "Chrome Dino Pin"
sparse_text = get_sparse_embedding(text_text)
print(sparse_text)

# Create an input data file
items = []
for i in range(len(df)):
    id = i
    title = df.title[i]
    sparse_embedding = get_sparse_embedding(title)
    items.append({"id": id, "title": title, "sparse_embedding": sparse_embedding})

print(items[:5])

# output as a JSONL file and save to the GCS bucket
with open("../data/items.json", "w") as f:
    for item in items:
        f.write(f"{item}\n")


# create HybridQuery
query_text = "Kids"
query_emb = get_sparse_embedding(query_text)
query = HybridQuery(
    sparse_embedding_dimensions=query_emb["dimensions"],
    sparse_embedding_values=query_emb["values"],
)

# build a query request
response = my_index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_TOKEN_INDEX_ID,
    queries=[query],
    num_neighbors=5,
)

# print results
for idx, neighbor in enumerate(response[0]):
    title = df.title[int(neighbor.id)]
    print(f"{title:<40}")