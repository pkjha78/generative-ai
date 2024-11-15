
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load data and build an index
PERSIST_DIR = "../storage"
documents = SimpleDirectoryReader("../data/txt").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=PERSIST_DIR)

# Query your data
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

