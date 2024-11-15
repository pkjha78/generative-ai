from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone  # Alias for langchain's Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec # Official Pinecone library
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


load_dotenv(dotenv_path="../../.env", override=True)

private_pinecone_api_key = os.getenv("PINECONE_API_KEY")
private_open_api_key = os.getenv("OPENAI_API_KEY")

# Loading the PDF
loader = PyPDFLoader("../data/The-Book-Thief.pdf")
pages = loader.load()


# Splitting the Text
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=5000, chunk_overlap=200)
texts = text_splitter.split_documents(pages)

num_documents = len(texts)
print(f"Now our book is split up into {num_documents} documents")

# Generating Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

## Creating a Knowledge Base Index

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
#print(pc.list_indexes())
index_name = "the-book-thief"

#pc.delete_index(index_name)
# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
  # if does not exist, create index
  # Do something, such as create the index
  pc.create_index(
    name=index_name,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(
      cloud="aws",
      region="us-east-1"
    )
  )

#print(pc.list_indexes())
docsearch = LangchainPinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# Querying the Knowledge Base
llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name='gpt-3.5-turbo', openai_api_key=os.environ.get("OPENAI_API_KEY"))


# Retrieval
index_name = "the-book-thief"
text_field = "text"
# connect to index
index = pc.Index(index_name)
# view index stats
#print(index.describe_index_stats())

vectorstore = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

query = "Describe the scenarios when the Death met the book thief"

# Result
docs = vectorstore.similarity_search(query=query, k=3)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

output = qa.invoke(query)

print(output)

