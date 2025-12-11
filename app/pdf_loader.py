import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import chroma, Chroma

# Load PDF data
loader = PyPDFLoader(file_path="RULES & PLAYING CONDITIONS -Creative Software.pdf")
docs = loader.load()

# Chunk data, split text into chunks of 1000 characters with 200 character overlap to feed the AI
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Instantiate the local model
local_embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# Embed and store, turns text into vectors and save locally in a folder, and we call it db
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=local_embeddings,
    persist_directory= "./db"
)

print(f"Stored {len(splits)} in vector database!")