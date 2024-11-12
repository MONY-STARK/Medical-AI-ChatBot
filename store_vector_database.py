print("\n")
from src.helper import load_pdf,text_split, download_huggingface_embedding
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from uuid import uuid4
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


pc = Pinecone(api_key=PINECONE_API_KEY)

"""can create Pinecone index via code, if (not created/don't want to create) in Pinecone site 
             for more information please check documentation :)  """

index = pc.Index("medchatbot")

# Loading Data
docs = load_pdf("data")
print("Documents Loaded.......")

# Dividing Data into Chunks
chunks = text_split(docs)
print("Chunks Created.......")

# Creating id for vector database (optional) - mean id must, using uuid is optional
uuids = [str(uuid4()) for _ in chunks]
print("Creating id's for Vector Database is Done.....")

print("\n")
print("Downloading model for Embedding")
# Using Huggingface Embedding Model -> sentence-transformers/all-MiniLM-L6-v2" Don'tknow wh i chose this but this model is pretty good
embedding_model = download_huggingface_embedding()
print("Done Preparations!!")
print("\n")

# print(len(embedding_model.embed_query("Hello_world")))

# Adding Vector Embeddings to Vector Store (Pinecone in this case)
print("Initializing VectorStore.....")
vector_store = PineconeVectorStore(index, embedding=embedding_model)
print("Adding Vectors to Base")
time.sleep(5)
print("Starting...")
vector_store.add_documents(documents=chunks, id=uuids)     # this pushes vectors\embedding to database
print("Done!!")


# print Index Status after vectors are pushed into database
print("Database Status: ")
print(index.describe_index_stats())

