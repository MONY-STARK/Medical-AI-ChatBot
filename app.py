from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embedding
from langchain_pinecone import  PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from src.prompt import *


load_dotenv()

HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
PINECONE_API_KEY  = os.environ.get("PINECONE_API_KEY")

app = Flask(__name__)

print("Preparing Embedding Model")
embedding_model = download_huggingface_embedding()
print("\n")

pincone = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medchatbot"

print("pulling embeddings from database.....")
# To pull existing database from pinecone
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
print("vectordatabase ready!!")

query = " what is Abortion and  is types?"

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

print("\n")
print("Init-ing llm")
llm = HuggingFaceEndpoint(repo_id=repo_id,
                           max_new_tokens=512,
                           temperature=0.7,
                           huggingfacehub_api_token=HUGGINGFACE_API_KEY)

Prompt = PromptTemplate(
    template = prompt_templete,
    input_variables = ["context",  "question"])

chain_type_kwargs={"prompt":Prompt}

print("creating QA Object.....")
qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore.as_retriever(search_kwargs = {"k" : 2}),
    return_source_documents = True,
    chain_type_kwargs = chain_type_kwargs)
print("\n")

# print("Invoking Query")

# print(f"Your Query is {query}")
# result = qa.invoke({"query":query})
# print("Response ", result["result"])


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query" : input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(debug=True)

