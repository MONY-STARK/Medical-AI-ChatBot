from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



def load_pdf(data_dir):
    loader = DirectoryLoader(data_dir,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500,
                                   chunk_overlap = 20)
    chunks = text_splitter.split_documents(documents=docs)
    return chunks


def download_huggingface_embedding():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding