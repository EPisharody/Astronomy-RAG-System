from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

FILE_PATH = "./data/general_astronomy_textbook.pdf"
DB_PATH = "./chroma"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to load documents from data folder
def load_documents():
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    return docs

# Function to split documents into smaller chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500, length_function=len, is_separator_regex=False)
    split_text = text_splitter.split_documents(docs)
    return split_text

def create_vector_db(split_docs):
    vector_db = Chroma.from_documents(split_docs, OpenAIEmbeddings(), persist_directory=DB_PATH)
    vector_db.persist()
    print(F"Saved {len(split_docs)} chunks to {DB_PATH}.")

split_text = split_documents(load_documents())
create_vector_db(split_text)