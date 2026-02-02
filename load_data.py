# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib
import os
import shutil

FILE_PATH = "./data"
DB_PATH = "./chroma"

def main():
    build_chroma_db()

def build_chroma_db():
    split_text = split_documents(load_documents())
    update_vector_db(split_text)

def generate_id(content):
    return hashlib.md5(content.encode()).hexdigest()

def load_documents():
    loader = PyPDFDirectoryLoader(FILE_PATH)
    docs = loader.load()
    return docs

def split_documents(docs):
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
    split_docs = text_splitter.split_documents(docs)

    # Add "id" field to chunk metadata
    for doc in split_docs:
        doc.metadata["id"] = generate_id(doc.page_content)
    return split_docs

def get_embedding_function():
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    return embedding

def update_vector_db(split_docs):
    # Load existing db
    db = Chroma(persist_directory=DB_PATH, embedding_function=get_embedding_function())

    # Get existing db items and their corresponding ids
    existing_items = db.get(limit=None)
    existing_ids = existing_items["ids"]

    # Determine which docs are new (not already in the db) and add to new_docs
    new_docs = []
    for doc in split_docs:
        if (doc.metadata["id"]) not in existing_ids:
            new_docs.append(doc)
    
    # If there are new docs, generate an id and add them to the db
    if new_docs:
        new_ids = [doc.metadata["id"] for doc in new_docs]
        db.add_documents(new_docs, ids=new_ids)

        print(F"Saved {len(new_docs)} chunks to {DB_PATH}.")
    else:
        print("No new documents were saved to the database.")

def clear_database():
    if (os.path.exists(DB_PATH)):
        shutil.rmtree(DB_PATH)

if __name__ == "__main__":
    main()