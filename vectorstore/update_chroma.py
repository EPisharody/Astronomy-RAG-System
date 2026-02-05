from langchain_chroma import Chroma
from embedding.huggingface_embedding import get_embedding_function
from preprocess.pipeline import ingest
from config import DB_PATH
import os
import shutil

def build_chroma_db():
    docs = ingest()
    update_vector_db(docs)

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