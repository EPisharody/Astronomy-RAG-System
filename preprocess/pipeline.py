from ingest.load_data import load_documents
from preprocess.chunk_data import split_documents

def ingest():
    docs = load_documents()
    docs = split_documents(docs)
    return docs