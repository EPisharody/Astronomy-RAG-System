from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import hashlib

def generate_id(content):
    return hashlib.md5(content.encode()).hexdigest()

def split_documents(docs):
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, is_separator_regex=False)
    split_docs = text_splitter.split_documents(docs)

    # Add "id" field to chunk metadata
    for doc in split_docs:
        doc.metadata["id"] = generate_id(doc.page_content)
    return split_docs