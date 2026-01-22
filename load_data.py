from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "./data/general_astronomy_textbook.pdf"

# Function to load documents from data folder
def load_documents():
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    return docs

# Function to split documents into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500, length_function=len, is_separator_regex=False)
    split_text = text_splitter.split_documents(docs)
    return split_text

print(split_documents(load_documents())[5].page_content)