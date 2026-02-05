from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredFileLoader
)

from config import FILE_PATH

def load_documents():
    docs = []

    # Load pdf files
    pdfLoader = DirectoryLoader(FILE_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdfs = pdfLoader.load()
    docs.extend(pdfs)

    # Load text files
    txtLoader = DirectoryLoader(FILE_PATH, glob="**/*.txt", loader_cls=TextLoader)
    textFiles = txtLoader.load()
    docs.extend(textFiles)

    # # Load html files
    # htmlLoader = DirectoryLoader(FILE_PATH, glob="**/*.html", loader_cls=UnstructuredHTMLLoader)
    # htmlFiles = htmlLoader.load()
    # docs.extend(htmlFiles)

    # # Fallback loader
    # unstructuredFileLoader = DirectoryLoader(FILE_PATH, glob="**/*", loader_cls=UnstructuredFileLoader)
    # unstructuredFiles = unstructuredFileLoader.load()
    # docs.extend(unstructuredFiles)

    return docs