import argparse
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import load_data
from transformers import logging
import logging as py_logging

# Set logging levels
logging.set_verbosity_error()
py_logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

DB_PATH = "./chroma"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_query')
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    # Check if --clear flag was included in CLI arguments and clear db accordingly
    if (args.clear):
        print("Clearing database...")
        load_data.clear_database()
    query = args.input_query

    # Load db
    load_data.build_chroma_db()
    embedding_function = load_data.get_embedding_function()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # Find chunks that relate to query
    results = db.similarity_search_with_score(query, k=3)
    if (len(results) <= 0):
        print("Unable to find results.")

    # Results are in the form: [(Doc1, score1), (doc2, score2) ...]
    context = "\n----------\n".join([doc.page_content for doc, score in results])
    prompt = f"Use only the following context to answer the question: \n\n{context}\n\nAnswer this question based on the above context: {query}\n"

    model = ChatOllama(model="mistral")
    print(prompt)

    response = model.invoke(prompt)
    formatted_response = f"Response: {response.content}"
    print(formatted_response)

if __name__ == "__main__":
    main()