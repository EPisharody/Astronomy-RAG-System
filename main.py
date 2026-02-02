import argparse
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from load_data import get_embedding_function

DB_PATH = "./chroma"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_query')
    args = parser.parse_args()
    query = args.input_query

    # Load db
    db = Chroma(persist_directory=DB_PATH, embedding_function=get_embedding_function())

    # Find chunks that relate to query
    results = db.similarity_search_with_score(query, k=3)
    if (len(results) <= 0):
        print("Unable to find results.")

    # Results are in the form: [(Doc1, score1), (doc2, score2) ...]
    context = "\n----------\n".join([doc.page_content for doc, score in results])
    prompt = f"""Use only the following context to answer the question:
    
    {context}
    
    Question: {query}
    """

    model = ChatOllama(model="mistral")

    response = model.invoke(prompt)
    formatted_response = f"Response: {response.content}"
    print(formatted_response)

if __name__ == "__main__":
    main()