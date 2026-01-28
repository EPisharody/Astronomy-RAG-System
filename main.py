import argparse
from langchain_community.vectorstores import Chroma
from load_data import get_embedding_function

DB_PATH = "./chroma"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_query')
    args = parser.parse_args()
    query = args.input_query
    print(args.input_query)

    # Load db
    db = Chroma(persist_directory=DB_PATH, embedding_function=get_embedding_function())

    # Find chunks that relate to query
    results = db.similarity_search_with_score(query, k=5)
    if (len(results) <= 0):
        print("Unable to find results.")

    # Results: [(Doc1, score1), (doc2, score2) ...]


    # response = model.invoke(query)

main()