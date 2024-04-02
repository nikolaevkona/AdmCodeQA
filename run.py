"""Administrative fine QA chat example"""

import argparse
from src import AdminData, ChatClient

def parse_args():
    parser = argparse.ArgumentParser(
        prog="QA chat",
        description="Runs simple chat")
    parser.add_argument("--code-path", type=str,
                        help="Path to txt file with administrative code")
    parser.add_argument("--save-path", type=str,
                         help="Path to save vector database")
    parser.add_argument("--vector-model-name", type=str,
                        default="distiluse-base-multilingual-cased-v1",
                        help="Sentence-transformers model for vector database")
    parser.add_argument("--api-key", type=str, help="Open AI api keys")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top k for retrieval")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    vector_store = AdminData(args.code_path, args.save_path,
                             model_name=args.vector_model_name)
    client = ChatClient(args.api_key, vector_store, args.top_k)
    while True:
        print("Штраф для какого нарушения вас интересует?")
        query = input().rstrip("\n")
        respond = client.respond(query)
        print(respond)
