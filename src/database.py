"""Module for information retrieval from administrative code of Russia."""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

class AdminData:
    """Administrative code vector database.

    Args:
        data_path (str): Path to txt file with administrative code taken from
            ConsultantPlus.
        save_path (str): Path to store or load database.
        database_name (str): Database collection name.
            Defaults to ru_code.
        model_name (str): Name of sentence-transformers vector model.
            Defaults to distiluse-base-multilingual-cased-v1.
    """

    def __init__(self, data_path: str, save_path: str,
                 database_name: str = "ru_code",
                 model_name: str = "distiluse-base-multilingual-cased-v1"
                 ) -> None:
        paragraphs = self._parse_doc(data_path)
        self._build_database(paragraphs, save_path, database_name, model_name)

    def retrieve(self, query: str, top_k: int) -> str:
        """Retrieve top_k articles

        Args:
            query (str): Query text.
            top_k (int): Number of articles to return.

        Returns:
            respond: Text of selected articles.
        """
        query_results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        context = list()
        for meta, title in zip(query_results["metadatas"][0],
                                  query_results["documents"][0]):
            document = meta["text"].replace("\n\n", "\n").replace("\n\n", "\n")
            context.extend([title, document, "\n"])
        return "".join(context)

    def _build_database(self, paragraphs: List[Dict], save_path: str,
                        database_name: str, model_name: str) -> None:
        client = chromadb.PersistentClient(path=save_path)
        embedding_func = \
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        self._collection = client.get_or_create_collection(
            name=database_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"},
        )
        if self._collection.count() == 0:
            self._collection.add(
                documents=[element["title"] for element in paragraphs],
                ids=[f"id{i}" for i in range(len(paragraphs))],
                metadatas=[{"text": element["text"]} for element in paragraphs]
            )

    def _parse_doc(self, data_path: str) -> List[Dict]:
        paragraphs = list()
        with open(data_path, "r", encoding="utf-8") as f:
            article = {"text": []}
            for line in f.readlines():
                if line.startswith("Статья"):
                    if "title" in article:
                        article["text"] = "\n".join(article["text"])
                        article["number"] = \
                            " ".join(article["text"].split()[:2]).rstrip(".")
                        if "штраф" in article["text"]:
                            paragraphs.append(article)
                    article = {"title": line, "text": []}
                elif "title" in article:
                    article["text"].append(line)
        return paragraphs
