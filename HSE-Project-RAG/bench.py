import numpy as np
import pandas as pd
from glob import glob
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer


class Bench:
    """
    A class for working with text embeddings and storing them in Qdrant.

    Attributes:
        model_name (str): Name of the model used for generating embeddings.
        device (str): Device for computation (e.g., 'cuda' or 'cpu').
        client (QdrantClient): Client for interacting with Qdrant.
        dataset (pd.DataFrame): Dataset containing texts and metadata.
        sparse_model_list (list): List of models for sparse embeddings.
        dense_model_list (list): List of models for dense embeddings.
        sparse (bool): Flag indicating whether sparse embeddings are used.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        """
        Initializes the Bench class.

        Args:
            model_name (str): Name of the model for generating embeddings. Defaults to "all-MiniLM-L6-v2".
            device (str): Device for computation. Defaults to "cuda".
        """
        self.model_name = model_name
        self.device = device
        self.client = QdrantClient()
        self.dataset = pd.read_parquet(
            'hf://datasets/neural-bridge/rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
        ).dropna().reset_index(drop=True)

        self.sparse_model_list = ['prithvida/Splade_PP_en_v1', "Qdrant/bm25"]
        self.dense_model_list = ["all-MiniLM-L6-v2", 'intfloat/multilingual-e5-small', 'all-mpnet-base-v2']
        self.sparse = False

    def prepare_collection(self):
        """
        Prepares the Qdrant collection based on the type of model (sparse or dense).
        """
        if self.model_name in self.dense_model_list:
            self.prepare_dense_collection()
        elif self.model_name in self.sparse_model_list:
            self.sparse = True
            self.prepare_sparse_collection()

    def prepare_dense_collection(self):
        """
        Prepares a collection with dense embeddings.
        """
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if '/' in self.model_name:
            self.model_name = self.model_name.split('/')[-1]

        if f"{self.model_name}_vectors.npy" not in glob("*"):
            vectors = self.model.encode(self.dataset['context'], show_progress_bar=True)
            np.save(f"{self.model_name}_vectors.npy", vectors, allow_pickle=False)
        else:
            vectors = np.load(f"{self.model_name}_vectors.npy")

        if self.client.collection_exists(collection_name="test"):
            self.client.delete_collection(collection_name="test")

        self.client.create_collection(
            collection_name="test",
            vectors_config={
                self.model_name: VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                )
            }
        )

        self.client.upload_points(
            'test',
            points=[
                models.PointStruct(
                    id=i,
                    vector={self.model_name: vector.tolist()},
                    payload={"context": row[0], "question": row[1], "answer": row[2]}
                ) for i, (vector, row) in enumerate(zip(vectors, self.dataset.values))
            ]
        )

    def prepare_sparse_collection(self):
        """
        Prepares a collection with sparse embeddings.
        """
        self.model = SparseTextEmbedding(model_name=self.model_name)

        if '/' in self.model_name:
            self.model_name = self.model_name.split('/')[-1]

        if f"{self.model_name}_vectors.npy" not in glob("*"):
            vectors = [vec for vec in self.model.embed(self.dataset['context'], show_progress_bar=True, batch_size=128)]
            np.save(f"{self.model_name}_vectors.npy", np.array(vectors), allow_pickle=True)
        else:
            vectors = np.load(f"{self.model_name}_vectors.npy", allow_pickle=True)

        if self.client.collection_exists(collection_name="test"):
            self.client.delete_collection(collection_name="test")

        self.client.create_collection(
            collection_name="test",
            vectors_config={},
            sparse_vectors_config={
                self.model_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )

        self.client.upload_points(
            'test',
            points=[
                models.PointStruct(
                    id=i,
                    vector={self.model_name: vector.as_object()},
                    payload={"context": row[0], "question": row[1], "answer": row[2]}
                ) for i, (vector, row) in enumerate(zip(vectors, self.dataset.values))
            ]
        )

    def get_dense_scores(self):
        """
        Retrieves scores for dense embeddings.

        Returns:
            list: A list of search results for each query.
        """
        queries = self.model.encode(self.dataset['question'], show_progress_bar=True)

        search_results = [
            self.client.query_points(
                collection_name="test",
                query=query,
                using=self.model_name,
                with_payload=False,
                limit=100
            ).points for query in queries
        ]

        return search_results

    def get_sparse_scores(self):
        """
        Retrieves scores for sparse embeddings.

        Returns:
            list: A list of search results for each query.
        """
        queries = [vec for vec in self.model.embed(self.dataset['question'], show_progress_bar=True, batch_size=128)]

        search_results = [
            self.client.query_points(
                collection_name="test",
                query=models.SparseVector(**query.as_object()),
                using=self.model_name,
                with_payload=False,
                limit=100
            ).points for query in queries
        ]

        return search_results

    def score_collection(self):
        """
        Evaluates the quality of the collection based on search results.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics.
        """
        if self.sparse:
            search_results = self.get_sparse_scores()
        else:
            search_results = self.get_dense_scores()

        ids = [list(map(lambda x: x.id, result)) for result in search_results]

        biases = []
        for vector in range(len(ids)):
            if vector in ids[vector]:
                bias = ids[vector].index(vector)
                biases.append(bias)
            else:
                biases.append(101)

        metrics_dict = {
            'recall@10': np.mean(np.array(biases) <= 10),
            'recall@20': np.mean(np.array(biases) <= 20),
            'recall@30': np.mean(np.array(biases) <= 30),
            'recall@50': np.mean(np.array(biases) <= 50),
            'recall@100': np.mean(np.array(biases) <= 100)
        }

        metric_df = pd.DataFrame([metrics_dict])
        metric_df.index = [self.model_name]
        metric_df.to_csv(f"{self.model_name}_result.csv")
        return metric_df


if __name__ == "__main__":
    import sys

    bench = Bench(sys.argv[1])
    bench.prepare_collection()