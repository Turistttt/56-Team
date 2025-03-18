from glob import glob

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.notebook import tqdm


class Bench:
    def __init__(self,
                 model_name="all-MiniLM-L6-v2",
                 device="cuda"):
        self.model_name = model_name
        self.device = device
        self.client = QdrantClient()
        self.dataset = pd.read_parquet(
            'hf://datasets/neural-bridge/rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet').dropna().reset_index(
            drop=True)

        self.sparse_model_list = ['prithvida/Splade_PP_en_v1', "Qdrant/bm25"]
        self.dense_model_list = ["all-MiniLM-L6-v2", 'intfloat/multilingual-e5-small', 'all-mpnet-base-v2']

    def prepare_collection(self):

        if self.model_name in self.dense_model_list:
            self.prepare_dense_collection()
        elif self.model_name in self.sparse_model_list:
            self.prepare_sparse_collection()

    def prepare_dense_collection(self):
        self.model = SentenceTransformer(
            self.model_name, device=self.device,
        )

        if '/' in self.model_name:
            self.model_name = self.model_name.split('/')[-1]

        if f"{self.model_name}_vectors.npy" not in glob("*"):
            vectors = self.model.encode(
                self.dataset['context'],
                show_progress_bar=True,
            )
            np.save(f"{self.model_name}_vectors.npy", vectors, allow_pickle=False)

        vectors = np.load(f"{self.model_name}_vectors.npy")

        self.client.delete_collection(collection_name="test")

        self.client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(),
                                        distance=Distance.COSINE),
        )

        self.client.upload_collection(
            collection_name="test",
            vectors=vectors,
            payload=[{"context": i[0], "question": i[1], "answer": i[2], } for i in self.dataset.values],
            ids=[i for i in range(len(self.dataset))],  # Vector ids will be assigned automatically
            batch_size=256,  # How many vectors will be uploaded in a single request?
        )

    def prepare_sparse_collection(self):
        self.model = SparseTextEmbedding(model_name=self.model_name)

        if '/' in self.model_name:
            self.model_name = self.model_name.split('/')[-1]

        if f"{self.model_name}_vectors.npy" not in glob("*"):
            vectors = [vec for vec in tqdm(self.model.embed(
                self.dataset['context'],
                show_progress_bar=True,
                batch_size=128,
            ))]
            np.save(f"{self.model_name}_vectors.npy", np.array(vectors), allow_pickle=True)

        vectors = np.load(f"{self.model_name}_vectors.npy", allow_pickle=True)

        # if not self.client.collection_exists(collection_name="test"):
        self.client.delete_collection(collection_name="test")

        self.client.create_collection(
            collection_name="test",
            vectors_config={},
            # comment this line to use dense vectors only
            sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
        )

        self.client.upload_collection(
            collection_name="test",
            vectors=[],
            sparse_vectors=vectors,
            payload=[{"context": i[0], "question": i[1], "answer": i[2], } for i in self.dataset.values],
            ids=[i for i in range(len(self.dataset))],  # Vector ids will be assigned automatically
            batch_size=256,  # How many vectors will be uploaded in a single request?
        )

    def score_collection(self):
        queries = self.model.encode(
            self.dataset['question'],
            show_progress_bar=True,
        )

        search_result = [self.client.query_points(
            collection_name="test",
            query=query,
            with_payload=False,
            limit=10000
        ).points for query in tqdm(queries)]

        ids = [list(map(lambda x: x.id, result)) for result in search_result]
        biases = []

        for vector in tqdm(range(len(ids))):
            bias = ids[vector].index(vector)

            biases.append(bias)

            metrics_dict = {}

        metrics_dict['recall@10'] = np.mean(np.array(biases) <= 10)
        metrics_dict['recall@20'] = np.mean(np.array(biases) <= 20)
        metrics_dict['recall@30'] = np.mean(np.array(biases) <= 30)
        metrics_dict['recall@50'] = np.mean(np.array(biases) <= 50)
        metrics_dict['recall@100'] = np.mean(np.array(biases) <= 100)

        return metrics_dict

# bench = Bench("all-MiniLM-L6-v2")
# bench.prepare_collection()
# print(bench.score_collection())