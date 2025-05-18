from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm.notebook import tqdm
import numpy as np

class HybridBench():
    
    def __init__(self,
                 dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 sparse_model_name = 'Qdrant/bm25',
                 device="cpu",
                 load_batch_size = 32,
                 m = 16,
                 ef_construct = 100,
                 full_scan_threshold = 10):
        """
        Initializes the Bench class.

        Args:
            model_name (str): Name of the model for generating embeddings. Defaults to "all-MiniLM-L6-v2".
            device (str): Device for computation. Defaults to "cuda".
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name

        self.dense_model = SentenceTransformer(self.dense_model_name, device=device)
        self.sparse_model = SparseTextEmbedding(self.sparse_model_name)
        self.load_batch_size = load_batch_size

        self.client = QdrantClient()
        self.dataset = pd.read_parquet(
            'hf://datasets/neural-bridge/rag-dataset-12000/data/train-00000-of-00001-9df3a936e1f63191.parquet'
        ).dropna().reset_index(drop=True)
        
        self.m = m
        self.full_scan_threshold = full_scan_threshold
        self.ef_construct = ef_construct

    def prepare_collection(self):
        """
        Prepares the Qdrant collection based on the type of model (sparse or dense).
        """

        if self.client.collection_exists(collection_name="test"):
            self.client.delete_collection(collection_name="test")

        self.client.create_collection(
            collection_name="test",
            vectors_config={
                self.dense_model_name: models.VectorParams(
                    size=self.dense_model.get_sentence_embedding_dimension(), 
                    distance=models.Distance.COSINE
                )
            },  
            sparse_vectors_config={
                    self.sparse_model_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
                } if self.sparse_model_name == 'Qdrant/bm25' else None,
        )


        for i in tqdm(range(len(self.dataset)//32 + 1)):
            row = self.dataset.iloc[i * self.load_batch_size : (1 + i) * self.load_batch_size]
            
            dense_embeddings = list(self.dense_model.encode(row["context"].values))
            bm25_embeddings = list(self.sparse_model.passage_embed(row["context"].values))
          
            self.client.upload_points(
                "test",
                points=[
                    models.PointStruct(
                        id=int(id_),
                        vector={
                            self.dense_model_name: dense_embeddings[i],
                            self.sparse_model_name: bm25_embeddings[i].as_object(),
                        },
                        payload={
                            "_id": i,
                            "text": row["context"][id_],
                        }
                    ) for i, id_ in enumerate(self.dataset.iloc[i * self.load_batch_size : (i +1) * self.load_batch_size].index)
                ],
                batch_size=self.load_batch_size,
            )
    def score_collection(self):
        """
        Evaluates the quality of the collection based on search results.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics.
        """
        dense_queries = list(self.dense_model.encode(self.dataset['question'], show_progress_bar=True,batch_size = self.load_batch_size))
        sparse_queries = [vec for vec in self.sparse_model.embed(self.dataset['question'], show_progress_bar=True, batch_size=self.load_batch_size)]

        
        search_results = []
        
        for query_idx in tqdm(range(len(dense_queries))):
            
            dense_query_vector = dense_queries[query_idx]
            sparse_query_vector = sparse_queries[query_idx]
            
            prefetch = [
                models.Prefetch(
                    query=dense_query_vector,
                    using=self.dense_model_name,
                    limit=100,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_query_vector.as_object()),
                    using=self.sparse_model_name,
                    limit=100,
                ),
            ]
            
            results = self.client.query_points(
                "test",
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                with_payload=False,
                limit = 100
                # query=late_query_vector,
                # using="colbertv2.0",
                # with_payload=False,
                # limit=10,
            ).points  
        
        
            search_results.append(results)
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
        metric_df.index = [self.dense_model_name.split("/")[-1] + "_" + self.sparse_model_name.split("/")[-1]]
        metric_df.to_csv(f"{metric_df.index[0]}_result.csv")
        return metric_df

if __name__ == "__main__":
    import sys

    bench = HybridBench(sys.argv[1])
    bench.prepare_collection()