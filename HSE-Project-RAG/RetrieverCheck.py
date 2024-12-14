from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd


class ContextRetriever:
    """
    Класс создан для подсчета метрик для Embedder'а, в данный момент реализовано 2 метрики recall и precision

    """
    def __init__(self, encoder_model):
        """
        Инициализация класса с моделью энкодера.
        :param encoder_model: модель энкодера (SentenceTransformer)
        """
        self.model =  SentenceTransformer(encoder_model)


    def compute_embeddings(self, texts):
        """
        Вычисляет эмбеддинги для списка текстов.
        :param texts: список строк
        :return: матрица эмбеддингов
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def find_relevant_context(self, query_embeddings, context_embeddings, top_k=5):
        """
        Находит релевантный контекст для заданных запросов.
        :param query_embeddings: эмбеддинги запросов
        :param context_embeddings: эмбеддинги контекстов
        :param top_k: количество релевантных результатов
        :return: индексы топ-k релевантных контекстов
        """
        scores = util.cos_sim(query_embeddings, context_embeddings)
        top_results = np.argpartition(-scores.cpu(), range(top_k))[:, :top_k]
        return top_results

    def evaluate_metrics(self, queries, contexts, true_relevant, top_k=5):
        """
        Оценивает метрики поиска релевантного контекста.
        :param queries: список запросов
        :param contexts: список контекстов
        :param true_relevant: список списков индексов истинно релевантных контекстов для каждого запроса
        :param top_k: количество топовых результатов для оценки
        :return: метрики (например, precision, recall)
        """
        query_embeddings = self.compute_embeddings(queries)
        context_embeddings = self.compute_embeddings(contexts)

        top_results = self.find_relevant_context(query_embeddings, context_embeddings, top_k=top_k)

        precision_list = []
        recall_list = []

        for i, top_indices in enumerate(top_results):
            relevant_set = set(true_relevant[i])
            retrieved_set = set(top_indices.numpy())
            true_positives = len(relevant_set & retrieved_set)
            precision = true_positives / len(retrieved_set) if retrieved_set else 0
            recall = true_positives / len(relevant_set) if relevant_set else 0
            precision_list.append(precision)
            recall_list.append(recall)

        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        return {'precision': avg_precision, 'recall': avg_recall}



if __name__ == '__main__':
    # Изначальный датасет
    c = pd.read_csv("dataset.csv")
    encoders_list = ['all-MiniLM-L6-v2',"Qdrant/bm25"]

    retriever = ContextRetriever('all-MiniLM-L6-v2')

    metrics = retriever.evaluate_metrics(c['question'], c['context'], c.index.values.reshape(-1, 1),
                                         top_k=3)
    print(pd.DataFrame({"retriever":encoders_list,
                        "metrics":metrics
                        }))