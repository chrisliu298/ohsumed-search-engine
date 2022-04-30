import itertools

import numpy as np
import torch
from bs4 import BeautifulSoup
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from data import Document, Query
from text import tokenize
from utils import timefunc

field_map = {
    ".I": "seq_id",
    ".U": "medline_ui",
    ".S": "source",
    ".M": "mesh_terms",
    ".T": "title",
    ".P": "publication_type",
    ".W": "abstract",
    ".A": "author",
}


def read_documents(file_path):
    fields = list(field_map.keys())
    doc = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(".I "):
                if len(doc) > 0:
                    yield Document(**doc)
                    doc = {}
                doc[field_map[".I"]] = line.split(" ")[1].strip()
            else:
                if line.startswith("."):
                    line = line.strip()
                    if line in fields:
                        field = line
                    continue
                doc[field_map[field]] = line.strip()
        yield Document(**doc)


def read_queries(file_path):
    with open(file_path, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser").find_all("top")
        for q in soup:
            query = q.text.strip().split("\n")
            q = {
                "num": query[0].split(" ")[1].strip(),
                "title": query[1].strip(),
                "desc": query[3].strip(),
            }
            yield Query(**q)


class Indexer:
    def __init__(self, document_path, document_embeddings_path):
        self.document_path = document_path
        self.inverted_index = {}  # token -> document ids
        self.document_index = {}  # document id -> full document
        self.tf_vectorizer = TfidfVectorizer(stop_words="english", use_idf=False)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.document_embeddings = sparse.csr_matrix(
            torch.load(document_embeddings_path, map_location="cpu").numpy()
        )
        self.query_embeddings = sparse.csr_matrix(
            torch.load(
                "embeddings/query_embeddings_all-mpnet-base-v2.pt", map_location="cpu"
            ).numpy()
        )

    @timefunc
    def build_index(self):
        for document in tqdm(read_documents(self.document_path)):
            # Create a new document entry if it has not been added yet
            if document.medline_ui not in self.document_index:
                self.document_index[document.medline_ui] = document

                tokens = tokenize(document.full_text)
                for token in tokens:
                    # Create a new token entry if it does not been added yet
                    if token not in self.inverted_index:
                        self.inverted_index[token] = set()
                    # Add document id to the set (i.e., the posting list)
                    self.inverted_index[token].add(document.medline_ui)

            self.documents = list(self.document_index.values())

    @timefunc
    def fit_transform(self):
        documents = [document.full_text for _, document in self.document_index.items()]
        self.tf_vectorizer.fit(documents)
        self.tfidf_vectorizer.fit(documents)
        self.documents_tf_vectors = self.tf_vectorizer.transform(documents)
        self.documents_tfidf_vectors = self.tfidf_vectorizer.transform(documents)

    def boolean(self, query, top=50):
        tokenized_query = tokenize(query)
        results = [self.inverted_index.get(token, set()) for token in tokenized_query]
        results_sorted_by_len = sorted(results, key=len)
        results_iteratively_union = list(
            itertools.accumulate(results_sorted_by_len, set.intersection)
        )[-1]
        results = [
            {
                "score": np.random.rand(),
                "document": self.document_index[i],
                "rank": idx + 1,
            }
            for idx, i in enumerate(results_iteratively_union)
        ]
        return sorted(results, key=lambda x: x["score"])[:top]

    def tfidf(self, query, use_idf=True, top=50):
        if use_idf:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarity_scores = cosine_similarity(
                query_vector, self.documents_tfidf_vectors
            ).squeeze()
        else:
            query_vector = self.tf_vectorizer.transform([query])
            similarity_scores = cosine_similarity(
                query_vector, self.documents_tf_vectors
            ).squeeze()
        top_50_similar_idx = np.argsort(similarity_scores)[-top:][::-1]
        results = [
            {
                "score": similarity_scores[i],
                "document": self.documents[i],
                "rank": idx + 1,
            }
            for idx, i in enumerate(top_50_similar_idx)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def pseudo_relevance_feedback(
        self, query, alpha=1, beta=0.75, gamma=0.15, top_k=5, top=50
    ):
        # First round
        query_vector = self.tfidf_vectorizer.transform([query])
        similarity_scores = cosine_similarity(
            query_vector, self.documents_tfidf_vectors
        ).squeeze()
        top_k_similar_idx = np.argsort(similarity_scores)[-top_k:][::-1]
        top_k_similar_documents = sparse.vstack(
            [self.documents_tfidf_vectors[i] for i in top_k_similar_idx]
        )
        top_k_nonsimilar_idx = np.argsort(similarity_scores)[:top_k]
        top_k_nonsimilar_documents = sparse.vstack(
            [self.documents_tfidf_vectors[i] for i in top_k_nonsimilar_idx]
        )
        query_vector = (
            alpha * query_vector
            + beta * np.mean(top_k_similar_documents, axis=0)
            - gamma * np.mean(top_k_nonsimilar_documents, axis=0)
        )
        # Second round
        similarity_scores = cosine_similarity(
            query_vector, self.documents_tfidf_vectors
        ).squeeze()
        top_similar_idx = np.argsort(similarity_scores)[-top:][::-1]
        results = [
            {
                "score": similarity_scores[i],
                "document": self.documents[i],
                "rank": idx + 1,
            }
            for idx, i in enumerate(top_similar_idx)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def bm25(self, query, b=0.75, k1=1.2, top=50):
        documents = self.documents_tfidf_vectors
        avgdl = documents.sum(1).mean()
        document_len = documents.sum(1).A1
        query_vector = self.tfidf_vectorizer.transform([query])
        documents = documents.tocsc()[:, query_vector.indices]
        denominator = documents + (k1 * (1 - b + b * document_len / avgdl))[:, None]
        idf = self.tfidf_vectorizer.idf_[None, query_vector.indices] - 1.0
        numerator = documents.multiply(np.broadcast_to(idf, documents.shape)) * (k1 + 1)
        bm25_scores = (numerator / denominator).sum(1).A1
        top_k_similar_idx = np.argsort(bm25_scores)[-top:][::-1]
        results = [
            {
                "score": bm25_scores[i],
                "document": self.documents[i],
                "rank": idx + 1,
            }
            for idx, i in enumerate(top_k_similar_idx)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def sentence_embedding(self, embedding_path, top=50):
        similarities = torch.load(
            embedding_path, map_location=torch.device("cpu")
        ).numpy()
        results = []
        for similarity_scores in tqdm(
            similarities, desc="se", total=similarities.shape[0]
        ):
            top_similar_idx = np.argsort(similarity_scores)[-top:][::-1]
            top_results = [
                {
                    "score": similarity_scores[i],
                    "document": self.documents[i],
                    "rank": idx + 1,
                }
                for idx, i in enumerate(top_similar_idx)
            ]
            results.append(sorted(top_results, key=lambda x: x["score"], reverse=True))
        return results

    def sentence_embedding_prf(self, alpha=1, beta=0.75, gamma=0.15, top_k=5, top=50):
        results = []
        for query_embedding in tqdm(
            self.query_embeddings, desc="se-prf", total=self.query_embeddings.shape[0]
        ):
            # query_embedding = query_embedding.reshape(1, -1)
            similarity_scores = cosine_similarity(
                query_embedding, self.document_embeddings
            ).squeeze()
            top_k_similar_idx = np.argsort(similarity_scores)[-top_k:][::-1]
            top_k_similar_documents = sparse.vstack(
                [self.document_embeddings[i] for i in top_k_similar_idx]
            )
            top_k_nonsimilar_idx = np.argsort(similarity_scores)[:top_k]
            top_k_nonsimilar_documents = sparse.vstack(
                [self.document_embeddings[i] for i in top_k_nonsimilar_idx]
            )
            query_embedding = (
                alpha * query_embedding
                + beta * np.mean(top_k_similar_documents, axis=0)
                - gamma * np.mean(top_k_nonsimilar_documents, axis=0)
            )
            similarity_scores = cosine_similarity(
                query_embedding, self.document_embeddings
            ).squeeze()
            top_similar_idx = np.argsort(similarity_scores)[-top:][::-1]
            top_results = [
                {
                    "score": similarity_scores[i],
                    "document": self.documents[i],
                    "rank": idx + 1,
                }
                for idx, i in enumerate(top_similar_idx)
            ]
            results.append(sorted(top_results, key=lambda x: x["score"], reverse=True))
        return results


@timefunc
def search(index, queries, search_result_path, method="prf"):
    assert method in ["bool", "tf", "tfidf", "prf", "bm25"]
    with open(f"{search_result_path}", "w") as f:
        for query in tqdm(queries, desc=method):
            if method in ["tf", "tfidf"]:
                use_idf = True if method == "tfidf" else False
                results = index.tfidf(query.full_text, use_idf=use_idf)
            elif method == "prf":
                results = index.pseudo_relevance_feedback(query.full_text)
            elif method == "bm25":
                results = index.bm25(query.full_text)
            elif method == "bool":
                results = index.boolean(query.full_text)
            for result in results:
                f.write(
                    "{} {} {} {} {} {}\n".format(
                        query.num,
                        "Q0",
                        result["document"].medline_ui,
                        result["rank"],
                        result["score"],
                        method,
                    )
                )


@timefunc
def search_with_sentence_embedding(index, queries, embedding_path):
    with open(f"results/se.txt", "w") as f:
        sentence_embedding_results = index.sentence_embedding(embedding_path)
        for query, result in zip(queries, sentence_embedding_results):
            for r in result:
                f.write(
                    "{} {} {} {} {} {}\n".format(
                        query.num,
                        "Q0",
                        r["document"].medline_ui,
                        r["rank"],
                        r["score"],
                        "se",
                    )
                )


@timefunc
def search_with_sentence_embedding_prf(index, queries):
    with open(f"results/se-prf.txt", "w") as f:
        sentence_embedding_results = index.sentence_embedding_prf()
        for query, result in zip(queries, sentence_embedding_results):
            for r in result:
                f.write(
                    "{} {} {} {} {} {}\n".format(
                        query.num,
                        "Q0",
                        r["document"].medline_ui,
                        r["rank"],
                        r["score"],
                        "se-prf",
                    )
                )
