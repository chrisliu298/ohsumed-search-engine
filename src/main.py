import os
import warnings

from retrieval import (
    Indexer,
    read_documents,
    read_queries,
    search,
    search_with_sentence_embedding,
    search_with_sentence_embedding_prf,
)

from utils import format_results

warnings.filterwarnings("ignore")

METHODS = ["bool", "tf", "tfidf", "prf", "bm25"]


def run_trec_eval(
    trec_eval_path, qrel_path, search_result_path, evaluation_result_path
):
    command = "{} -q -c -M1000 {} {} > {}".format(
        trec_eval_path, qrel_path, search_result_path, evaluation_result_path
    )
    os.system(command)


if __name__ == "__main__":
    # os.chdir("..")
    print(os.getcwd())
    # fix_qrels_format("data/qrels.ohsu.88-91")
    # fix_qrels_format("data/qrels.ohsu.87")
    trec_path = "trec_eval/trec_eval"
    document_path = "data/ohsumed.88-91"
    query_path = "data/query.ohsu.1-63"
    relevance_path = "data/qrels.ohsu.88-91.fixed"

    documents = read_documents(document_path)
    queries = list(read_queries(query_path))

    index = Indexer(
        document_path, "embeddings/ohsumed.88-91_embeddings_all-mpnet-base-v2.pt"
    )
    index.build_index()
    index.fit_transform()
    for method in METHODS:
        search(index, queries, f"results/{method}.txt", method)
        run_trec_eval(
            trec_path,
            relevance_path,
            f"results/{method}.txt",
            f"results/{method}_eval.txt",
        )
    search_with_sentence_embedding(
        index, queries, "embeddings/ohsumed.88-91_similarities_all-mpnet-base-v2.pt"
    )
    run_trec_eval(
        trec_path,
        relevance_path,
        "results/se.txt",
        "results/se_eval.txt",
    )
    search_with_sentence_embedding_prf(index, queries)
    run_trec_eval(
        trec_path,
        relevance_path,
        "results/se-prf.txt",
        "results/se-prf_eval.txt",
    )
    format_results()
