import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from utils import read_documents, read_queries

model_name = "all-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name).to(device)

document_file = "cse272-hw1-data/ohsumed.88-91"
query_file = "cse272-hw1-data/query.ohsu.1-63"

documents = [document.full_text for document in read_documents(document_file)]
queries = [query.full_text for query in read_queries(query_file)]
document_embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_tensor=True,
    device=device,
)
query_embeddings = model.encode(
    queries,
    show_progress_bar=True,
    convert_to_tensor=True,
    device=device,
)
similarities = cos_sim(query_embeddings, document_embeddings)

name = document_file.split("/")[-1]
torch.save(document_embeddings.cpu(), f"{name}_embeddings_{model_name}.pt")
torch.save(query_embeddings.cpu(), f"query_embeddings_{model_name}.pt")
torch.save(similarities, f"{name}_similarities_{model_name}.pt")
