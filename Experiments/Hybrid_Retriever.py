from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import numpy as np
from datasets import Dataset

df = pd.read_csv("...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus_embeddings = model.encode(df["Model_Answer"].tolist(), convert_to_tensor=True)

def retrieve_similar_contexts(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [(df.iloc[idx]["Model_Answer"], cos_scores[idx].item()) for idx in top_results[1]]

sample_query = "What causes inflation?"
results = retrieve_similar_contexts(sample_query, top_k=3)

for idx, (text, score) in enumerate(results):
    print(f"Result {idx + 1}:")
    print(f"Score: {score:.4f}")
    print(f"Text: {text}\n")
