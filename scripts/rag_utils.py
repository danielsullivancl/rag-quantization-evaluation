from sentence_transformers import SentenceTransformer, util
import torch

from config import (
    CHUNK_SIZE,
    TOP_K,
    EMBEDDING_MODEL
)

# ==================================================
# CARREGANDO MODELO DE EMBEDDINGS
# ==================================================

print("Carregando modelo de embeddings...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# ==================================================
# CHUNKING
# ==================================================

def chunk_text(text, chunk_size=CHUNK_SIZE):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks

# ==================================================
# RETRIEVAL
# ==================================================

def retrieve_top_k_chunks(question, document):

    # ----------------------------------------------
    # GERANDO CHUNKS
    # ----------------------------------------------

    chunks = chunk_text(document)

    # ----------------------------------------------
    # EMBEDDINGS DOS CHUNKS
    # ----------------------------------------------

    chunk_embeddings = embedding_model.encode(
        chunks,
        convert_to_tensor=True
    )

    # ----------------------------------------------
    # EMBEDDING DA PERGUNTA
    # ----------------------------------------------

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    # ----------------------------------------------
    # COSINE SIMILARITY
    # ----------------------------------------------

    scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    # ----------------------------------------------
    # TOP-K
    # ----------------------------------------------

    top_results = torch.topk(
        scores,
        k=min(TOP_K, len(chunks))
    )

    retrieved_chunks = []

    for score, idx in zip(
        top_results.values,
        top_results.indices
    ):

        retrieved_chunks.append({
            "chunk": chunks[idx],
            "similarity": float(score)
        })

    return retrieved_chunks

# ==================================================
# BUILD RAG PROMPT
# ==================================================

def build_rag_prompt(question, retrieved_chunks):

    context = "\n\n".join(
        [item["chunk"] for item in retrieved_chunks]
    )

    prompt = f"""
Answer the following technical support question using the provided documentation.

Documentation:
{context}

Question:
{question}
"""

    return prompt