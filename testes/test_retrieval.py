from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

print("Carregando TechQA...")

dataset = load_dataset("rojagtap/tech-qa")

# ==================================================
# CONFIG
# ==================================================

CHUNK_SIZE = 300

TOP_K = 3

# ==================================================
# FUNÇÃO DE CHUNKING
# ==================================================

def chunk_text(text, chunk_size=300):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks

# ==================================================
# PEGANDO UM EXEMPLO
# ==================================================

example = dataset["train"][0]

question = example["question"]

document = example["document"]

expected = example["answer"]

# ==================================================
# GERANDO CHUNKS
# ==================================================

chunks = chunk_text(document, CHUNK_SIZE)

print("\n===================================")
print("QUESTION")
print("===================================")

print(question)

print("\n===================================")
print("TOTAL CHUNKS")
print("===================================")

print(len(chunks))

# ==================================================
# CARREGANDO MODELO DE EMBEDDING
# ==================================================

print("\nCarregando modelo de embeddings...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# EMBEDDINGS DOS CHUNKS
# ==================================================

chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True
)

# ==================================================
# EMBEDDING DA PERGUNTA
# ==================================================

question_embedding = model.encode(
    question,
    convert_to_tensor=True
)

# ==================================================
# COSINE SIMILARITY
# ==================================================

scores = util.cos_sim(
    question_embedding,
    chunk_embeddings
)[0]

# ==================================================
# TOP K CHUNKS
# ==================================================

top_results = torch.topk(scores, k=min(TOP_K, len(chunks)))

print("\n===================================")
print("TOP RETRIEVED CHUNKS")
print("===================================")

for rank, (score, idx) in enumerate(
    zip(top_results.values, top_results.indices)
):

    print(f"\n----- TOP {rank+1} -----")

    print(f"Similarity: {score:.4f}")

    print("\nChunk:\n")

    print(chunks[idx][:1000])