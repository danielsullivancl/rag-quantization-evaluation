from datasets import load_dataset
import statistics

print("Carregando TechQA...")

dataset = load_dataset("rojagtap/tech-qa")

# ==================================================
# CONFIG
# ==================================================

CHUNK_SIZE = 300

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
# ESTATÍSTICAS
# ==================================================

num_chunks_per_doc = []

chunk_lengths = []

# ==================================================
# PROCESSANDO DATASET
# ==================================================

for example in dataset["train"]:

    document = example["document"]

    chunks = chunk_text(document, CHUNK_SIZE)

    num_chunks_per_doc.append(len(chunks))

    for chunk in chunks:

        chunk_lengths.append(len(chunk.split()))

# ==================================================
# RESULTADOS
# ==================================================

print("\n===================================")
print("CHUNKING ANALYSIS")
print("===================================")

print(f"Chunk size used: {CHUNK_SIZE} words")

print(f"\nTotal documents: {len(num_chunks_per_doc)}")

print(f"Total chunks generated: {sum(num_chunks_per_doc)}")

print("\n-----------------------------------")
print("CHUNKS PER DOCUMENT")
print("-----------------------------------")

print(f"Min: {min(num_chunks_per_doc)}")
print(f"Max: {max(num_chunks_per_doc)}")
print(f"Mean: {statistics.mean(num_chunks_per_doc):.2f}")
print(f"Median: {statistics.median(num_chunks_per_doc):.2f}")

print("\n-----------------------------------")
print("CHUNK LENGTHS")
print("-----------------------------------")

print(f"Min: {min(chunk_lengths)} words")
print(f"Max: {max(chunk_lengths)} words")
print(f"Mean: {statistics.mean(chunk_lengths):.2f} words")
print(f"Median: {statistics.median(chunk_lengths):.2f} words")