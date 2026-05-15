from datasets import load_dataset
import statistics
import math

print("Carregando TechQA...")

dataset = load_dataset("rojagtap/tech-qa")

sizes = []

# ==================================================
# MEDINDO TAMANHO DOS DOCUMENTOS
# ==================================================

for example in dataset["train"]:

    context = example["document"]

    word_count = len(context.split())

    sizes.append(word_count)

# ==================================================
# ESTATÍSTICAS
# ==================================================

mean_words = statistics.mean(sizes)
max_words = max(sizes)
median_words = statistics.median(sizes)
std_words = statistics.stdev(sizes)

print("\n===================================")
print("DOCUMENT SIZE STATISTICS")
print("===================================")

print(f"Total documents: {len(sizes)}")
print(f"Minimum words: {min(sizes)}")
print(f"Maximum words: {max_words}")
print(f"Mean words: {mean_words:.2f}")
print(f"Median words: {median_words:.2f}")
print(f"Standard deviation: {std_words:.2f}")

# ==================================================
# ESTIMATIVA DE TOKENS
# ==================================================

# Aproximação:
# 1 palavra ≈ 1.3 tokens

mean_tokens = mean_words * 1.3
max_tokens = max_words * 1.3

print("\n===================================")
print("TOKEN ESTIMATION")
print("===================================")

print(f"Estimated mean tokens: {mean_tokens:.0f}")
print(f"Estimated max tokens: {max_tokens:.0f}")

# ==================================================
# ESTIMATIVA DE NUM_CTX
# ==================================================

# Reserva:
# - pergunta
# - instruções do prompt
# - resposta do modelo
# - margem de segurança

SAFETY_MARGIN = 400

recommended_ctx = max_tokens + SAFETY_MARGIN

# Arredondando para potência de 2 mais próxima
# comum em context windows

possible_ctx = [512, 1024, 2048, 4096, 8192]

ideal_ctx = None

for ctx in possible_ctx:
    if ctx >= recommended_ctx:
        ideal_ctx = ctx
        break

print("\n===================================")
print("RECOMMENDED NUM_CTX")
print("===================================")

print(f"Recommended minimum context size: {recommended_ctx:.0f} tokens")

print(f"Ideal num_ctx for experiments: {ideal_ctx}")

# ==================================================
# MAIORES DOCUMENTOS
# ==================================================

sorted_sizes = sorted(sizes, reverse=True)

print("\n===================================")
print("TOP 10 LARGEST DOCUMENTS")
print("===================================")

for i, size in enumerate(sorted_sizes[:10]):
    print(f"{i+1}. {size} words")