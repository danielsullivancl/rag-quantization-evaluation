from datasets import load_dataset
import ollama
import time

print("Carregando TechQA...")

dataset = load_dataset("rojagtap/tech-qa")

example = dataset["train"][0]

question = example["question"]

context = example["document"]

word_count = len(context.split())

print("\nDOCUMENT SIZE")
print(word_count, "words")

expected = example["answer"]

# ==================================================
# SEM RAG
# ==================================================

start = time.time()

response_no_rag = ollama.chat(
    model="llama3.1:8b-instruct-q4_K_M",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ],
    options={
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 200
    }
)

latency_no_rag = time.time() - start

answer_no_rag = response_no_rag["message"]["content"]

# ==================================================
# COM RAG
# ==================================================

rag_prompt = f"""
Use the technical documentation below to answer the question.

Documentation:
{context}

Question:
{question}
"""

start = time.time()

response_rag = ollama.chat(
    model="llama3.1:8b-instruct-q4_K_M",
    messages=[
        {
            "role": "user",
            "content": rag_prompt
        }
    ],
    options={
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 200
    }
)

latency_rag = time.time() - start

answer_rag = response_rag["message"]["content"]

# ==================================================
# RESULTADOS
# ==================================================

print("\n==================================================")
print("QUESTION")
print("==================================================")
print(question)

print("\n==================================================")
print("WITHOUT RAG")
print("==================================================")
print(answer_no_rag)

print("\n==================================================")
print("WITH RAG")
print("==================================================")
print(answer_rag)

print("\n==================================================")
print("EXPECTED")
print("==================================================")
print(expected)

print("\n==================================================")
print("LATENCY")
print("==================================================")
print("Without RAG:", latency_no_rag)
print("With RAG:", latency_rag)