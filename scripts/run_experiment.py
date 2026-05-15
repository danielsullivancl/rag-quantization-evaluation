import os
import time
from datetime import datetime

import pandas as pd
from datasets import load_dataset

from config import (
    MODELS,
    DATASET_NAME,
    NUM_SAMPLES,
    OUTPUT_CSV,
    PAUSE_BETWEEN_RUNS_S
)

from rag_utils import (
    retrieve_top_k_chunks,
    build_rag_prompt
)

from ollama_utils import (
    call_ollama,
    extract_response_text,
    extract_token_metrics,
    calculate_throughput_tokens_s
)

from monitoring import (
    start_monitoring,
    stop_monitoring
)


# ==================================================
# SAVE RESULT
# ==================================================

def save_result(result, output_csv):
    output_dir = os.path.dirname(output_csv)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([result])

    file_exists = os.path.exists(output_csv)

    df.to_csv(
        output_csv,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )


# ==================================================
# NORMALIZE BOOL
# ==================================================

def normalize_bool(value):
    return str(value).strip().lower()


# ==================================================
# LOAD COMPLETED EXECUTIONS
# ==================================================

def load_completed_executions(output_csv):
    if not os.path.exists(output_csv):
        return set()

    df = pd.read_csv(output_csv)

    completed = set()

    for _, row in df.iterrows():

        if "success" in df.columns:
            if str(row.get("success")).strip().lower() != "true":
                continue

        # Preferir sample_index se existir.
        # Isso evita problema com id textual, NaN ou mudança no dataset_id.
        if "sample_index" in df.columns:
            sample_key = str(int(row["sample_index"]))
        else:
            sample_key = str(row["id"])

        completed.add(
            (
                sample_key,
                str(row["quantization"]),
                normalize_bool(row["rag"])
            )
        )

    return completed


# ==================================================
# BUILD NO-RAG PROMPT
# ==================================================

def build_no_rag_prompt(question):
    return f"""
Answer the following technical support question.

Question:
{question}
"""


# ==================================================
# SAFE STOP MONITORING
# ==================================================

def safe_stop_monitoring(monitoring_data):
    try:
        if monitoring_data is not None:
            return stop_monitoring(**monitoring_data)
    except Exception:
        pass

    return {
        "peak_ram_mb": None,
        "ram_usage_mb": None,
        "peak_vram_mb": None,
        "vram_delta_mb": None
    }


# ==================================================
# DATASET
# ==================================================

print("Carregando dataset...")

dataset = load_dataset(DATASET_NAME)

samples = dataset["train"].select(
    range(NUM_SAMPLES)
)

print(f"Total de samples: {len(samples)}")


# ==================================================
# LOAD RESUME STATE
# ==================================================

completed_executions = load_completed_executions(
    OUTPUT_CSV
)

print("\n===================================")
print("COMPLETED EXECUTIONS")
print("===================================")
print(len(completed_executions))


# ==================================================
# LOOP PRINCIPAL
# ==================================================

for model_data in MODELS:

    model_name = model_data["model_name"]
    quantization = model_data["quantization"]

    print("\n===================================")
    print(f"MODEL: {model_name}")
    print("===================================")

    for i, example in enumerate(samples):

        sample_index = i
        dataset_id = example.get("id", "")

        question = example["question"]
        document = example["document"]
        expected = example["answer"]

        experiment_configs = [
            {
                "rag": False,
                "label": "WITHOUT RAG"
            },
            {
                "rag": True,
                "label": "WITH RAG"
            }
        ]

        for exp_config in experiment_configs:

            rag_enabled = exp_config["rag"]
            label = exp_config["label"]

            execution_key = (
                str(sample_index),
                str(quantization),
                normalize_bool(rag_enabled)
            )

            # ==============================================
            # SKIP EXECUÇÕES JÁ FEITAS
            # ==============================================

            if execution_key in completed_executions:
                print(
                    f"[SKIP] "
                    f"sample_index={sample_index} | "
                    f"dataset_id={dataset_id} | "
                    f"{quantization} | "
                    f"RAG={rag_enabled}"
                )
                continue

            print("\n-----------------------------------")
            print(
                f"[{i + 1}/{len(samples)}] "
                f"{label} | "
                f"{quantization}"
            )
            print("-----------------------------------")

            monitoring_data = None
            retrieved_chunks_text = ""
            retrieval_time_s = 0.0
            prompt_build_time_s = 0.0
            ollama_latency_s = None
            end_to_end_latency_s = None

            try:
                # ==========================================
                # END-TO-END START
                # Inclui retrieval + prompt build + Ollama
                # ==========================================

                end_to_end_start = time.perf_counter()

                # ==========================================
                # PROMPT / RETRIEVAL
                # ==========================================

                if rag_enabled:

                    retrieval_start = time.perf_counter()

                    retrieved_chunks = retrieve_top_k_chunks(
                        question=question,
                        document=document
                    )

                    retrieval_time_s = (
                        time.perf_counter() - retrieval_start
                    )

                    prompt_build_start = time.perf_counter()

                    prompt = build_rag_prompt(
                        question=question,
                        retrieved_chunks=retrieved_chunks
                    )

                    prompt_build_time_s = (
                        time.perf_counter() - prompt_build_start
                    )

                    retrieved_chunks_text = "\n\n".join(
                        [
                            chunk["chunk"][:500]
                            for chunk in retrieved_chunks
                        ]
                    )

                else:

                    prompt_build_start = time.perf_counter()

                    prompt = build_no_rag_prompt(question)

                    prompt_build_time_s = (
                        time.perf_counter() - prompt_build_start
                    )

                # ==========================================
                # MONITORAMENTO
                # Começa antes do Ollama.
                # Não inclui retrieval, mas inclui inferência.
                # ==========================================

                monitoring_data = start_monitoring()

                ollama_start = time.perf_counter()

                response = call_ollama(
                    model_name=model_name,
                    prompt=prompt
                )

                ollama_latency_s = (
                    time.perf_counter() - ollama_start
                )

                monitoring_results = stop_monitoring(
                    **monitoring_data
                )

                end_to_end_latency_s = (
                    time.perf_counter() - end_to_end_start
                )

                # ==========================================
                # RESPOSTA
                # ==========================================

                response_text = extract_response_text(
                    response
                )

                token_metrics = extract_token_metrics(
                    response
                )

                throughput_tokens_s = (
                    calculate_throughput_tokens_s(
                        token_metrics["output_token_count"],
                        token_metrics["generation_time_ns"]
                    )
                )

                # ==========================================
                # RESULTADO SUCESSO
                # ==========================================

                result = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),

                    "sample_index": sample_index,
                    "dataset_id": dataset_id,

                    "question": question,
                    "expected": expected,

                    "model_name": model_name,
                    "quantization": quantization,
                    "rag": rag_enabled,

                    "success": True,
                    "error": "",

                    "response": response_text,

                    # Latências
                    "retrieval_time_s": retrieval_time_s,
                    "prompt_build_time_s": prompt_build_time_s,
                    "ollama_latency_s": ollama_latency_s,
                    "end_to_end_latency_s": end_to_end_latency_s,

                    # Mantém compatibilidade com análises antigas
                    "latency_s": end_to_end_latency_s,

                    # Tokens / throughput
                    "throughput_tokens_s": throughput_tokens_s,
                    "prompt_token_count": token_metrics["prompt_token_count"],
                    "output_token_count": token_metrics["output_token_count"],
                    "generation_time_ns": token_metrics["generation_time_ns"],
                    "total_time_ns": token_metrics["total_time_ns"],

                    # Memória
                    "baseline_ram_mb": monitoring_data["baseline_ram_mb"],
                    "peak_ram_mb": monitoring_results["peak_ram_mb"],
                    "ram_usage_mb": monitoring_results["ram_usage_mb"],

                    "baseline_vram_mb": monitoring_data["baseline_vram_mb"],
                    "peak_vram_mb": monitoring_results["peak_vram_mb"],
                    "vram_delta_mb": monitoring_results["vram_delta_mb"],

                    # RAG
                    "retrieved_chunks": retrieved_chunks_text
                }

                save_result(
                    result,
                    OUTPUT_CSV
                )

                completed_executions.add(
                    execution_key
                )

                print(f"Retrieval: {retrieval_time_s:.4f}s")
                print(f"Ollama: {ollama_latency_s:.2f}s")
                print(f"End-to-end: {end_to_end_latency_s:.2f}s")

            except Exception as e:

                print(f"ERROR: {e}")

                monitoring_results = safe_stop_monitoring(
                    monitoring_data
                )

                if end_to_end_latency_s is None:
                    end_to_end_latency_s = (
                        time.perf_counter() - end_to_end_start
                        if "end_to_end_start" in locals()
                        else None
                    )

                error_result = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),

                    "sample_index": sample_index,
                    "dataset_id": dataset_id,

                    "question": question,
                    "expected": expected,

                    "model_name": model_name,
                    "quantization": quantization,
                    "rag": rag_enabled,

                    "success": False,
                    "error": str(e),

                    "response": "",

                    "retrieval_time_s": retrieval_time_s,
                    "prompt_build_time_s": prompt_build_time_s,
                    "ollama_latency_s": ollama_latency_s,
                    "end_to_end_latency_s": end_to_end_latency_s,

                    # Mantém compatibilidade
                    "latency_s": end_to_end_latency_s,

                    "throughput_tokens_s": None,
                    "prompt_token_count": None,
                    "output_token_count": None,
                    "generation_time_ns": None,
                    "total_time_ns": None,

                    "baseline_ram_mb": (
                        monitoring_data["baseline_ram_mb"]
                        if monitoring_data is not None
                        else None
                    ),
                    "peak_ram_mb": monitoring_results["peak_ram_mb"],
                    "ram_usage_mb": monitoring_results["ram_usage_mb"],

                    "baseline_vram_mb": (
                        monitoring_data["baseline_vram_mb"]
                        if monitoring_data is not None
                        else None
                    ),
                    "peak_vram_mb": monitoring_results["peak_vram_mb"],
                    "vram_delta_mb": monitoring_results["vram_delta_mb"],

                    "retrieved_chunks": retrieved_chunks_text
                }

                save_result(
                    error_result,
                    OUTPUT_CSV
                )

            time.sleep(PAUSE_BETWEEN_RUNS_S)


# ==================================================
# FIM
# ==================================================

print("\n===================================")
print("EXPERIMENTO FINALIZADO")
print("===================================")
print(f"CSV salvo em: {OUTPUT_CSV}")