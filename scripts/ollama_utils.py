import ollama

from config import (
    NUM_CTX,
    NUM_PREDICT,
    TEMPERATURE
)

# ==================================================
# CALL OLLAMA
# ==================================================

def call_ollama(model_name, prompt):

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT
        }
    )

    return response

# ==================================================
# EXTRACT RESPONSE TEXT
# ==================================================

def extract_response_text(response):

    return response["message"]["content"]

# ==================================================
# EXTRACT TOKEN METRICS
# ==================================================

def extract_token_metrics(response):

    return {
        "prompt_token_count": response.get("prompt_eval_count"),
        "output_token_count": response.get("eval_count"),
        "generation_time_ns": response.get("eval_duration"),
        "total_time_ns": response.get("total_duration")
    }

# ==================================================
# THROUGHPUT
# ==================================================

def calculate_throughput_tokens_s(
    output_token_count,
    generation_time_ns
):

    if (
        output_token_count is None
        or generation_time_ns is None
        or generation_time_ns <= 0
    ):
        return None

    return output_token_count / (
        generation_time_ns / 1e9
    )