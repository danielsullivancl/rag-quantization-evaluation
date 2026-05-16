# RAG Quantization Evaluation

This repository contains the code and experimental artifacts used to evaluate the impact of quantization and a controlled Retrieval-Augmented Generation (RAG) configuration on the performance and automatic response quality of local Large Language Models (LLMs).

The experiments compare 4-bit and 8-bit quantized versions of Llama 3.1 8B Instruct executed locally with Ollama. The evaluation uses samples from the TechQA dataset and compares configurations with and without RAG in different computational environments.

## Overview

The main goal of this project is to analyze the trade-off between:

- computational performance;
- memory usage;
- quantization level;
- use of retrieved context;
- automatic response quality.

The evaluated configurations are:

- 4-bit without RAG;
- 4-bit with RAG;
- 8-bit without RAG;
- 8-bit with RAG.

For the RAG configuration, relevant passages are retrieved from the technical document associated with each question in the dataset. This is a controlled RAG setup, intended to isolate the effect of adding relevant context to the prompt rather than evaluating large-scale document retrieval.

## Project Structure

    rag-quantization-evaluation/
    ├── results/
    │   ├── results_combined.csv
    │   ├── results_with_metrics.csv
    │   ├── summary_with_ci95.csv
    │   └── table_results_ci95.tex
    │
    ├── images_desktop/
    │   └── figures generated for the desktop environment
    │
    ├── images_notebook/
    │   └── figures generated for the notebook environment
    │
    ├── images_comparison/
    │   └── cross-machine comparison figures
    │
    ├── scripts/
    │   ├── config.py
    │   ├── run_experiment.py
    │   ├── calculate_metrics.py
    │   ├── results_combined.py
    │   ├── compute_ci_table.py
    │   ├── generate_desktop_figures.py
    │   ├── comparison_machines.py
    │   ├── monitoring.py
    │   ├── ollama_utils.py
    │   └── rag_utils.py
    │
    ├── requirements.txt
    └── README.md

## Requirements

The experiments require:

- Python 3.10 or higher;
- Ollama installed locally;
- Llama 3.1 8B quantized models available in Ollama;
- Python dependencies listed in `requirements.txt`.

Install the Python dependencies with:

    pip install -r requirements.txt

## Ollama Models

Before running the experiment, make sure the required models are available locally in Ollama.

Example:

    ollama pull llama3.1:8b-instruct-q4_K_M
    ollama pull llama3.1:8b-instruct-q8_0

The model names used by the scripts are configured in:

    scripts/config.py

Check this file before running the experiment to confirm the model names, dataset name, number of samples, and output paths.

## Running the Experiment

From the project root, run:

    python scripts/run_experiment.py

This script executes the experiment for each model and configuration defined in `config.py`.

For each sample, the script runs:

- the model without RAG;
- the model with RAG.

The output CSV contains:

- question and expected answer;
- generated response;
- quantization level;
- RAG flag;
- latency metrics;
- throughput;
- token counts;
- RAM and VRAM measurements;
- retrieved chunks for RAG executions.

The main output file is configured in `scripts/config.py`.

## Calculating Automatic Quality Metrics

After running the experiment, calculate the automatic response quality metrics with:

    python scripts/calculate_metrics.py

This script computes:

- BERTScore Precision, Recall and F1;
- ROUGE-1 F1;
- ROUGE-2 F1;
- ROUGE-L F1;
- Exact Match.

It saves a new CSV file with the calculated metrics.

## Combining Results from Multiple Machines

If the experiment was executed in more than one environment, for example desktop and notebook, use:

    python scripts/results_combined.py

This script combines the result files into a single CSV and adds the machine identifier used in the comparative analysis.

The combined output is saved in:

    results/results_combined.csv

## Generating Figures

To generate the desktop figures, run:

    python scripts/generate_desktop_figures.py

The figures are saved in:

    images_desktop/

To generate cross-machine comparison figures, run:

    python scripts/comparison_machines.py

The figures are saved in:

    images_comparison/

If notebook-specific figures are used, they should be generated with the corresponding notebook figure script or by adapting the desktop figure script to the notebook result file.

## Computing Confidence Intervals

To compute 95% confidence intervals for the main metrics and generate the LaTeX table used in the paper, run:

    python scripts/compute_ci_table.py

This script generates:

- `results/summary_with_ci95.csv`
- `results/table_results_ci95.csv`
- `results/table_results_ci95.tex`

The LaTeX table can be included in the paper with:

    \input{tables/table_results_ci95}

or copied directly into the LaTeX source.

## Main Metrics

The experiment evaluates the following performance metrics:

- end-to-end latency;
- Ollama inference latency;
- throughput in tokens per second;
- prompt token count;
- output token count;
- RAM usage;
- VRAM usage.

The automatic quality metrics are:

- BERTScore F1;
- ROUGE-L F1;
- Exact Match.

## Notes on Memory Metrics

The experiment records baseline and peak memory values during execution. The figures in the paper use peak RAM and peak VRAM values to represent the observed memory footprint during model execution.

## Reproducibility Notes

The experiment uses a fixed temperature of 0 to reduce stochastic variation in model generation. Each question is executed once per configuration. Therefore, the confidence intervals reported in the analysis reflect variability across different questions in the workload, not repeated executions of the same question.

## Suggested Execution Order

A typical execution workflow is:

    python scripts/run_experiment.py
    python scripts/calculate_metrics.py
    python scripts/results_combined.py
    python scripts/compute_ci_table.py
    python scripts/generate_desktop_figures.py
    python scripts/comparison_machines.py

Depending on the environment, some steps may need to be executed separately on each machine before combining the results.

## License

This repository is intended for academic use in the context of a doctoral course assignment.
