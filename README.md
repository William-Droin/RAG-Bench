# RAG-Bench

A simple, pipeline-agnostic benchmark project for evaluating LLM/RAG answers with an LLM judge.

## What this project does

- Reads input from a CSV with `Question` and `Answer` columns.
- Sends each `Question` to a target OpenAI-compatible chat endpoint.
- Stores the target model response.
- Uses a judge LLM to score the response against the ideal `Answer`.
- Judge scoring focuses on semantic key-point overlap and contradictions, not literal text similarity.
- Writes timestamped result CSV files for each run.
- Provides a static HTML viewer to inspect benchmark result CSV files locally.

## Project structure

- `/Users/studio1/Desktop/RAG-Bench/src/benchmark.py`: benchmark runner
- `/Users/studio1/Desktop/RAG-Bench/data/input/example_qa.csv`: sample input CSV
- `/Users/studio1/Desktop/RAG-Bench/data/output/`: run outputs (timestamped CSV files)
- `/Users/studio1/Desktop/RAG-Bench/viewer/index.html`: static results viewer
- `/Users/studio1/Desktop/RAG-Bench/.env.example`: environment variable template

## Setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy environment template and fill values:

```bash
cp .env.example .env
```

Required env vars:
- `INPUT_CSV`
- `TARGET_BASE_URL`
- `TARGET_API_KEY`
- `TARGET_MODEL`
- `JUDGE_BASE_URL` (optional; defaults to target)
- `JUDGE_API_KEY` (optional; defaults to target)
- `JUDGE_MODEL`
- `OUTPUT_DIR` (optional)
- `VIEWER_DATA_JS` (optional; default `viewer/results_data.js`)
- `TARGET_TEMPERATURE` (optional; default `0`)
- `JUDGE_TEMPERATURE` (optional; default `0`)

## Input format

CSV header must include exactly these columns (case-sensitive):

```csv
Question,Answer
```

Example:

```csv
Question,Answer
"What is RAG?","RAG combines retrieval with generation so answers are grounded in retrieved context."
```

## Run benchmark

All config is read from `.env` (no CLI args):

```bash
python src/benchmark.py
```

## Output format

Each run writes a CSV file named like:

```text
data/output/benchmark_run_YYYYMMDD_HHMMSS.csv
```

Columns in output:
- `Question`
- `Ideal Response`
- `LLM Answer`
- `Judge Grade` (0-100)
- `Judge Notes`

The benchmark also updates a viewer dataset file (default: `viewer/results_data.js`) by aggregating all files in `OUTPUT_DIR`.

## View results (no web server required)

Open `/Users/studio1/Desktop/RAG-Bench/viewer/index.html` directly in your browser, then:

1. Run the benchmark at least once so `viewer/results_data.js` is generated/updated.
2. Open the page.
3. Apply filters/sort.
4. Compare row-by-row question, ideal answer, model answer, and judge notes.
