#!/usr/bin/env python3
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


TARGET_COLUMNS = {"Question", "Answer"}
OUTPUT_COLUMNS = [
    "Question",
    "Ideal Response",
    "LLM Answer",
    "Judge Grade",
    "Judge Notes",
]


class BenchmarkError(Exception):
    pass


@dataclass
class JudgeResult:
    grade: float
    notes: str


@dataclass
class Config:
    input_csv: str
    output_dir: str
    target_base_url: str
    target_api_key: str
    target_model: str
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    target_temperature: float
    judge_temperature: float
    viewer_data_js: str


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise BenchmarkError(
                f"Chat completion failed ({response.status_code}): {response.text}"
            )

        body = response.json()
        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise BenchmarkError(f"Unexpected completion response shape: {body}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise BenchmarkError(f"{name} must be a valid number, got: {value}") from exc


def load_config() -> Config:
    input_csv = os.getenv("INPUT_CSV", "data/input/example_qa.csv")
    output_dir = os.getenv("OUTPUT_DIR", "data/output")
    target_base_url = os.getenv("TARGET_BASE_URL", "https://api.openai.com")
    target_api_key = os.getenv("TARGET_API_KEY", "")
    target_model = os.getenv("TARGET_MODEL", "")
    judge_base_url = os.getenv("JUDGE_BASE_URL", "") or target_base_url
    judge_api_key = os.getenv("JUDGE_API_KEY", "") or target_api_key
    judge_model = os.getenv("JUDGE_MODEL", "")

    cfg = Config(
        input_csv=input_csv,
        output_dir=output_dir,
        target_base_url=target_base_url,
        target_api_key=target_api_key,
        target_model=target_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_model=judge_model,
        target_temperature=_env_float("TARGET_TEMPERATURE", 0.0),
        judge_temperature=_env_float("JUDGE_TEMPERATURE", 0.0),
        viewer_data_js=os.getenv("VIEWER_DATA_JS", "viewer/results_data.js"),
    )

    missing = []
    if not cfg.target_api_key:
        missing.append("TARGET_API_KEY")
    if not cfg.target_model:
        missing.append("TARGET_MODEL")
    if not cfg.judge_model:
        missing.append("JUDGE_MODEL")

    if missing:
        raise BenchmarkError("Missing required .env values: " + ", ".join(missing))

    return cfg


def load_qa_rows(path: str) -> List[Dict[str, str]]:
    input_path = Path(path)
    if not input_path.exists():
        raise BenchmarkError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise BenchmarkError("Input CSV is missing a header row.")

        field_set = {name.strip() for name in reader.fieldnames}
        if not TARGET_COLUMNS.issubset(field_set):
            raise BenchmarkError(
                "Input CSV must include columns: Question, Answer. "
                f"Found: {', '.join(reader.fieldnames)}"
            )

        rows: List[Dict[str, str]] = []
        for i, row in enumerate(reader, start=2):
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question and not answer:
                continue
            if not question or not answer:
                raise BenchmarkError(
                    f"Row {i} must include both Question and Answer values."
                )
            rows.append({"Question": question, "Answer": answer})

    if not rows:
        raise BenchmarkError("Input CSV has no valid rows.")

    return rows


def generate_answer(
    client: OpenAICompatibleClient,
    model: str,
    question: str,
    temperature: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant. Answer the user's question directly and factually.",
        },
        {"role": "user", "content": question},
    ]
    return client.chat_completion(model=model, messages=messages, temperature=temperature)


def _extract_json(text: str) -> Optional[Dict]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_answer(
    client: OpenAICompatibleClient,
    model: str,
    question: str,
    ideal_answer: str,
    llm_answer: str,
    temperature: float,
) -> JudgeResult:
    system_prompt = (
        "You are an expert benchmark evaluator. Grade the candidate answer against the ideal answer "
        "based on semantic key-point coverage, factual correctness, and contradictions. "
        "Do NOT use text similarity as a criterion. "
        "Do NOT penalize extra details unless they directly contradict the ideal answer or question context. "
        "Return strictly valid JSON with this exact schema: "
        '{"grade": number 0-100, "notes": string, '
        '"key_points_present": [string], "missing_key_points": [string], "contradictions": [string]}.'
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Ideal Answer:\n{ideal_answer}\n\n"
        f"Candidate Answer:\n{llm_answer}\n\n"
        "Scoring guidance:\n"
        "- 90-100: Covers all key points; no contradictions.\n"
        "- 70-89: Mostly correct; minor omissions; no major contradictions.\n"
        "- 40-69: Partial key-point coverage or notable inaccuracies.\n"
        "- 0-39: Misses core points or contains major contradictions.\n"
        "Return only JSON."
    )

    raw = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    data = _extract_json(raw)
    if not data:
        return JudgeResult(
            grade=0.0,
            notes=(
                "Judge output was not valid JSON. Raw output captured for inspection: "
                + raw[:500]
            ),
        )

    grade_raw = data.get("grade")
    notes = str(data.get("notes", "")).strip()

    try:
        grade = float(grade_raw)
    except (TypeError, ValueError):
        grade = 0.0
        if notes:
            notes = f"{notes} | Invalid grade format from judge."
        else:
            notes = "Invalid grade format from judge."

    grade = max(0.0, min(100.0, grade))

    kp = data.get("key_points_present")
    mk = data.get("missing_key_points")
    ct = data.get("contradictions")

    details = []
    if isinstance(kp, list) and kp:
        details.append("Key points present: " + "; ".join(str(x) for x in kp[:5]))
    if isinstance(mk, list) and mk:
        details.append("Missing key points: " + "; ".join(str(x) for x in mk[:5]))
    if isinstance(ct, list) and ct:
        details.append("Contradictions: " + "; ".join(str(x) for x in ct[:5]))

    merged_notes = notes
    if details:
        suffix = " | ".join(details)
        merged_notes = f"{merged_notes} | {suffix}" if merged_notes else suffix

    return JudgeResult(grade=grade, notes=merged_notes or "No notes provided by judge.")


def write_results_csv(rows: List[Dict[str, str]], output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"benchmark_run_{timestamp}.csv"

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _read_result_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "Question": (row.get("Question") or "").strip(),
                    "Ideal Response": (row.get("Ideal Response") or "").strip(),
                    "LLM Answer": (row.get("LLM Answer") or "").strip(),
                    "Judge Grade": (row.get("Judge Grade") or "").strip(),
                    "Judge Notes": (row.get("Judge Notes") or "").strip(),
                    "__file": path.name,
                }
            )
    return rows


def write_viewer_data(output_dir: str, viewer_data_js: str) -> Path:
    out_dir = Path(output_dir)
    files = sorted(out_dir.glob("benchmark_run_*.csv"))
    all_rows: List[Dict[str, str]] = []
    for file_path in files:
        all_rows.extend(_read_result_csv(file_path))

    payload = {"rows": all_rows, "updated_at": datetime.now().isoformat(timespec="seconds")}

    target = Path(viewer_data_js)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "window.BENCHMARK_DATA = " + json.dumps(payload, ensure_ascii=True) + ";\n"
    target.write_text(content, encoding="utf-8")
    return target


def run() -> int:
    load_dotenv()
    cfg = load_config()

    qa_rows = load_qa_rows(cfg.input_csv)

    target_client = OpenAICompatibleClient(cfg.target_base_url, cfg.target_api_key)
    judge_client = OpenAICompatibleClient(cfg.judge_base_url, cfg.judge_api_key)

    output_rows: List[Dict[str, str]] = []

    for idx, qa in enumerate(qa_rows, start=1):
        question = qa["Question"]
        ideal = qa["Answer"]

        print(f"[{idx}/{len(qa_rows)}] Generating answer...")
        llm_answer = generate_answer(
            client=target_client,
            model=cfg.target_model,
            question=question,
            temperature=cfg.target_temperature,
        )

        print(f"[{idx}/{len(qa_rows)}] Judging answer...")
        judge = judge_answer(
            client=judge_client,
            model=cfg.judge_model,
            question=question,
            ideal_answer=ideal,
            llm_answer=llm_answer,
            temperature=cfg.judge_temperature,
        )

        output_rows.append(
            {
                "Question": question,
                "Ideal Response": ideal,
                "LLM Answer": llm_answer,
                "Judge Grade": f"{judge.grade:.2f}",
                "Judge Notes": judge.notes,
            }
        )

    result_path = write_results_csv(output_rows, cfg.output_dir)

    grades = [float(r["Judge Grade"]) for r in output_rows]
    avg_grade = sum(grades) / len(grades)

    print(f"\nCompleted {len(output_rows)} rows")
    print(f"Average Judge Grade: {avg_grade:.2f}")
    print(f"Results written to: {result_path}")
    viewer_data = write_viewer_data(cfg.output_dir, cfg.viewer_data_js)
    print(f"Viewer data updated: {viewer_data}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
