from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_leads.json"
MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@dataclass
class Lead:
    id: int
    text: str
    label: str


@dataclass
class RunRecord:
    id: int
    text: str
    expected: str
    predicted: str
    correct: bool = False


def _load_leads() -> list[Lead]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return [Lead(**item) for item in json.load(f)]


def _preprocess(text: str) -> str:
    return text.lower()


def train_model(leads: list[Lead]):
    texts = [_preprocess(l.text) for l in leads]
    labels = [l.label for l in leads]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X, labels)

    return vectorizer, clf


def evaluate(vectorizer, clf, leads: list[Lead]) -> tuple[list[RunRecord], dict]:
    texts = [_preprocess(l.text) for l in leads]
    X = vectorizer.transform(texts)
    preds = clf.predict(X)

    records: list[RunRecord] = []
    for lead, pred in zip(leads, preds):
        records.append(
            RunRecord(
                id=lead.id,
                text=lead.text,
                expected=lead.label,
                predicted=pred,
                correct=pred == lead.label,
            )
        )

    acc = accuracy_score([r.expected for r in records], [r.predicted for r in records])
    precision, recall, f1, _ = precision_recall_fscore_support(
        [r.expected for r in records], [r.predicted for r in records], average="weighted", zero_division=0
    )

    summary = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "samples": len(records),
    }
    return records, summary


def save_runs(records: list[RunRecord], summary: dict, runs_dir: str = "runs") -> tuple[Path, Path]:
    latest_path = Path(runs_dir) / "latest.csv"
    summary_path = Path(runs_dir) / "summary.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(latest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return latest_path, summary_path


def predict_text(vectorizer, clf, text: str) -> str:
    X = vectorizer.transform([_preprocess(text)])
    return str(clf.predict(X)[0])


def _demo_predict(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["quero contratar", "quero automatizar", "preciso de um site", "quero um aplicativo", "quero contratar um personal"]):
        return "hot"
    if any(w in t for w in ["estou procurando", "quero melhorar", "precisa de um sistema", "precisa de ajuda", "começando"]):
        return "warm"
    return "cold"


def run_experiment(demo: bool = False) -> tuple[list[RunRecord], dict]:
    leads = _load_leads()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if demo:
        records = [RunRecord(id=l.id, text=l.text, expected=l.label, predicted=_demo_predict(l.text)) for l in leads]
        acc = sum(1 for r in records if r.correct) / max(len(records), 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            [r.expected for r in records], [r.predicted for r in records], average="weighted", zero_division=0
        )
        summary = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "samples": len(records),
            "mode": "demo",
        }
        return records, summary

    vectorizer, clf = train_model(leads)
    records, summary = evaluate(vectorizer, clf, leads)

    import joblib
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(clf, MODEL_DIR / "model.pkl")

    summary["mode"] = "trained"
    return records, summary
