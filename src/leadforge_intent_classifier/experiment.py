from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_leads.json"
MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

CV_FOLDS = 5
RANDOM_STATE = 42
C_GRID = [0.5, 1.0, 2.0, 4.0]


def _make_features() -> FeatureUnion:
    """Uniao de features: word n-grams (sintaxe) + char_wb n-grams (morfologia
    do PT-BR — pega radical de 'contrat/automat/pesquis' sem depender de stemmer).
    """
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)),
        ]
    )


def _make_clf(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, C=C)


def _tune_c(texts: list[str], labels: list[str]) -> tuple[float, dict]:
    """Grid manual de C com CV interno — usado DENTRO do fold de treino
    (nested CV): o tuning nunca ve o fold de teste.
    """
    best_c, best_score = 1.0, -1.0
    scores: dict[str, float] = {}
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for c in C_GRID:
        fold_scores = []
        for tr, va in inner.split(texts, labels):
            vec = _make_features()
            X_tr = vec.fit_transform([texts[i] for i in tr])
            X_va = vec.transform([texts[i] for i in va])
            clf = _make_clf(c)
            clf.fit(X_tr, [labels[i] for i in tr])
            _, _, f1, _ = precision_recall_fscore_support(
                [labels[i] for i in va], clf.predict(X_va), average="macro", zero_division=0
            )
            fold_scores.append(float(f1))
        mean = float(np.mean(fold_scores))
        scores[str(c)] = round(mean, 4)
        if mean > best_score:
            best_c, best_score = c, mean
    return best_c, scores


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


def _make_pipeline():
    return (
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    )


def train_model(leads: list[Lead]):
    """Treina o artefato final em TODOS os dados (uso em producao/demo).

    O C vem de grid CV em dados completos; a avaliacao honesta NAO usa este
    modelo — ver `cross_validate` (tuning aninhado dentro de cada fold).
    """
    texts = [_preprocess(l.text) for l in leads]
    labels = [l.label for l in leads]

    best_c, _ = _tune_c(texts, labels)
    vectorizer = _make_features()
    X = vectorizer.fit_transform(texts)
    clf = _make_clf(best_c)
    clf.fit(X, labels)
    clf.C_TUNED = best_c  # type: ignore[attr-defined]

    return vectorizer, clf


def evaluate(vectorizer, clf, leads: list[Lead]) -> tuple[list[RunRecord], dict]:
    """Avaliacao in-sample (smoke test). NAO usar como metrica publicada:
    aqui o modelo ve os mesmos exemplos no treino e na avaliacao.
    """
    texts = [_preprocess(l.text) for l in leads]
    X = vectorizer.transform(texts)
    preds = clf.predict(X)

    records: list[RunRecord] = []
    for lead, pred in zip(leads, preds):
        records.append(
            RunRecord(id=lead.id, text=lead.text, expected=lead.label, predicted=str(pred), correct=str(pred) == lead.label)
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
        "mode": "in-sample",
    }
    return records, summary


def cross_validate(leads: list[Lead], k: int = CV_FOLDS) -> tuple[list[RunRecord], dict, list[dict], list[dict]]:
    """Avaliacao honesta: StratifiedKFold com vectorizer treinado SOMENTE
    no fold de treino (sem vazamento). Retorna predicoes pooled out-of-fold,
    resumo com media + desvio por fold, linhas por fold e relatorio por classe.
    """
    texts = [_preprocess(l.text) for l in leads]
    labels = [l.label for l in leads]
    classes = sorted(set(labels))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    pooled: list[RunRecord] = []
    fold_rows: list[dict] = []
    fold_weighted: list[dict] = []
    tuned_cs: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        # tuning aninhado: C escolhido só com o fold de treino
        best_c, _ = _tune_c(train_texts, train_labels)
        tuned_cs.append(best_c)

        vectorizer = _make_features()
        X_train = vectorizer.fit_transform(train_texts)
        clf = _make_clf(best_c)
        clf.fit(X_train, train_labels)
        preds = clf.predict(vectorizer.transform([texts[i] for i in test_idx]))

        for idx, pred in zip(test_idx, preds):
            pooled.append(
                RunRecord(id=leads[idx].id, text=leads[idx].text, expected=labels[idx], predicted=str(pred))
            )

        y_true = [labels[i] for i in test_idx]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="weighted", zero_division=0
        )
        fold_weighted.append(
            {
                "accuracy": float(accuracy_score(y_true, preds)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        fold_rows.append(
            {
                "fold": fold,
                "train": len(train_idx),
                "test": len(test_idx),
                "accuracy": round(fold_weighted[-1]["accuracy"], 4),
                "precision": round(fold_weighted[-1]["precision"], 4),
                "recall": round(fold_weighted[-1]["recall"], 4),
                "f1": round(fold_weighted[-1]["f1"], 4),
            }
        )

    y_true = [r.expected for r in pooled]
    y_pred = [r.predicted for r in pooled]

    pooled_precision, pooled_recall, pooled_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    class_rows = [
        {
            "class": cls,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(classes, pooled_precision, pooled_recall, pooled_f1, support)
    ]

    def _mean_std(key: str) -> tuple[float, float]:
        values = [row[key] for row in fold_weighted]
        return float(statistics.mean(values)), float(statistics.stdev(values) if len(values) > 1 else 0.0)

    mean_acc, std_acc = _mean_std("accuracy")
    mean_p, std_p = _mean_std("precision")
    mean_r, std_r = _mean_std("recall")
    mean_f1, std_f1 = _mean_std("f1")

    summary = {
        "mode": f"cv{k}",
        "accuracy": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4),
        "precision": round(mean_p, 4),
        "precision_std": round(std_p, 4),
        "recall": round(mean_r, 4),
        "recall_std": round(std_r, 4),
        "f1": round(mean_f1, 4),
        "f1_std": round(std_f1, 4),
        "macro_f1": round(float(macro_f1), 4),
        "folds": k,
        "samples": len(pooled),
        "c_per_fold": ";".join(str(c) for c in tuned_cs),
        "_fold_rows": fold_rows,
        "_class_rows": class_rows,
    }
    return pooled, summary, fold_rows, class_rows


def save_runs(records: list[RunRecord], summary: dict, runs_dir: str = "runs") -> tuple[Path, Path]:
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    latest_path = runs / "latest.csv"
    summary_path = runs / "summary.csv"

    with open(latest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    detail = {k: summary.pop(k) for k in list(summary) if k.startswith("_")}
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    for key, name in (("_fold_rows", "cv_folds.csv"), ("_class_rows", "class_report.csv")):
        if key in detail:
            with open(runs / name, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(detail[key][0].keys()))
                writer.writeheader()
                writer.writerows(detail[key])

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


def run_experiment(demo: bool = False, k: int = CV_FOLDS) -> tuple[list[RunRecord], dict]:
    """Fluxo principal.

    demo=True: classificacao por regras (sem treino).
    demo=False: treina o artefato final em todos os dados E publica as metricas
    honestas de validacao cruzada estratificada (out-of-fold) em runs/.
    """
    leads = _load_leads()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if demo:
        records = []
        for l in leads:
            pred = _demo_predict(l.text)
            records.append(
                RunRecord(id=l.id, text=l.text, expected=l.label, predicted=pred, correct=pred == l.label)
            )
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

    import joblib

    vectorizer, clf = train_model(leads)
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(clf, MODEL_DIR / "model.pkl")

    records, summary, _, _ = cross_validate(leads, k=k)
    summary["final_C"] = float(getattr(clf, "C_TUNED", clf.C))
    return records, summary
