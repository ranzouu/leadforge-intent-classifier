# 🔥 LeadForge Intent Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success)](https://github.com/ranzouu/leadforge-intent-classifier)

Classificador de intenção de prospecção em **hot / warm / cold** a partir de texto do lead. Modelo leve baseado em **TF-IDF + Logistic Regression**, sem API paga, com métricas objetivas e CLI própria.

---

## 🎯 O que faz

Transforma frases de leads em classificação de intenção de compra:

| Entrada | Saída |
|---|---|
| *"Quero contratar um plano de musculação para minha academia"* | 🔥 **hot** |
| *"Estou procurando um sistema para minha loja"* | 🌡️ **warm** |
| *"Só queria ver como funciona"* | ❄️ **cold** |

---

## ✨ Features

- **Classificação multiclasse**: hot, warm, cold a partir de texto livre
- **Pipeline reprodutível**: pré-processamento → TF-IDF → Logistic Regression → avaliação
- **Métricas objetivas**: accuracy, precision, recall, F1 por classe e geral
- **Modo demo**: classificação rápida via regras sem treino
- **CLI própria**: comando `leadforge-intent` instalado no ambiente
- **Exportação de modelo**: `.pkl` serializado via `joblib`
- **Runs rastreáveis**: `runs/latest.csv` e `runs/summary.csv`

---

## 🧠 Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python ≥ 3.10 |
| Vetorização | TF-IDF: word n-grams (1,2) + char_wb n-grams (2,4) em `FeatureUnion` |
| Modelo | Logistic Regression (balanced, C tunado por nested CV) |
| Métricas | accuracy, precision, recall, F1 por classe e geral (weighted/macro) |
| Avaliação | StratifiedKFold out-of-fold, seed=42 |
| Serialização | joblib |
| Testes | pytest |

---

## 📈 Métricas

**Avaliação honesta — StratifiedKFold (5 folds, seed=42) com tuning de `C` aninhado dentro de cada fold. Os números são out-of-fold: toda amostra é classificada por um modelo que nunca a viu no treino.**

| Métrica (média ± desvio entre folds) | Valor |
|---|---|
| Accuracy | **0.633 ± 0.192** |
| Precision (weighted) | **0.634 ± 0.197** |
| Recall (weighted) | **0.633 ± 0.192** |
| F1 (weighted) | **0.630 ± 0.189** |
| F1 (macro) | **0.633** |

**Por classe (predições pooled out-of-fold):**

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| cold | 0.700 | 0.700 | 0.700 | 20 |
| hot | 0.750 | 0.750 | 0.750 | 20 |
| warm | 0.450 | 0.450 | 0.450 | 20 |

- Features: união de word n-grams (1,2) + char_wb n-grams (2,4), ambos `sublinear_tf`
- `C` tunado por fold: `2.0, 0.5, 0.5, 2.0, 0.5` (grid 0.5–4.0, CV interno 3-fold); artefato final treinado em todos os dados com `C=4.0`
- `warm` é a fronteira difícil do problema — separar "interesse" de "só curiosidade" é tênue até em leitura humana; é onde o ganho está
- Dataset: 60 amostras (20/classe) de prospecção PT-BR escritas para exercitar o pipeline. Em produção, o mesmo código roda com dados reais de leads

> Antes desta correção o projeto publicava "accuracy 1.0" medida no mesmo conjunto de treino — inflação por avaliação in-sample. Os valores acima são a métrica real, com o procedimento completo em `runs/` (`cv_folds.csv`, `class_report.csv`, `latest.csv` com as predições out-of-fold).

---

## 🚀 Como rodar

```bash
# 1. Clone o repositório
git clone https://github.com/ranzouu/leadforge-intent-classifier.git
cd leadforge-intent-classifier

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -e .
```

### Uso

```bash
# Treinar, avaliar e exportar modelo
python -m leadforge_intent_classifier

# Demo sem treino (regras apenas)
python -m leadforge_intent_classifier --demo

# Via CLI instalado
leadforge-intent
```

---

## 🧪 Testes

```bash
pytest
```

Testes inclusos:
- Carregamento de dados
- Modo demo
- Treino e avaliação do modelo

---

## 📁 Estrutura

```
leadforge-intent-classifier/
├── src/leadforge_intent_classifier/
│   └── experiment.py
├── tests/
│   └── test_intent_classifier.py
├── data/
│   └── sample_leads.json
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── runs/
│   ├── latest.csv          # predições pooled out-of-fold
│   ├── summary.csv         # métricas CV (média ± desvio)
│   ├── cv_folds.csv        # métricas por fold
│   └── class_report.csv    # precision/recall/F1 por classe
├── scripts/
├── pyproject.toml
└── README.md
```

---

## 📝 Sobre o projeto

Pergunta mensurável: *quão bem o modelo classifica a intenção de compra/prestação de serviços a partir do texto do lead?*

O projeto demonstra:
- Entendimento de **classificação de texto**
- Pipeline completo: pré-processamento, vetorização, treino, avaliação, exportação
- Métricas objetivas: **accuracy, precision, recall, F1**
- Reprodutibilidade: seed fixo (`random_state=42`), CSV de resultados, CLI

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE) para detalhes.

---

Feito com 💜 por [Rayson](https://github.com/ranzouu)
