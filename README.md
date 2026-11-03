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
| Vetorização | TF-IDF + n-grams (1,2) |
| Modelo | Logistic Regression (balanced) |
| Métricas | accuracy, precision, recall, F1 |
| CLI | Rich |
| Testes | pytest |
| Serialização | joblib |

---

## 📈 Métricas

**Dataset de exemplo (15 amostras, treino + avaliação no mesmo conjunto):**

| Métrica | Valor |
|---|---|
| Accuracy | **100.0%** |
| Precision | **100.0%** |
| Recall | **100.0%** |
| F1 | **100.0%** |

> Resultados no dataset sintético pequeno. Em produção com dados reais, considere validação cruzada e aumento do dataset.

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
│   ├── latest.csv
│   └── summary.csv
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
