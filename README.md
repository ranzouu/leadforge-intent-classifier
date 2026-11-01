# LeadForge Intent Classifier

Classificador de intenção de prospecção em **hot / warm / cold** a partir de texto do lead. Usa modelo simples baseado em regras + TF-IDF + Logistic Regression, sem API paga.

## Pergunta mensurável
Quão bem o modelo classifica a intenção de compra/prestação de serviços a partir do texto do lead?

## O que ele prova
- Entendimento de **classificação de texto**
- Pipeline: pré-processamento, vetorização, treino, avaliação, exportação
- Métricas objetivas: **accuracy, precision, recall, F1**
- Reprodutibilidade: seed fixo, CSV de resultados, CLI

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Executar
```bash
# Treinar e avaliar
python -m leadforge_intent_classifier

# Demo sem treino (regras apenas)
python -m leadforge_intent_classifier --demo
```

## Output
- `runs/latest.csv` — métricas por classe
- `runs/summary.csv` — accuracy, precision, recall, F1
- `model/` — modelo treinado exportado
