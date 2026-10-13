# InsightEngine AI

A six-hour portfolio prototype combining classical ML, NLP retrieval, and an evidence-grounded support assistant.

## What's included
- Ticket intent classifier: TF-IDF + Logistic Regression
- Local knowledge base with source IDs
- Retrieval: TF-IDF + cosine similarity
- Evidence-grounded extractive answer baseline and abstention when no passage is relevant
- Holdout classification metrics and retrieval smoke tests
- Streamlit interface; no paid API required

## Run on Windows
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal.

## Demo flow
1. Ticket Analyzer: enter a billing, login, delivery, refund, or technical issue.
2. Knowledge Assistant: ask a question covered by the knowledge base, then ask something unsupported to see abstention.
3. Evaluation Lab: inspect holdout metrics and retrieval smoke tests.

## Important limitations
- Included data are small, hand-curated demo examples, not a production dataset.
- Retrieval is TF-IDF based, not a neural embedding model in this initial six-hour version.
- Answering is extractive/evidence-grounded; a local LLM integration is a stretch goal.
- Metrics are for demonstrating the evaluation workflow only. Do not claim production-level accuracy.
- Never use the demo policies as real business or legal guidance.

## Suggested next improvements
1. Add Sentence Transformers embeddings and compare Recall@K against the TF-IDF retriever.
2. Fine-tune a compact Hugging Face classifier and compare macro-F1 and latency.
3. Add optional Ollama generation behind a feature flag.
4. Add LangGraph only when you have a real conditional workflow to orchestrate.
5. Add tests for low-confidence predictions, missing evidence, and retrieval regressions.
