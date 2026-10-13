import json
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA = ROOT / "data"
st.set_page_config(page_title="InsightEngine AI", page_icon="🧠", layout="wide")

@st.cache_data
def load_data():
    tickets = pd.read_csv(DATA / "tickets.csv")
    kb = json.loads((DATA / "knowledge_base.json").read_text(encoding="utf-8"))
    return tickets, kb

@st.cache_resource
def train_model():
    df, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.25, random_state=42, stratify=df["intent"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "test_size": len(y_test)
    }
    return model, metrics

@st.cache_resource
def build_retriever():
    _, kb = load_data()
    docs = [f'{d["title"]}. {d["text"]}' for d in kb]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    return kb, vectorizer, matrix

def retrieve(query, top_k=3):
    kb, vectorizer, matrix = build_retriever()
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()
    order = scores.argsort()[::-1][:top_k]
    return [(kb[i], float(scores[i])) for i in order if scores[i] > 0.03]

def draft_answer(question, results):
    if not results:
        return ("I couldn't find sufficiently relevant evidence in the demo knowledge base. "
                "Please contact a human support representative or add an appropriate policy document.")
    # Reliable extractive baseline; no external model or API is required.
    best = results[0][0]
    return (f"Based on **{best['title']}**, {best['text']} "
            "This is an evidence-based draft from the local demo knowledge base; verify policy details before acting.")

tickets, kb = load_data()
model, metrics = train_model()

st.title("🧠 InsightEngine AI")
st.caption("Hybrid ML + retrieval demo · local data · no paid API required")
st.info("Portfolio prototype: the included ticket examples and policy articles are demo data, not a real company's records or policies.")

with st.sidebar:
    st.header("Project modules")
    page = st.radio("Navigate", ["Ticket Analyzer", "Knowledge Assistant", "Evaluation Lab"])
    st.divider()
    st.metric("Training examples", len(tickets))
    st.metric("Knowledge articles", len(kb))
    st.caption("ML: TF-IDF + Logistic Regression · Retrieval: TF-IDF cosine similarity")

if page == "Ticket Analyzer":
    st.subheader("Classify a customer-support ticket")
    text = st.text_area("Customer message", "I was charged twice and need help with my invoice.", height=110)
    if st.button("Analyze ticket", type="primary"):
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        idx = probs.argmax()
        label, confidence = classes[idx], float(probs[idx])
        c1, c2 = st.columns(2)
        c1.metric("Predicted intent", label.replace("_", " ").title())
        c2.metric("Model confidence", f"{confidence:.1%}")
        if confidence < 0.40:
            st.warning("Low confidence: send this ticket for human review rather than trusting the prediction.")
        st.write("Top predictions")
        ranking = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        st.bar_chart(pd.DataFrame({"Intent": [x[0] for x in ranking], "Probability": [x[1] for x in ranking]}).set_index("Intent"))
        st.subheader("Related knowledge")
        results = retrieve(text)
        if results:
            for doc, score in results:
                with st.expander(f'{doc["id"]} · {doc["title"]} · relevance {score:.2f}'):
                    st.write(doc["text"])
        else:
            st.write("No sufficiently relevant knowledge articles found.")

elif page == "Knowledge Assistant":
    st.subheader("Ask the support knowledge base")
    question = st.text_area("Question", "I reset my password but the email has not arrived. What should I do?", height=100)
    if st.button("Search and answer", type="primary"):
        results = retrieve(question)
        answer = draft_answer(question, results)
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Retrieved evidence")
        if not results:
            st.warning("No relevant passages found. The system abstained.")
        for doc, score in results:
            with st.expander(f'{doc["id"]} · {doc["title"]} · relevance {score:.2f}', expanded=True):
                st.write(doc["text"])
                st.caption(f'Article category: {doc["category"]} · Source ID: {doc["id"]}')
        st.caption("This first version uses an extractive, evidence-grounded answer baseline. Optional local LLM integration is a later milestone.")

else:
    st.subheader("Evaluation Lab")
    c1, c2, c3 = st.columns(3)
    c1.metric("Holdout accuracy", f'{metrics["accuracy"]:.1%}')
    c2.metric("Holdout macro-F1", f'{metrics["macro_f1"]:.3f}')
    c3.metric("Test examples", metrics["test_size"])
    st.write("### Per-class classification report")
    report = pd.DataFrame(metrics["report"]).T
    st.dataframe(report.round(3), use_container_width=True)
    st.write("### Retrieval smoke test")
    queries = [
        "I need to reset my password",
        "Where is my shipment?",
        "I need my invoice",
        "The app keeps freezing",
        "I need to return an item"
    ]
    rows = []
    for q in queries:
        r = retrieve(q, top_k=1)
        rows.append({"Query": q, "Top result": r[0][0]["title"] if r else "No result",
                     "Top score": round(r[0][1], 3) if r else 0.0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Metrics are calculated on a small demo dataset and should not be presented as production performance.")
