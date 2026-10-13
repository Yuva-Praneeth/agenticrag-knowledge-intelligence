
import streamlit as st
from pathlib import Path
from agents import ask_agentic_rag

st.set_page_config(
    page_title="AgenticRAG | Knowledge Intelligence",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

st.title("🧠 AgenticRAG")
st.caption("Multi-agent enterprise document intelligence")
st.divider()

with st.sidebar:
    st.header("Knowledge Base")
    st.write("Upload TXT or PDF documents to add to your knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Save uploaded documents"):
        saved_count = 0

        for uploaded_file in uploaded_files or []:
            destination = DOCUMENTS_DIR / Path(uploaded_file.name).name

            # Avoid accidentally overwriting an existing document.
            if destination.exists():
                st.warning(
                    f"{uploaded_file.name} already exists. "
                    "Rename the upload if you want to add a new version."
                )
                continue

            destination.write_bytes(uploaded_file.getvalue())
            saved_count += 1

        if saved_count:
            st.success(f"Saved {saved_count} document(s).")

    st.divider()
    st.subheader("Current documents")

    document_names = sorted(
        p.name for p in DOCUMENTS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}
    )

    if document_names:
        for name in document_names:
            st.write(f"📄 {name}")
    else:
        st.info("No documents found.")

st.subheader("Ask your knowledge base")
question = st.text_area(
    "Enter your question",
    placeholder="What should a customer do if a product arrives damaged?",
    height=100,
)

if st.button("Research and answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Researching documents and verifying the answer..."):
            try:
                result = ask_agentic_rag(question.strip())

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Verification")
                if result["verification"] == "SUPPORTED":
                    st.success("Answer judged supported by the evidence.")
                else:
                    st.warning(
                        "The answer could not be verified as supported. "
                        "Please review the evidence."
                    )

                st.caption(f"Retry count: {result['retry_count']}")

                with st.expander("View retrieved source evidence", expanded=True):
                    for index, source in enumerate(result["sources"], start=1):
                        st.markdown(f"**{index}. {Path(source['source']).name}**")
                        st.write(source["content"])
                        st.divider()

            except Exception as error:
                st.error(f"Something went wrong: {error}")
                st.info(
                    "Check your API key, document files, and terminal "
                    "for details. Avoid sharing secrets in screenshots."
                )

st.divider()
st.caption(
    "Prototype: answers and verification are AI-generated. "
    "Check important decisions against the original documents."
)
