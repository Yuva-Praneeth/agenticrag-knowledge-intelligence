
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load API key from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# Local embedding model: no Gemini API calls needed for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load all .txt files from the documents folder."""
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No .txt documents found. Add the NovaMart files "
            "to the documents folder."
        )

    return documents


def create_retriever():
    """Split documents and create a local semantic search index."""
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store.as_retriever(
        search_kwargs={"k": 4}
    )


def answer_question(question: str):
    """Retrieve evidence and generate an answer grounded in it."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    retriever = create_retriever()
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are NovaMart's document assistant.
Answer only using the supplied document context.
Do not invent policies, numbers, or procedures.
If the context does not contain enough information, say so.
Be clear and concise. Cite the source filenames you used."""
        ),
        (
            "human",
            "Question: {question}\n\nDocument context:\n{context}"
        ),
    ])

    # Gemini model name; availability depends on your API access.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "context": context,
    })

    sources = [
        {
            "filename": Path(doc.metadata.get("source", "Unknown")).name,
            "content": doc.page_content,
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
