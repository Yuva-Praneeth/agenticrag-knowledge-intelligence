
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_pipeline import create_retriever


class AgentState(TypedDict):
    question: str
    documents: List[Dict[str, str]]
    answer: str
    verification: str
    retry_count: int


# One Gemini model can perform multiple specialized agent roles.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


def research_agent(state: AgentState) -> dict:
    """Retrieve evidence relevant to the user's question."""
    retriever = create_retriever()
    docs = retriever.invoke(state["question"])

    documents = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    return {"documents": documents}


def answer_agent(state: AgentState) -> dict:
    """Generate an answer using retrieved evidence only."""
    if not state["documents"]:
        return {
            "answer": (
                "I could not find relevant information in the "
                "available documents."
            )
        }

    context = "\n\n".join(
        f"Source: {doc['source']}\n{doc['content']}"
        for doc in state["documents"]
    )

    prompt = f"""
You are NovaMart's enterprise document assistant.

Answer the question using only the supplied evidence.
Do not invent facts, policies, or numbers.
If the evidence does not answer the question, say so.
Mention the source filenames in your answer.

Question:
{state['question']}

Evidence:
{context}

Answer:
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def verification_agent(state: AgentState) -> dict:
    """Check whether the answer is supported by the evidence."""
    context = "\n\n".join(
        f"Source: {doc['source']}\n{doc['content']}"
        for doc in state["documents"]
    )

    prompt = f"""
You are an answer verification agent.

Check whether the answer is supported by the supplied evidence.
An answer that correctly says the documents do not contain
the requested information is acceptable.

Return exactly one label:
SUPPORTED
UNSUPPORTED

Question:
{state['question']}

Evidence:
{context}

Answer:
{state['answer']}
"""

    response = llm.invoke(prompt)
    label = response.content.strip().upper()

    if "SUPPORTED" in label and "UNSUPPORTED" not in label:
        return {"verification": "SUPPORTED"}

    return {"verification": "UNSUPPORTED"}


def route_after_verification(state: AgentState) -> str:
    """Retry research once if the answer isn't supported."""
    if (
        state["verification"] == "UNSUPPORTED"
        and state["retry_count"] < 1
    ):
        return "retry"

    return "finish"


def retry_agent(state: AgentState) -> dict:
    """Increase the retry count before researching again."""
    return {"retry_count": state["retry_count"] + 1}


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("research", research_agent)
    graph.add_node("answer", answer_agent)
    graph.add_node("verify", verification_agent)
    graph.add_node("retry", retry_agent)

    graph.add_edge(START, "research")
    graph.add_edge("research", "answer")
    graph.add_edge("answer", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "retry": "retry",
            "finish": END,
        },
    )

    graph.add_edge("retry", "research")

    return graph.compile()


agent_app = build_agent_graph()


def ask_agentic_rag(question: str) -> dict:
    """Run the multi-agent workflow for a user question."""
    result = agent_app.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "verification": "",
        "retry_count": 0,
    })

    return {
        "answer": result["answer"],
        "sources": result["documents"],
        "verification": result["verification"],
        "retry_count": result["retry_count"],
    }
