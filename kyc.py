import os
from typing import TypedDict, Annotated, List, Union
import operator

from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END


# --- Ollama Setup ---
# The 'llama3' model must be pulled locally using `ollama pull llama3`
llm = ChatOllama(model="llama3", temperature=0)
embeddings = OllamaEmbeddings(model="llama3")


# --- RAG Tools ---
def create_internal_rag_tool():
    """Simulates a company's internal vector database of KYC documents."""
    private_data = [
        "Customer ABC Inc. is a financial services company incorporated in Delaware in 2020.",
        "Beneficial owners of ABC Inc. are John Doe (60%) and Jane Smith (40%).",
        "The risk assessment for ABC Inc. was last reviewed on 2024-03-15 and is classified as 'medium risk'.",
        "Another customer, XYZ Ltd., is a technology company with no adverse media history found in internal records."
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(private_data)

    vector_store = Chroma.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    return Tool(
        name="internal_kyc_tool",
        func=retriever.invoke,
        description="""
            Useful for retrieving internal, private KYC and risk assessment information about a customer.
            Input must be a query about a specific customer or KYC detail.
        """
    )


def create_public_search_tool():
    """Performs a public internet search for adverse media or sanctions."""
    return DuckDuckGoSearchRun(
        name="public_web_search",
        description="""
            Useful for performing a public internet search for adverse media, sanctions, or company information.
            Input must be a clear search query.
        """
    )


# --- Agentic Workflow with LangGraph ---
class AgentState(TypedDict):
    """The state for the graph."""
    input: str
    chat_history: List[str]
    agent_outcome: Union[str, None]
    tool_input: Union[str, None]
    intermediate_steps: Annotated[list, operator.add]


def orchestrator(state):
    """
    Decides which tool to use next and prepares the tool input.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a KYC orchestrator. Your task is to analyze a customer review request and determine what information is needed."),
        ("user", "For the customer review, analyze: {input}. Previous steps: {intermediate_steps}"),
        ("user", "Do I need to run an `internal_kyc_tool` or a `public_web_search`? State your decision clearly, then suggest a query for the tool to run. Format your response as 'tool_name: query'. For example: 'internal_kyc_tool: beneficial owners of ABC Inc.'"),
    ])

    chain = prompt | llm
    decision = chain.invoke({"input": state["input"], "intermediate_steps": state.get("intermediate_steps", [])})
    
    print(f"\n---Orchestrator Decision: {decision.content}---")

    # Parse the LLM's structured output
    if ":" in decision.content:
        tool_name, query = decision.content.split(":", 1)
        tool_name = tool_name.strip()
        query = query.strip()
        return {"agent_outcome": tool_name, "tool_input": query}
    
    return {"agent_outcome": "final_response", "tool_input": None}


def run_agent(state, agent_tool):
    """
    Executes the selected agent or tool.
    """
    tool_input = state['tool_input']
    print(f"\n---Running agent '{agent_tool.name}' with input: '{tool_input}'---")

    response = agent_tool.invoke(tool_input)
    
    # LangGraph requires a dictionary for the state update
    return {"agent_outcome": str(response), "tool_input": None}


def determine_next_step(state):
    """
    The conditional edge that decides where to go next based on orchestrator's output.
    """
    if state['agent_outcome'] == "internal_kyc_tool":
        return "internal_kyc"
    elif state['agent_outcome'] == "public_web_search":
        return "public_search"
    else:
        return "final_response"


# --- Build the LangGraph Workflow ---
internal_kyc_tool = create_internal_rag_tool()
public_search_tool = create_public_search_tool()

workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("internal_kyc", lambda state: run_agent(state, internal_kyc_tool))
workflow.add_node("public_search", lambda state: run_agent(state, public_search_tool))

workflow.add_conditional_edges(
    "orchestrator", 
    determine_next_step,
    {
        "internal_kyc": "internal_kyc",
        "public_search": "public_search",
        "final_response": END,
    }
)
workflow.add_edge("internal_kyc", END)
workflow.add_edge("public_search", END)

workflow.set_entry_point("orchestrator")
app = workflow.compile()


# --- Example Invocation ---
if __name__ == "__main__":
    inputs = {"input": "Analyze the beneficial owners of ABC Inc."}
    for step in app.stream(inputs):
        print(step)
