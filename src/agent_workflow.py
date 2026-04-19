import json
from typing import Annotated, TypedDict, List, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .llm_client import LLMClient
from .agent_tools import CARDIO_TOOLS

# ── State Definition ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """The state of our medical reasoning agent."""
    messages: Annotated[List[BaseMessage], "The conversation history"]
    patient_data: Dict[str, Any]
    risk_summary: Dict[str, Any]
    clinical_flags: List[str]
    medical_context: str
    is_complete: bool

# ── Node Definitions ────────────────────────────────────────────────────────

def call_model(state: AgentState):
    """The 'brain' node that decides which tool to call or responds."""
    messages = state["messages"]
    client = LLMClient(temperature=0)
    
    # Bind tools to the LLM
    llm_with_tools = client.llm.bind_tools(CARDIO_TOOLS)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def process_tools(state: AgentState):
    """Executes the selected tools and returns the results to the state."""
    tool_node = ToolNode(CARDIO_TOOLS)
    return tool_node

# ── Graph Construction ─────────────────────────────────────────────────────

def create_cardio_agent():
    """Builds and compiles the LangGraph workflow."""
    
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(CARDIO_TOOLS))

    # Add Edges
    workflow.add_edge(START, "agent")

    # Conditional edge: tools or finish
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    # Compile with memory for session tracking
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ── Execution Helper ───────────────────────────────────────────────────────

class CardioAgent:
    """Convenience class to run the agentic workflow."""
    
    def __init__(self):
        self.app = create_cardio_agent()

    def run(self, query: str, patient_data: Dict[str, Any], thread_id: str = "default"):
        """Executes the agent for a given patient and query."""
        
        system_prompt = f"""
        You are an Intelligent Patient Risk Assessment and Health Support System.
        
        CONTEXT: 
        You have access to a patient's clinical data and a trained Machine Learning model.
        Current Patient Data: {json.dumps(patient_data, indent=2)}
        
        YOUR GOAL:
        Provide evidence-based health guidance and preventive recommendations.
        Always verify clinical thresholds (BP, Cholesterol, etc.) using your tools.
        Always include a medical disclaimer stating this is for educational use.
        
        INSTRUCTIONS:
        1. If you don't have enough context, use 'search_medical_guidelines' to look up standard thresholds.
        2. Use 'get_risk_prediction' to see the ML model's output for this patient.
        3. Use 'get_clinical_flags' to see high-level clinical warnings.
        4. Synthesize all findings into a structured response.
        """
        
        inputs = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ],
            "patient_data": patient_data,
            "risk_summary": {},
            "clinical_flags": [],
            "medical_context": "",
            "is_complete": False
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = self.app.invoke(inputs, config=config)
        return final_state["messages"][-1].content

# Simple test script
if __name__ == "__main__":
    agent = CardioAgent()
    # test_data = {"age": 60, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
    # response = agent.run("What are the risk factors for this patient?", test_data)
    # print(response)
