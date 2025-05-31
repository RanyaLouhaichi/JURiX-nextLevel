import sys
import os
from typing import Dict
from langgraph.graph import StateGraph, END # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from .intent_router import classify_intent
from agents.chat_agent import ChatAgent
from agents.retrieval_agent import RetrievalAgent
from agents.recommendation_agent import RecommendationAgent
from agents.jira_data_agent import JiraDataAgent
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import functools
import logging
import uuid


logging.basicConfig(
    filename='workflow.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logging.debug(f"Executing {func.__name__} with state: {state}")
        result = func(state)
        logging.debug(f"Completed {func.__name__} with result: {result}")
        return result
    return wrapper

shared_memory = JurixSharedMemory()
chat_agent = ChatAgent(shared_memory)
retrieval_agent = RetrievalAgent(shared_memory)
recommendation_agent = RecommendationAgent(shared_memory)
jira_data_agent = JiraDataAgent()

@log_aspect
def classify_intent_node(state: JurixState) -> JurixState:
    intent_result = classify_intent(state["query"], state["conversation_history"])
    updated_state = state.copy()
    updated_state["intent"] = intent_result
    updated_state["project"] = intent_result.get("project")
    logging.debug(f"Intent classification result: {intent_result}")
    return updated_state

@log_aspect
def jira_data_agent_node(state: JurixState) -> JurixState:
    project = state.get("project")
    if not project:
        logging.warning("No project specified for JiraDataAgent, skipping ticket retrieval")
        return state.copy()
    
    input_data = {
        "project_id": project,
        "time_range": {
            "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-17T23:59:59Z"
        }
    }
    logging.debug(f"Calling JiraDataAgent with input: {input_data}")
    result = jira_data_agent.run(input_data)
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    logging.debug(f"JiraDataAgent returned {len(updated_state['tickets'])} tickets")
    return updated_state

@log_aspect
def recommendation_agent_node(state: JurixState) -> JurixState:
    input_data = {
        "session_id": state["conversation_id"],
        "user_prompt": state["query"],
        "articles": state["articles"],
        "project": state["project"],
        "tickets": state["tickets"],
        "workflow_type": "prompting",
        "intent": state["intent"]
    }
    logging.debug(f"Calling RecommendationAgent with input: {input_data}")
    result = recommendation_agent.run(input_data)
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["needs_context"] = result.get("needs_context", False)
    logging.debug(f"RecommendationAgent returned recommendations: {updated_state['recommendations']}, needs_context: {updated_state['needs_context']}")
    return updated_state

@log_aspect
def retrieval_agent_node(state: JurixState) -> JurixState:
    input_data = {
        "session_id": state["conversation_id"],
        "user_prompt": state["query"],
        "intent": state["intent"]
    }
    logging.debug(f"Calling RetrievalAgent with input: {input_data}")
    result = retrieval_agent.run(input_data)
    updated_state = state.copy()
    updated_state["articles"] = result.get("articles", [])
    logging.debug(f"RetrievalAgent returned articles: {updated_state['articles']}")
    # Notify ChatAgent of retrieved articles via shared memory
    shared_memory.memory["articles"] = updated_state["articles"]
    return updated_state

@log_aspect
def chat_agent_node(state: JurixState) -> JurixState:
    input_data = {
        "session_id": state["conversation_id"],
        "user_prompt": state["query"],
        "articles": state["articles"],
        "recommendations": state["recommendations"],
        "tickets": state["tickets"],
        "intent": state["intent"]
    }
    logging.debug(f"Calling ChatAgent with input: {input_data}")
    result = chat_agent.run(input_data)
    updated_state = state.copy()
    updated_state.update({
        "response": result.get("response", "No response generated"),
        "conversation_history": chat_agent.shared_memory.get_conversation(state["conversation_id"]),
        "articles_used": result.get("articles_used", []),
        "tickets": result.get("tickets", state["tickets"]),
        "workflow_status": result.get("workflow_status", "completed")
    })
    # Handle feedback loop if more context is needed
    if state.get("needs_context", False):
        updated_state["needs_context"] = False
        logging.debug("Re-entering workflow due to need for context")
        return updated_state
    logging.debug(f"Updated state in chat_agent_node: {updated_state}")
    return updated_state

def build_workflow():
    workflow = StateGraph(JurixState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("jira_data_agent", jira_data_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)
    workflow.add_node("retrieval_agent", retrieval_agent_node)
    workflow.add_node("chat_agent", chat_agent_node)

    workflow.set_entry_point("classify_intent")

    def route(state: JurixState) -> str:
        intent = state["intent"]["intent"] if "intent" in state and "intent" in state["intent"] else "generic_question"
        needs_context = state.get("needs_context", False)
        logging.debug(f"Routing with intent: {intent}, needs_context: {needs_context}")
        if needs_context:
            return "chat_agent"  # Loop back to ChatAgent for more context
        routing = {
            "generic_question": "chat_agent",
            "follow_up": "chat_agent",
            "article_retrieval": "retrieval_agent",
            "recommendation": "jira_data_agent"
        }
        return routing.get(intent, "chat_agent")

    workflow.add_conditional_edges(
        "classify_intent",
        route,
    )

    workflow.add_edge("jira_data_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "chat_agent")
    workflow.add_edge("retrieval_agent", "chat_agent")
    workflow.add_edge("chat_agent", END)

    return workflow.compile()

def run_workflow(query: str, conversation_id: str = None) -> JurixState:
    conversation_id = conversation_id or str(uuid.uuid4())
    state = JurixState(
        query=query,
        intent={},
        conversation_id=conversation_id,
        conversation_history=[],
        articles=[],
        recommendations=[],
        tickets=[],
        status="pending",
        response="",
        articles_used=[],
        workflow_status="",
        next_agent="",
        project=None
    )
    workflow = build_workflow()
    final_state = None
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            logging.debug(f"Event from {node_name}: {node_state}")
            final_state = node_state
    logging.debug(f"Final state after stream: {final_state}")
    return final_state