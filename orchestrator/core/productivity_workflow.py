import sys
import os
from typing import Dict, Optional, List, Any
from langgraph.graph import StateGraph, END # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import functools
import logging
import uuid
from datetime import datetime
from agents.jira_data_agent import JiraDataAgent
from agents.productivity_dashboard_agent import ProductivityDashboardAgent
from agents.recommendation_agent import RecommendationAgent

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logging.basicConfig(filename='productivity_workflow.log', level=logging.INFO)
        logging.info(f"Executing {func.__name__} with state: {state}")
        result = func(state)
        logging.info(f"Completed {func.__name__} with result: {result}")
        return result
    return wrapper

shared_memory = JurixSharedMemory()
jira_data_agent = JiraDataAgent()
productivity_dashboard_agent = ProductivityDashboardAgent()
recommendation_agent = RecommendationAgent(shared_memory)

@log_aspect
def jira_data_agent_node(state: JurixState) -> JurixState:
    input_data = {
        "project_id": state["project_id"],
        "time_range": state["time_range"]
    }
    result = jira_data_agent.run(input_data)
    
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["metadata"] = result.get("metadata", {})
    
    if updated_state["workflow_status"] == "failure":
        updated_state["error"] = "Failed to retrieve Jira ticket data"
    
    logging.info(f"Updated state in jira_data_agent_node: {updated_state}")
    return updated_state

@log_aspect
def recommendation_agent_node(state: JurixState) -> JurixState:
    project_id = state["project_id"]
    ticket_count = len(state["tickets"])
    prompt = f"Analyze productivity data for {project_id} with {ticket_count} tickets and provide recommendations for improving team efficiency"
    
    input_data = {
        "session_id": state["conversation_id"],
        "user_prompt": prompt,
        "project": project_id,
        "ticket_id": state["tickets"][0]["key"] if state["tickets"] else None,
        "tickets": state["tickets"],
        "project_id": project_id,
        "workflow_type": "productivity"  # Pass workflow type
    }
    result = recommendation_agent.run(input_data)
    
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["recommendation_status"] = result.get("workflow_status", "failure")
    
    if updated_state["recommendation_status"] == "failure" and updated_state["workflow_status"] == "success":
        updated_state["recommendations"] = [
            "Consider balanced workload distribution among team members",
            "Review tickets in bottleneck stages to identify common blockers",
            "Schedule regular process review meetings to address efficiency issues"
        ]
        logging.warning("Using default recommendations due to recommendation agent failure")
    
    logging.info(f"Updated state in recommendation_agent_node: {updated_state}")
    return updated_state

@log_aspect
def productivity_dashboard_agent_node(state: JurixState) -> JurixState:
    input_data = {
        "tickets": state["tickets"],
        "recommendations": state["recommendations"],
        "project_id": state["project_id"]
    }
    result = productivity_dashboard_agent.run(input_data)
    
    updated_state = state.copy()
    updated_state["metrics"] = result.get("metrics", {})
    updated_state["visualization_data"] = result.get("visualization_data", {})
    updated_state["report"] = result.get("report", "")
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    
    dashboard_id = f"dashboard_{state['project_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shared_memory.store(dashboard_id, {
        "project_id": state["project_id"],
        "time_range": state["time_range"],
        "metrics": updated_state["metrics"],
        "visualization_data": updated_state["visualization_data"],
        "report": updated_state["report"],
        "recommendations": updated_state["recommendations"],
        "timestamp": datetime.now().isoformat()
    })
    
    updated_state["dashboard_id"] = dashboard_id
    
    logging.info(f"Updated state in productivity_dashboard_agent_node: {updated_state}")
    return updated_state

def build_productivity_workflow():
    workflow = StateGraph(JurixState)
    
    workflow.add_node("jira_data_agent", jira_data_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)
    workflow.add_node("productivity_dashboard_agent", productivity_dashboard_agent_node)
    
    workflow.set_entry_point("jira_data_agent")
    
    workflow.add_edge("jira_data_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "productivity_dashboard_agent")
    workflow.add_edge("productivity_dashboard_agent", END)
    
    def handle_error(state: JurixState) -> str:
        if state["workflow_status"] == "failure":
            return END
        return "recommendation_agent"
    
    workflow.add_conditional_edges(
        "jira_data_agent",
        handle_error
    )
    
    return workflow.compile()

def run_productivity_workflow(project_id: str, time_range: Dict[str, str], conversation_id: str = None) -> JurixState:
    conversation_id = conversation_id or str(uuid.uuid4())
    
    state = JurixState(
        query="",
        intent={},
        conversation_id=conversation_id,
        conversation_history=[],
        articles=[],
        recommendations=[],
        status="pending",
        response="",
        articles_used=[],
        workflow_status="",
        next_agent="",
        project=project_id,
        project_id=project_id,
        time_range=time_range,
        tickets=[],
        metrics={},
        visualization_data={},
        report="",
        metadata={},
        ticket_id="",
        article={},
        redundant=False,
        refinement_suggestion=None,
        approved=False,
        refinement_count=0,
        has_refined=False,
        iteration_count=0,
        workflow_stage="",
        recommendation_id=None,
        workflow_history=[],
        error=None,
        recommendation_status=None,
        dashboard_id=None
    )
    
    logging.info(f"Initial state before stream: {state}")
    workflow = build_productivity_workflow()
    final_state = state
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            logging.info(f"Event from {node_name}: {node_state}")
            final_state = node_state
    
    logging.info(f"Final state after stream: {final_state}")
    return final_state