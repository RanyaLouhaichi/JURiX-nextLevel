import sys
import os
from typing import Dict, Optional, List, Any
from langgraph.graph import StateGraph, END # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import functools
import logging
import uuid
from agents.jira_article_generator_agent import JiraArticleGeneratorAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.recommendation_agent import RecommendationAgent
from agents.jira_data_agent import JiraDataAgent

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logging.basicConfig(filename='jira_workflow.log', level=logging.INFO)
        logging.info(f"Executing {func.__name__} with state: {state}")
        result = func(state)
        logging.info(f"Completed {func.__name__} with result: {result}")
        return result
    return wrapper

shared_memory = JurixSharedMemory()
jira_article_generator = JiraArticleGeneratorAgent(shared_memory)
knowledge_base = KnowledgeBaseAgent(shared_memory)
recommendation_agent = RecommendationAgent(shared_memory)
jira_data_agent = JiraDataAgent()  

def store_recommendations(ticket_id: str, recommendations: List[str]) -> str:
    recommendation_id = f"rec_{ticket_id}_{str(uuid.uuid4())}"
    shared_memory.store(recommendation_id, {"ticket_id": ticket_id, "recommendations": recommendations})
    logging.info(f"Stored recommendations with ID {recommendation_id}: {recommendations}")
    return recommendation_id

def run_recommendation_agent(state: Dict[str, Any]) -> tuple[str, List[str]]:
    # Get ticket data first to provide context to the recommendation agent
    logging.info(f"Fetching ticket data for recommendation agent for ticket {state['ticket_id']}")
    
    # Fetch related ticket data
    project_id = state.get("project", "PROJ123")
    jira_input = {
        "project_id": project_id,
        "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"}
    }
    
    try:
        # Get the ticket data first
        ticket_data_result = jira_data_agent.run(jira_input)
        tickets = ticket_data_result.get("tickets", [])
        logging.info(f"Retrieved {len(tickets)} tickets for project {project_id}")
        
        # Create a more specific input for recommendation agent with the tickets
        input_data = {
            "session_id": state.get("conversation_id", f"rec_{state['ticket_id']}"),
            "user_prompt": f"Analyze resolved ticket {state['ticket_id']} and provide specific action items",
            "articles": [state.get("article", {})] if state.get("article") else [],
            "project": project_id,
            "tickets": tickets,  # Pass all the tickets for context
            "workflow_type": "knowledge_base",
            "intent": {"intent": "ticket_recommendations"}
        }
        
        logging.info(f"Calling recommendation agent with {len(tickets)} tickets...")
        result = recommendation_agent.run(input_data)
        recommendations = result.get("recommendations", [])
        
        if not recommendations or (len(recommendations) == 1 and "provide more project-specific details" in recommendations[0].lower()):
            logging.warning("Recommendation agent returned a request for more context. Trying with more specific ticket data.")
            
            # Find the specific ticket data
            target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
            if target_ticket:
                # Try again with more specific data
                ticket_fields = target_ticket.get("fields", {})
                enriched_prompt = (
                    f"Analyze resolved ticket {state['ticket_id']} - {ticket_fields.get('summary', 'No summary')}. "
                    f"Status: {ticket_fields.get('status', {}).get('name', 'Unknown')}. "
                    f"Consider how the resolution of this ticket can benefit other parts of the project."
                )
                input_data["user_prompt"] = enriched_prompt
                result = recommendation_agent.run(input_data)
                recommendations = result.get("recommendations", [])
        
        if not recommendations:
            logging.warning("Recommendation agent still couldn't generate recommendations. Using fallback options.")
            recommendations = [
                "Consider applying the solution from this ticket to similar issues in the project.",
                "Schedule a knowledge-sharing session to communicate this resolution to the team.",
                "Update relevant documentation to reflect the new solution approach.",
                "Analyze similar tickets to identify pattern of recurring issues."
            ]
    
    except Exception as e:
        logging.error(f"Error in recommendation agent: {str(e)}")
        recommendations = [
            "Review similar tickets to apply this solution pattern.",
            "Document this resolution in the team knowledge base.",
            "Consider creating a reusable component from this solution."
        ]
    
    recommendation_id = store_recommendations(state["ticket_id"], recommendations)
    return recommendation_id, recommendations

def run_jira_workflow(ticket_id: str, conversation_id: str = None, project_id: str = "PROJ123") -> JurixState:
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
        project=project_id,  # Use the provided project ID
        ticket_id=ticket_id,
        article={},
        redundant=False,
        refinement_suggestion=None,
        approved=False,
        refinement_count=0,
        has_refined=False,
        iteration_count=0,
        workflow_stage="started",
        recommendation_id=None,
        workflow_history=[],
        autonomous_refinement_done=False
    )
    
    # Run recommendation agent in parallel
    logging.info(f"Running recommendation agent for ticket {ticket_id}, project {project_id}")
    recommendation_id, recommendations = run_recommendation_agent(state)
    state["recommendation_id"] = recommendation_id
    state["recommendations"] = recommendations
    logging.info(f"RecommendationAgent completed, ID: {recommendation_id}, recommendations: {recommendations}")

    workflow = build_jira_workflow()
    final_state = state
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            logging.info(f"Event from {node_name}: {node_state}")
            final_state = node_state
            
            # Update recommendations if they were placeholder
            if (node_name == "jira_article_generator" and 
                final_state.get("article") and 
                "provide more project-specific details" in str(final_state.get("recommendations", []))):
                logging.info("Updating recommendations with article context")
                final_state_dict = dict(final_state)
                final_state_dict["article"] = final_state.get("article", {})
                recommendation_id, recommendations = run_recommendation_agent(final_state_dict)
                final_state["recommendation_id"] = recommendation_id
                final_state["recommendations"] = recommendations
                
    logging.info(f"Final state after stream: {final_state}")
    return final_state

def build_jira_workflow():
    workflow = StateGraph(JurixState)
    workflow.add_node("jira_article_generator", jira_article_generator_node)
    workflow.add_node("knowledge_base", knowledge_base_node)

    workflow.set_entry_point("jira_article_generator")

    def route(state: JurixState) -> str:
        logging.info(f"[Route] Current state: {state}")
        
        # Check for critical workflow failures
        if state.get("workflow_status") == "failure" or not state.get("article") or state["article"].get("status") == "error":
            state["workflow_stage"] = "terminated_failure"
            state["workflow_history"].append({
                "step": "terminated",
                "article": state["article"],
                "redundant": state["redundant"],
                "refinement_suggestion": state["refinement_suggestion"],
                "workflow_status": state["workflow_status"],
                "workflow_stage": state["workflow_stage"],
                "recommendation_id": state["recommendation_id"],
                "recommendations": state["recommendations"],
                "autonomous_refinement_done": state.get("autonomous_refinement_done", False)
            })
            return END
        
        if state.get("workflow_stage") in ["article_generated", "article_refined"]:
            # Always go to knowledge base after article generation/refinement
            return "knowledge_base"
        
        elif state.get("workflow_stage") == "knowledge_base_evaluated":
            # If there's a refinement suggestion and we haven't refined yet, go back to article generator
            if state.get("refinement_suggestion") and not state.get("has_refined", False):
                logging.info("Refinement needed, going back to article generator")
                return "jira_article_generator"
            else:
                # Otherwise, we're done - set to waiting for approval
                logging.info("No refinement needed or already refined, going to approval")
                state["workflow_stage"] = "waiting_for_approval"
                state["workflow_history"].append({
                    "step": "waiting_for_approval",
                    "article": state["article"],
                    "redundant": state["redundant"],
                    "refinement_suggestion": state["refinement_suggestion"],
                    "workflow_status": state["workflow_status"],
                    "workflow_stage": state["workflow_stage"],
                    "recommendation_id": state["recommendation_id"],
                    "recommendations": state["recommendations"],
                    "autonomous_refinement_done": state.get("autonomous_refinement_done", False)
                })
                return END
        
        if state.get("approved", False):
            state["workflow_stage"] = "complete"
            state["workflow_history"].append({
                "step": "approval_submitted",
                "article": state["article"],
                "redundant": state["redundant"],
                "refinement_suggestion": state["refinement_suggestion"],
                "approved": state["approved"],
                "workflow_status": state["workflow_status"],
                "workflow_stage": state["workflow_stage"],
                "recommendation_id": state["recommendation_id"],
                "recommendations": state["recommendations"],
                "autonomous_refinement_done": state.get("autonomous_refinement_done", False)
            })
        
        return END

    workflow.add_conditional_edges(
        "jira_article_generator",
        route,
    )
    workflow.add_conditional_edges(
        "knowledge_base",
        route,
    )

    return workflow.compile()

# Existing nodes (unchanged)
@log_aspect
def jira_article_generator_node(state: JurixState) -> JurixState:
    input_data = {
        "ticket_id": state["ticket_id"],
        "refinement_suggestion": state.get("refinement_suggestion")
    }
    result = jira_article_generator.run(input_data)
    updated_state = state.copy()
    updated_state["article"] = result.get("article", {})
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["autonomous_refinement_done"] = result.get("autonomous_refinement_done", False)
    
    # Track if refinement has been performed
    if state.get("refinement_suggestion") is not None:
        updated_state["has_refined"] = True
        updated_state["workflow_stage"] = "article_refined"
    else:
        updated_state["has_refined"] = False
        updated_state["workflow_stage"] = "article_generated"
    
    # Don't clear the refinement suggestion until after KB evaluation
    updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    history_entry = {
        "step": "initial_generation" if not updated_state["has_refined"] else "manual_refinement",
        "article": updated_state["article"],
        "redundant": updated_state.get("redundant", False),
        "refinement_suggestion": updated_state.get("refinement_suggestion"),
        "workflow_status": updated_state["workflow_status"],
        "workflow_stage": updated_state["workflow_stage"],
        "recommendation_id": updated_state["recommendation_id"],
        "recommendations": updated_state["recommendations"],
        "autonomous_refinement_done": updated_state["autonomous_refinement_done"]
    }
    updated_state["workflow_history"].append(history_entry)
    return updated_state

@log_aspect
def knowledge_base_node(state: JurixState) -> JurixState:
    # Always perform evaluation, regardless of previous refinement status
    input_data = {
        "article": state["article"]
    }
    result = knowledge_base.run(input_data)
    updated_state = state.copy()
    updated_state["redundant"] = result.get("redundant", False)
    
    # Only set refinement suggestion if we haven't refined yet
    if not state.get("has_refined", False):
        # Force a refinement suggestion for the first evaluation
        if not result.get("refinement_suggestion"):
            logging.warning("KB agent didn't provide refinement suggestion, forcing one")
            updated_state["refinement_suggestion"] = "Add more specific metrics and technical implementation details to the Key Improvements section."
        else:
            updated_state["refinement_suggestion"] = result.get("refinement_suggestion")
        updated_state["refinement_count"] = 1
    else:
        # Clear the refinement suggestion after the article has been refined
        updated_state["refinement_suggestion"] = None
        updated_state["refinement_count"] = state.get("refinement_count", 0)
    
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
    updated_state["workflow_stage"] = "knowledge_base_evaluated"
    
    history_entry = {
        "step": "knowledge_base_evaluation" if not state.get("has_refined", False) else "final_knowledge_base_evaluation",
        "article": updated_state["article"],
        "redundant": updated_state.get("redundant", False),
        "refinement_suggestion": updated_state["refinement_suggestion"],
        "workflow_status": updated_state["workflow_status"],
        "workflow_stage": updated_state["workflow_stage"],
        "recommendation_id": updated_state["recommendation_id"],
        "recommendations": updated_state["recommendations"],
        "autonomous_refinement_done": updated_state.get("autonomous_refinement_done", False)
    }
    updated_state["workflow_history"].append(history_entry)
    return updated_state