# orchestrator/core/jira_workflow_orchestrator.py
# ENHANCED VERSION - Now with universal collaboration support!

from datetime import datetime
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

# NEW: Import universal collaboration coordinator
from orchestrator.core.universal_collaboration_coordinator import UniversalCollaborationCoordinator # type: ignore

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logging.basicConfig(filename='jira_workflow.log', level=logging.INFO)
        logging.info(f"Executing {func.__name__} with state: {state}")
        result = func(state)
        logging.info(f"Completed {func.__name__} with result: {result}")
        return result
    return wrapper

# Initialize components
shared_memory = JurixSharedMemory()
jira_article_generator = JiraArticleGeneratorAgent(shared_memory)
knowledge_base = KnowledgeBaseAgent(shared_memory)
recommendation_agent = RecommendationAgent(shared_memory)
jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)

# NEW: Create agents registry for collaboration
agents_registry = {
    "jira_article_generator_agent": jira_article_generator,
    "knowledge_base_agent": knowledge_base,
    "recommendation_agent": recommendation_agent,
    "jira_data_agent": jira_data_agent
}

# NEW: Initialize universal collaboration coordinator
collaboration_coordinator = UniversalCollaborationCoordinator(
    shared_memory.redis_client,
    agents_registry
)

logging.info("🎭 Enhanced Jira Workflow initialized with Universal Collaboration")

def store_recommendations(ticket_id: str, recommendations: List[str]) -> str:
    recommendation_id = f"rec_{ticket_id}_{str(uuid.uuid4())}"
    shared_memory.store(recommendation_id, {"ticket_id": ticket_id, "recommendations": recommendations})
    logging.info(f"Stored recommendations with ID {recommendation_id}: {recommendations}")
    return recommendation_id

def run_recommendation_agent(state: Dict[str, Any]) -> tuple[str, List[str]]:
    """ENHANCED: Now with collaboration support"""
    logging.info(f"Fetching ticket data for recommendation agent for ticket {state['ticket_id']}")
    
    project_id = state.get("project", "PROJ123")
    
    # NEW: Use collaboration coordinator for data retrieval
    jira_input = {
        "project_id": project_id,
        "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"},
        "collaboration_context": "recommendation_support",
        "workflow_context": "jira_article_generation"
    }
    
    try:
        # Get enhanced ticket data through collaboration
        ticket_data_result = collaboration_coordinator.enhance_agent_call(
            "jira_data_agent", 
            jira_input,
            {"step": "data_for_recommendations", "ticket_id": state['ticket_id']}
        )
        
        tickets = ticket_data_result.get("tickets", [])
        logging.info(f"Retrieved {len(tickets)} tickets for project {project_id} (collaborative mode)")
        
        # Create enhanced input for recommendation agent
        input_data = {
            "session_id": state.get("conversation_id", f"rec_{state['ticket_id']}"),
            "user_prompt": f"Analyze resolved ticket {state['ticket_id']} and provide specific action items",
            "articles": [state.get("article", {})] if state.get("article") else [],
            "project": project_id,
            "tickets": tickets,
            "workflow_type": "knowledge_base_creation",
            "intent": {"intent": "ticket_recommendations"},
            "collaboration_context": "article_enhancement",  # NEW: Collaboration context
            "primary_agent_result": ticket_data_result  # NEW: Pass data agent result
        }
        
        logging.info(f"Calling recommendation agent with {len(tickets)} tickets (collaborative mode)...")
        
        # NEW: Use collaboration coordinator
        result = collaboration_coordinator.enhance_agent_call(
            "recommendation_agent",
            input_data,
            {"step": "recommendation_generation", "ticket_id": state['ticket_id']}
        )
        
        recommendations = result.get("recommendations", [])
        
        # Enhanced fallback with context awareness
        if not recommendations or (len(recommendations) == 1 and "provide more project-specific details" in recommendations[0].lower()):
            logging.warning("Recommendation agent returned a request for more context. Trying with enhanced ticket-specific data.")
            
            # Find the specific ticket data and enhance the prompt
            target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
            if target_ticket:
                ticket_fields = target_ticket.get("fields", {})
                enriched_prompt = (
                    f"Analyze resolved ticket {state['ticket_id']} - {ticket_fields.get('summary', 'No summary')}. "
                    f"Status: {ticket_fields.get('status', {}).get('name', 'Unknown')}. "
                    f"Consider how the resolution of this ticket can benefit other parts of the project. "
                    f"Provide specific, actionable recommendations based on this resolution."
                )
                input_data["user_prompt"] = enriched_prompt
                
                # Retry with enhanced collaboration context
                input_data["collaboration_context"] = "enhanced_ticket_analysis"
                result = collaboration_coordinator.enhance_agent_call(
                    "recommendation_agent",
                    input_data,
                    {"step": "enhanced_recommendation_retry", "ticket_id": state['ticket_id']}
                )
                recommendations = result.get("recommendations", [])
        
        # Final fallback with intelligent defaults
        if not recommendations:
            logging.warning("Recommendation agent still couldn't generate recommendations. Using enhanced fallback options.")
            target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
            if target_ticket:
                ticket_summary = target_ticket.get("fields", {}).get("summary", "this ticket")
                recommendations = [
                    f"Apply the solution pattern from {state['ticket_id']} ({ticket_summary}) to similar issues in the project.",
                    f"Schedule a knowledge-sharing session about the resolution approach used in {state['ticket_id']}.",
                    f"Update project documentation to reflect the solution methodology from {state['ticket_id']}.",
                    f"Create a reusable component or template based on the {state['ticket_id']} resolution."
                ]
            else:
                recommendations = [
                    "Review similar tickets to apply this solution pattern.",
                    "Document this resolution in the team knowledge base.",
                    "Consider creating a reusable component from this solution."
                ]
    
    except Exception as e:
        logging.error(f"Error in enhanced recommendation agent: {str(e)}")
        recommendations = [
            f"Review similar tickets to apply this solution pattern from {state.get('ticket_id', 'this ticket')}.",
            "Document this resolution methodology in the team knowledge base.",
            "Consider creating a reusable component or template from this solution.",
            "Schedule team discussion about applying this resolution approach to similar issues."
        ]
    
    recommendation_id = store_recommendations(state["ticket_id"], recommendations)
    return recommendation_id, recommendations

def run_jira_workflow(ticket_id: str, conversation_id: str = None, project_id: str = "PROJ123") -> JurixState:
    """ENHANCED: Run Jira workflow with universal collaboration"""
    conversation_id = conversation_id or str(uuid.uuid4())
    
    # Create enhanced state with collaboration support
    state = JurixState(
        query=f"Generate article for ticket {ticket_id}",
        intent={"intent": "article_generation", "project": project_id},
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
        autonomous_refinement_done=False,
        # NEW: Collaboration fields
        collaboration_metadata=None,
        final_collaboration_summary=None,
        collaboration_insights=None,
        collaboration_trace=None,
        collaborative_agents_used=None
    )
    
    logging.info(f"🚀 Starting ENHANCED Jira workflow with collaboration for ticket {ticket_id}")
    
    # Run enhanced recommendation agent in parallel
    logging.info(f"Running enhanced recommendation agent for ticket {ticket_id}, project {project_id}")
    recommendation_id, recommendations = run_recommendation_agent(state)
    state["recommendation_id"] = recommendation_id
    state["recommendations"] = recommendations
    logging.info(f"Enhanced RecommendationAgent completed, ID: {recommendation_id}, recommendations: {recommendations}")

    workflow = build_jira_workflow()
    final_state = state
    collaboration_trace = []
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            logging.info(f"Event from {node_name}: {node_state}")
            
            # Track collaboration metadata
            collab_metadata = node_state.get("collaboration_metadata", {})
            if collab_metadata:
                collaboration_trace.append({
                    "node": node_name,
                    "collaboration": collab_metadata,
                    "timestamp": datetime.now().isoformat()
                })
                logging.info(f"🤝 {node_name} generated collaboration: {collab_metadata}")
            
            final_state = node_state
            
            # Update recommendations if they were placeholder and we now have article context
            if (node_name == "jira_article_generator" and 
                final_state.get("article") and 
                "provide more project-specific details" in str(final_state.get("recommendations", []))):
                logging.info("Updating recommendations with article context using collaboration")
                final_state_dict = dict(final_state)
                final_state_dict["article"] = final_state.get("article", {})
                recommendation_id, recommendations = run_recommendation_agent(final_state_dict)
                final_state["recommendation_id"] = recommendation_id
                final_state["recommendations"] = recommendations
    
    # Ensure final state has collaboration metadata
    if not final_state.get("collaboration_metadata") and collaboration_trace:
        logging.info("🚨 Reconstructing collaboration metadata from trace")
        final_state = collaboration_coordinator.fix_missing_collaboration(final_state)
    
    # Add collaboration trace
    final_state["collaboration_trace"] = collaboration_trace
    
    logging.info(f"🎉 Enhanced Jira workflow completed with collaboration")
    logging.info(f"Final state after stream: {final_state}")
    return final_state

def build_jira_workflow():
    """Build enhanced workflow with collaboration support"""
    workflow = StateGraph(JurixState)
    
    # Use enhanced collaborative nodes
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
                "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                "collaboration_metadata": state.get("collaboration_metadata", {})
            })
            return END
        
        if state.get("workflow_stage") in ["article_generated", "article_refined"]:
            return "knowledge_base"
        
        elif state.get("workflow_stage") == "knowledge_base_evaluated":
            if state.get("refinement_suggestion") and not state.get("has_refined", False):
                logging.info("Refinement needed, going back to article generator")
                return "jira_article_generator"
            else:
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
                    "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                    "collaboration_metadata": state.get("collaboration_metadata", {})
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
                "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                "collaboration_metadata": state.get("collaboration_metadata", {})
            })
        
        return END

    workflow.add_conditional_edges("jira_article_generator", route)
    workflow.add_conditional_edges("knowledge_base", route)

    return workflow.compile()

@log_aspect
def jira_article_generator_node(state: JurixState) -> JurixState:
    """ENHANCED: Now with collaboration support"""
    input_data = {
        "ticket_id": state["ticket_id"],
        "refinement_suggestion": state.get("refinement_suggestion"),
        "collaboration_context": "article_generation",
        "recommendations": state.get("recommendations", [])  # NEW: Provide recommendations context
    }
    
    # NEW: Use collaboration coordinator
    result = collaboration_coordinator.coordinate_workflow_step(
        primary_agent="jira_article_generator_agent",
        input_data=input_data,
        workflow_id=state.get("conversation_id", "jira_workflow"),
        step_name="article_generation"
    )
    
    updated_state = state.copy()
    updated_state["article"] = result.get("article", {})
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["autonomous_refinement_done"] = result.get("autonomous_refinement_done", False)
    
    # NEW: Preserve collaboration metadata
    if result.get("collaboration_metadata"):
        updated_state["collaboration_metadata"] = result["collaboration_metadata"]
        logging.info(f"🤝 Article generator: Collaboration metadata preserved")
    
    # Track refinement status
    if state.get("refinement_suggestion") is not None:
        updated_state["has_refined"] = True
        updated_state["workflow_stage"] = "article_refined"
    else:
        updated_state["has_refined"] = False
        updated_state["workflow_stage"] = "article_generated"
    
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
        "autonomous_refinement_done": updated_state["autonomous_refinement_done"],
        "collaboration_metadata": updated_state.get("collaboration_metadata", {})
    }
    updated_state["workflow_history"].append(history_entry)
    return updated_state

@log_aspect
def knowledge_base_node(state: JurixState) -> JurixState:
    """ENHANCED: Now with collaboration support"""
    input_data = {
        "article": state["article"],
        "collaboration_context": "article_evaluation",
        "primary_agent_result": {  # NEW: Provide context from article generation
            "article": state["article"],
            "recommendations": state.get("recommendations", []),
            "ticket_id": state.get("ticket_id")
        }
    }
    
    # NEW: Use collaboration coordinator
    result = collaboration_coordinator.coordinate_workflow_step(
        primary_agent="knowledge_base_agent",
        input_data=input_data,
        workflow_id=state.get("conversation_id", "jira_workflow"),
        step_name="article_evaluation"
    )
    
    updated_state = state.copy()
    updated_state["redundant"] = result.get("redundant", False)
    
    # Handle refinement suggestions
    if not state.get("has_refined", False):
        if not result.get("refinement_suggestion"):
            logging.warning("KB agent didn't provide refinement suggestion, forcing one")
            updated_state["refinement_suggestion"] = "Add more specific metrics and technical implementation details to the Key Improvements section."
        else:
            updated_state["refinement_suggestion"] = result.get("refinement_suggestion")
        updated_state["refinement_count"] = 1
    else:
        updated_state["refinement_suggestion"] = None
        updated_state["refinement_count"] = state.get("refinement_count", 0)
    
    updated_state["workflow_status"] = result.get("workflow_status", "failure")
    updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
    updated_state["workflow_stage"] = "knowledge_base_evaluated"
    
    # NEW: Merge collaboration metadata
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    if existing_collab or new_collab:
        merged_collab = merge_collaboration_metadata(existing_collab, new_collab)
        merged_collab["final_evaluation_complete"] = True
        updated_state["collaboration_metadata"] = merged_collab
        updated_state["final_collaboration_summary"] = merged_collab  # Backup
        logging.info(f"🤝 Knowledge base: Final collaboration metadata merged")
    
    history_entry = {
        "step": "knowledge_base_evaluation" if not state.get("has_refined", False) else "final_knowledge_base_evaluation",
        "article": updated_state["article"],
        "redundant": updated_state.get("redundant", False),
        "refinement_suggestion": updated_state["refinement_suggestion"],
        "workflow_status": updated_state["workflow_status"],
        "workflow_stage": updated_state["workflow_stage"],
        "recommendation_id": updated_state["recommendation_id"],
        "recommendations": updated_state["recommendations"],
        "autonomous_refinement_done": updated_state.get("autonomous_refinement_done", False),
        "collaboration_metadata": updated_state.get("collaboration_metadata", {})
    }
    updated_state["workflow_history"].append(history_entry)
    return updated_state

def merge_collaboration_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to merge collaboration metadata across workflow steps"""
    if not existing:
        return new.copy() if new else {}
    if not new:
        return existing.copy()
    
    merged = existing.copy()
    
    # Merge agent lists
    existing_agents = set(merged.get("collaborating_agents", []))
    new_agents = set(new.get("collaborating_agents", []))
    merged["collaborating_agents"] = list(existing_agents | new_agents)
    
    # Merge collaboration types
    existing_types = set(merged.get("collaboration_types", []))
    new_types = set(new.get("collaboration_types", []))
    merged["collaboration_types"] = list(existing_types | new_types)
    
    # Update metadata while preserving important fields
    merged.update(new)
    merged["total_workflow_collaborations"] = len(merged["collaborating_agents"])
    merged["workflow_enhanced"] = True
    
    return merged

# NEW: Enhanced wrapper function for easy integration
def run_collaborative_jira_workflow(ticket_id: str, conversation_id: str = None, project_id: str = "PROJ123") -> JurixState:
    """
    Enhanced wrapper that guarantees collaboration for Jira workflows
    Use this function for maximum collaboration features
    """
    enhanced_workflow = collaboration_coordinator.integrate_with_existing_workflow(run_jira_workflow)
    return enhanced_workflow(ticket_id, conversation_id, project_id)