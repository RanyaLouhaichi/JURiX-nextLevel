# orchestrator/core/orchestrator.py
# FIXED VERSION - Now actually uses collaborative framework!

import sys
import os
from typing import Dict
from langgraph.graph import StateGraph, END
from orchestrator.graph.state import JurixState # type: ignore
from .intent_router import classify_intent
from agents.chat_agent import ChatAgent
from agents.retrieval_agent import RetrievalAgent
from agents.recommendation_agent import RecommendationAgent
from agents.jira_data_agent import JiraDataAgent
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.collaborative_framework import CollaborativeFramework  # type: ignore # NOW ACTUALLY USED!
import functools
import logging
import uuid
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedOrchestrator")

# Initialize components
shared_memory = JurixSharedMemory()
chat_agent = ChatAgent(shared_memory)
retrieval_agent = RetrievalAgent(shared_memory)
recommendation_agent = RecommendationAgent(shared_memory)
jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)

# Create agents registry for collaboration
agents_registry = {
    "chat_agent": chat_agent,
    "retrieval_agent": retrieval_agent, 
    "recommendation_agent": recommendation_agent,
    "jira_data_agent": jira_data_agent
}

# INITIALIZE COLLABORATIVE FRAMEWORK - This was missing!
collaborative_framework = CollaborativeFramework(shared_memory.redis_client, agents_registry)

logger.info("🎭 Orchestrator initialized with REAL collaborative intelligence")

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logger.info(f"🎯 Executing {func.__name__}")
        result = func(state)
        
        # Log collaboration metadata
        collab_metadata = result.get("collaboration_metadata", {})
        if collab_metadata:
            logger.info(f"🤝 Collaboration: {collab_metadata}")
        return result
    return wrapper

@log_aspect
def classify_intent_node(state: JurixState) -> JurixState:
    """Intent classification - no collaboration needed here"""
    intent_result = classify_intent(state["query"], state["conversation_history"])
    updated_state = state.copy()
    updated_state["intent"] = intent_result
    updated_state["project"] = intent_result.get("project")
    return updated_state

# Fix for orchestrator/core/orchestrator.py
# The problem is that collaboration_metadata gets lost in state updates
# Here's how to fix each collaborative node:

@log_aspect
def collaborative_data_node(state: JurixState) -> JurixState:
    """COLLABORATIVE DATA RETRIEVAL - FIXED"""
    project = state.get("project")
    if not project:
        logger.warning("No project specified for data retrieval")
        return state.copy()
    
    def run_collaboration():
        task_context = {
            "project_id": project,
            "time_range": {
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-17T23:59:59Z"
            },
            "user_query": state["query"],
            "analysis_depth": "enhanced"
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(f"🚀 Starting collaborative data retrieval for {project}")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("jira_data_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    
    # CRITICAL FIX: Properly preserve collaboration metadata
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    
    # FIXED: Make sure collaboration_metadata persists
    collab_meta = result.get("collaboration_metadata", {})
    if collab_meta:
        updated_state["collaboration_metadata"] = collab_meta
        logger.info(f"🎼 Data agent collaboration metadata stored: {collab_meta}")
    else:
        logger.warning("⚠️ No collaboration metadata from data agent")
    
    return updated_state

@log_aspect
def collaborative_recommendation_node(state: JurixState) -> JurixState:
    """COLLABORATIVE RECOMMENDATION GENERATION - FIXED"""
    def run_collaboration():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "articles": state["articles"],
            "project": state["project"],
            "tickets": state["tickets"],
            "workflow_type": "collaborative_orchestration",
            "intent": state["intent"]
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("🎯 Starting collaborative recommendation generation")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("recommendation_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    
    # CRITICAL FIX: Preserve and merge collaboration metadata
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["needs_context"] = result.get("needs_context", False)
    
    # FIXED: Merge collaboration metadata from multiple agents
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    
    if new_collab:
        # Merge collaboration metadata intelligently
        merged_collab = existing_collab.copy() if existing_collab else {}
        
        # Merge collaborating agents lists
        existing_agents = merged_collab.get("collaborating_agents", [])
        new_agents = new_collab.get("collaborating_agents", [])
        all_agents = list(set(existing_agents + new_agents))
        
        merged_collab.update(new_collab)
        merged_collab["collaborating_agents"] = all_agents
        merged_collab["total_collaborations"] = len(all_agents)
        
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Recommendation agent collaboration metadata merged: {merged_collab}")
    
    return updated_state

@log_aspect
def collaborative_retrieval_node(state: JurixState) -> JurixState:
    """COLLABORATIVE ARTICLE RETRIEVAL - FIXED"""
    def run_collaboration():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "intent": state["intent"]
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("📚 Starting collaborative article retrieval")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("retrieval_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    
    # CRITICAL FIX: Preserve collaboration metadata
    updated_state = state.copy()
    updated_state["articles"] = result.get("articles", [])
    
    # FIXED: Merge collaboration metadata
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    
    if new_collab:
        merged_collab = existing_collab.copy() if existing_collab else {}
        merged_collab.update(new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Retrieval agent collaboration metadata stored: {merged_collab}")
    
    return updated_state

@log_aspect
def collaborative_chat_node(state: JurixState) -> JurixState:
    """COLLABORATIVE RESPONSE GENERATION - FIXED"""
    def run_collaboration():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "articles": state["articles"],
            "recommendations": state["recommendations"],
            "tickets": state["tickets"],
            "intent": state["intent"]
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("💬 Starting collaborative response generation")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("chat_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    
    # CRITICAL FIX: Preserve ALL collaboration metadata in final state
    updated_state = state.copy()
    updated_state.update({
        "response": result.get("response", "No response generated"),
        "conversation_history": chat_agent.shared_memory.get_conversation(state["conversation_id"]),
        "articles_used": result.get("articles_used", []),
        "tickets": result.get("tickets", state["tickets"]),
        "workflow_status": result.get("workflow_status", "completed")
    })
    
    # FIXED: Merge final collaboration metadata
    existing_collab = updated_state.get("collaboration_metadata", {})
    new_collab = result.get("collaboration_metadata", {})
    
    if new_collab or existing_collab:
        # Create comprehensive final collaboration summary
        final_collab = existing_collab.copy() if existing_collab else {}
        if new_collab:
            final_collab.update(new_collab)
        
        # Add summary information
        final_collab["workflow_completed"] = True
        final_collab["final_agent"] = "chat_agent"
        
        updated_state["collaboration_metadata"] = final_collab
        updated_state["final_collaboration_summary"] = final_collab  # Also store as summary
        
        logger.info(f"🎉 FINAL collaboration summary stored: {final_collab}")
    else:
        logger.warning("⚠️ No collaboration metadata found in final state")
    
    return updated_state

def build_workflow():
    """Build workflow with REAL collaborative intelligence"""
    workflow = StateGraph(JurixState)
    
    # Use the FIXED collaborative nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("jira_data_agent", collaborative_data_node)  # FIXED
    workflow.add_node("recommendation_agent", collaborative_recommendation_node)  # FIXED
    workflow.add_node("retrieval_agent", collaborative_retrieval_node)  # FIXED
    workflow.add_node("chat_agent", collaborative_chat_node)  # FIXED

    workflow.set_entry_point("classify_intent")

    def route(state: JurixState) -> str:
        intent = state["intent"]["intent"] if "intent" in state and "intent" in state["intent"] else "generic_question"
        needs_context = state.get("needs_context", False)
        
        logger.info(f"🎯 Routing with intent: {intent}")
        
        if needs_context:
            return "chat_agent"
        
        routing = {
            "generic_question": "chat_agent",
            "follow_up": "chat_agent",
            "article_retrieval": "retrieval_agent", 
            "recommendation": "jira_data_agent"
        }
        return routing.get(intent, "chat_agent")

    workflow.add_conditional_edges("classify_intent", route)
    workflow.add_edge("jira_data_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "chat_agent")
    workflow.add_edge("retrieval_agent", "chat_agent")
    workflow.add_edge("chat_agent", END)

    return workflow.compile()

def run_workflow(query: str, conversation_id: str = None) -> JurixState:
    """ENHANCED workflow runner with real collaborative intelligence"""
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
    
    logger.info(f"🚀 Starting REAL collaborative workflow for: '{query}'")
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            collab_info = node_state.get("collaboration_metadata", {})
            if collab_info:
                logger.info(f"🎼 {node_name} collaboration: {collab_info}")
            final_state = node_state
    
    # Log final results
    if final_state:
        final_collab = final_state.get("final_collaboration_summary", {})
        if final_collab:
            logger.info(f"🎉 Final collaboration summary: {final_collab}")
    
    return final_state or state

# NEW: Get collaboration insights
def get_collaboration_insights() -> Dict:
    """Get insights about collaborative performance"""
    return collaborative_framework.get_collaboration_insights()