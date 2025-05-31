# orchestrator/core/orchestrator.py
# FIXED VERSION - Now uses the corrected collaborative framework!

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
from orchestrator.core.collaborative_framework import CollaborativeFramework  # type: ignore # FIXED VERSION
import functools
import logging
import uuid
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FixedOrchestrator")

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

# Initialize FIXED collaborative framework
collaborative_framework = CollaborativeFramework(shared_memory.redis_client, agents_registry)

logger.info("🎭 FIXED Orchestrator initialized with corrected collaboration framework")

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logger.info(f"🎯 Executing {func.__name__}")
        result = func(state)
        
        # Enhanced logging for collaboration metadata
        collab_metadata = result.get("collaboration_metadata", {})
        if collab_metadata:
            articles_retrieved = collab_metadata.get("articles_retrieved", 0)
            articles_merged = collab_metadata.get("articles_merged", False)
            logger.info(f"🤝 {func.__name__} COLLABORATION: {collab_metadata}")
            logger.info(f"📚 Articles retrieved: {articles_retrieved}, merged: {articles_merged}")
        else:
            logger.warning(f"⚠️ {func.__name__} NO collaboration metadata found")
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

def merge_collaboration_metadata(existing_state: JurixState, new_collab: Dict) -> Dict:
    """ENHANCED merge function with article tracking"""
    existing_collab = existing_state.get("collaboration_metadata", {})
    
    if not new_collab:
        return existing_collab
    
    if not existing_collab:
        return new_collab.copy()
    
    # Merge the metadata intelligently
    merged = existing_collab.copy()
    
    # Merge collaborating agents lists
    existing_agents = set(merged.get("collaborating_agents", []))
    new_agents = set(new_collab.get("collaborating_agents", []))
    all_agents = list(existing_agents | new_agents)
    
    # Merge collaboration types
    existing_types = set(merged.get("collaboration_types", []))
    new_types = set(new_collab.get("collaboration_types", []))
    all_types = list(existing_types | new_types)
    
    # CRITICAL FIX: Merge article counts and status
    merged["articles_retrieved"] = merged.get("articles_retrieved", 0) + new_collab.get("articles_retrieved", 0)
    merged["articles_merged"] = merged.get("articles_merged", False) or new_collab.get("articles_merged", False)
    
    # Update with new data, preserving the merge
    merged.update(new_collab)
    merged["collaborating_agents"] = all_agents
    merged["collaboration_types"] = all_types
    merged["total_collaborations"] = len(all_agents)
    merged["workflow_collaboration_complete"] = True
    
    logger.info(f"🔗 MERGED collaboration metadata: {len(all_agents)} agents, {merged['articles_retrieved']} articles")
    return merged

@log_aspect
def collaborative_data_node(state: JurixState) -> JurixState:
    """FIXED DATA RETRIEVAL with proper collaboration"""
    project = state.get("project")
    if not project:
        logger.warning("No project specified for data retrieval")
        updated_state = state.copy()
        return updated_state
    
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
            logger.info(f"🚀 Starting FIXED collaborative data retrieval for {project}")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("jira_data_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    logger.info(f"📊 FIXED data collaboration result keys: {list(result.keys())}")
    
    # Create new state and preserve collaboration metadata
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    
    # FIXED: Properly merge collaboration metadata with article tracking
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(updated_state, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Data node: STORED collaboration metadata")
    
    return updated_state

@log_aspect
def collaborative_recommendation_node(state: JurixState) -> JurixState:
    """FIXED RECOMMENDATION GENERATION with proper article handling"""
    def run_collaboration():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "articles": state.get("articles", []),  # Pass existing articles
            "project": state["project"],
            "tickets": state["tickets"],
            "workflow_type": "collaborative_orchestration",
            "intent": state["intent"]
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("🎯 Starting FIXED collaborative recommendation generation")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("recommendation_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    logger.info(f"💡 FIXED recommendation collaboration result keys: {list(result.keys())}")
    
    # CRITICAL FIX: Preserve articles from collaboration
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["needs_context"] = result.get("needs_context", False)
    
    # CRITICAL FIX: Ensure articles are preserved in the state
    if result.get("articles"):
        updated_state["articles"] = result["articles"]
        logger.info(f"🎉 FIXED: {len(result['articles'])} articles preserved in recommendation node!")
    
    if result.get("articles_from_collaboration"):
        updated_state["articles_from_collaboration"] = result["articles_from_collaboration"]
        logger.info(f"🎉 FIXED: {len(result['articles_from_collaboration'])} articles from collaboration!")
    
    # Merge collaboration metadata
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(updated_state, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        
        articles_count = len(updated_state.get("articles", []))
        logger.info(f"🎼 Recommendation node: MERGED collaboration metadata with {articles_count} articles")
    
    return updated_state

@log_aspect
def collaborative_retrieval_node(state: JurixState) -> JurixState:
    """FIXED ARTICLE RETRIEVAL with proper metadata handling"""
    def run_collaboration():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "intent": state["intent"]
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("📚 Starting FIXED collaborative article retrieval")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("retrieval_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    logger.info(f"📖 FIXED retrieval collaboration result keys: {list(result.keys())}")
    
    # Preserve collaboration metadata
    updated_state = state.copy()
    updated_state["articles"] = result.get("articles", [])
    
    # Merge collaboration metadata
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(updated_state, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Retrieval node: MERGED collaboration metadata")
    
    return updated_state

@log_aspect
def collaborative_chat_node(state: JurixState) -> JurixState:
    """FIXED RESPONSE GENERATION with complete article preservation"""
    def run_collaboration():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "articles": state.get("articles", []),  # CRITICAL: Pass articles from state
            "recommendations": state["recommendations"],
            "tickets": state["tickets"],
            "intent": state["intent"]
        }
        
        # Log article availability for debugging
        articles_count = len(task_context.get("articles", []))
        logger.info(f"💬 Chat node starting with {articles_count} articles")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("💬 Starting FIXED collaborative response generation")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("chat_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    logger.info(f"💬 FIXED chat collaboration result keys: {list(result.keys())}")
    
    # Create final state with complete preservation
    updated_state = state.copy()
    
    # Update all the regular fields
    updated_state.update({
        "response": result.get("response", "No response generated"),
        "conversation_history": chat_agent.shared_memory.get_conversation(state["conversation_id"]),
        "articles_used": result.get("articles_used", []),
        "tickets": result.get("tickets", state["tickets"]),
        "workflow_status": result.get("workflow_status", "completed")
    })
    
    # CRITICAL FIX: Ensure articles are preserved throughout
    if state.get("articles"):
        updated_state["articles"] = state["articles"]
        logger.info(f"🎉 FINAL: Preserved {len(state['articles'])} articles from previous steps")
    
    if result.get("articles"):
        updated_state["articles"] = result["articles"]  
        logger.info(f"🎉 FINAL: Using {len(result['articles'])} articles from chat collaboration")
    
    # Final collaboration metadata merge
    new_collab = result.get("collaboration_metadata", {})
    final_collab = merge_collaboration_metadata(updated_state, new_collab)
    
    if final_collab:
        # Add final workflow completion information
        final_collab["workflow_completed"] = True
        final_collab["final_agent"] = "chat_agent"
        final_collab["final_state_preserved"] = True
        final_collab["total_workflow_agents"] = len(final_collab.get("collaborating_agents", []))
        final_collab["final_articles_count"] = len(updated_state.get("articles", []))
        
        # Store in BOTH fields to ensure it's captured
        updated_state["collaboration_metadata"] = final_collab
        updated_state["final_collaboration_summary"] = final_collab
        
        logger.info(f"🎉 FINAL CHAT NODE: Complete collaboration metadata with {final_collab.get('final_articles_count', 0)} articles")
    
    return updated_state

def build_workflow():
    """Build workflow with FIXED collaborative intelligence"""
    workflow = StateGraph(JurixState)
    
    # Use the FIXED collaborative nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("jira_data_agent", collaborative_data_node)
    workflow.add_node("recommendation_agent", collaborative_recommendation_node)
    workflow.add_node("retrieval_agent", collaborative_retrieval_node)
    workflow.add_node("chat_agent", collaborative_chat_node)

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
    """FIXED workflow runner with guaranteed article preservation"""
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
        project=None,
        collaboration_metadata=None,
        final_collaboration_summary=None,
        collaboration_insights=None,
        collaboration_trace=None,
        collaborative_agents_used=None
    )
    
    workflow = build_workflow()
    final_state = None
    
    logger.info(f"🚀 Starting FIXED collaborative workflow for: '{query}'")
    
    # Enhanced collaboration tracking
    collaboration_trace = []
    articles_tracking = []
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            # Track collaboration metadata AND articles through each step
            collab_info = node_state.get("collaboration_metadata", {})
            articles_count = len(node_state.get("articles", []))
            
            if collab_info:
                logger.info(f"🎼 {node_name} collaboration: {collab_info}")
                collaboration_trace.append({
                    "node": node_name,
                    "collaboration": collab_info,
                    "articles_count": articles_count
                })
            
            articles_tracking.append({
                "node": node_name,
                "articles_count": articles_count
            })
            
            logger.info(f"📚 {node_name}: {articles_count} articles in state")
            final_state = node_state
    
    # CRITICAL FIX: Verify final state has articles
    if final_state:
        final_articles = len(final_state.get("articles", []))
        final_collab = final_state.get("collaboration_metadata", {})
        
        logger.info(f"🎉 WORKFLOW COMPLETE:")
        logger.info(f"   📚 Final articles: {final_articles}")
        logger.info(f"   🤝 Collaboration: {bool(final_collab)}")
        
        if final_collab:
            articles_retrieved = final_collab.get("articles_retrieved", 0)
            articles_merged = final_collab.get("articles_merged", False)
            logger.info(f"   📊 Articles retrieved: {articles_retrieved}, merged: {articles_merged}")
        
        # Add tracking data for debugging
        final_state["collaboration_trace"] = collaboration_trace
        final_state["articles_tracking"] = articles_tracking
        
        # Emergency article recovery if needed
        if final_articles == 0 and any(track["articles_count"] > 0 for track in articles_tracking):
            logger.warning("🚨 EMERGENCY: Articles were lost during workflow - attempting recovery")
            max_articles_step = max(articles_tracking, key=lambda x: x["articles_count"])
            logger.info(f"   📊 Max articles were at {max_articles_step['node']}: {max_articles_step['articles_count']}")
    
    return final_state or state

# Enhanced: Get collaboration insights
def get_collaboration_insights() -> Dict:
    """Get insights about collaborative performance"""
    return collaborative_framework.get_collaboration_insights()

def test_collaboration_metadata_persistence(query: str = "Give me recommendations for PROJ123") -> Dict:
    """ENHANCED test function to verify complete article handling"""
    logger.info(f"🧪 TESTING COMPLETE collaboration system with query: {query}")
    
    result = run_workflow(query)
    
    test_results = {
        "query": query,
        "has_collaboration_metadata": "collaboration_metadata" in result,
        "has_backup_metadata": "final_collaboration_summary" in result,
        "articles_in_final_state": len(result.get("articles", [])),
        "collaboration_data": result.get("collaboration_metadata", {}),
        "articles_tracking": result.get("articles_tracking", []),
        "collaboration_trace": result.get("collaboration_trace", []),
        "all_result_keys": list(result.keys()),
        "success_indicators": {
            "workflow_completed": bool(result.get("response")),
            "collaboration_occurred": bool(result.get("collaboration_metadata")),
            "articles_present": len(result.get("articles", [])) > 0,
            "recommendations_present": len(result.get("recommendations", [])) > 0
        }
    }
    
    logger.info(f"🧪 ENHANCED TEST RESULTS: {test_results['success_indicators']}")
    return test_results