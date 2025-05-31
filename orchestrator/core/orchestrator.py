# orchestrator/core/orchestrator.py
# FIXED VERSION - Now properly preserves collaboration metadata throughout the workflow!

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
from orchestrator.core.collaborative_framework import CollaborativeFramework  # type: ignore
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

# Initialize collaborative framework
collaborative_framework = CollaborativeFramework(shared_memory.redis_client, agents_registry)

logger.info("🎭 FIXED Orchestrator initialized with proper collaboration metadata persistence")

def log_aspect(func):
    @functools.wraps(func)
    def wrapper(state):
        logger.info(f"🎯 Executing {func.__name__}")
        result = func(state)
        
        # Log collaboration metadata with detailed tracking
        collab_metadata = result.get("collaboration_metadata", {})
        if collab_metadata:
            logger.info(f"🤝 {func.__name__} COLLABORATION DETECTED: {collab_metadata}")
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

# CRITICAL FIX: Properly merge collaboration metadata across all nodes
def merge_collaboration_metadata(existing_state: JurixState, new_collab: Dict) -> Dict:
    """Intelligently merge collaboration metadata from multiple agents"""
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
    
    # Update with new data, preserving the merge
    merged.update(new_collab)
    merged["collaborating_agents"] = all_agents
    merged["collaboration_types"] = all_types
    merged["total_collaborations"] = len(all_agents)
    merged["workflow_collaboration_complete"] = True
    
    logger.info(f"🔗 MERGED collaboration metadata: {len(all_agents)} agents, {len(all_types)} types")
    return merged

@log_aspect
def collaborative_data_node(state: JurixState) -> JurixState:
    """COLLABORATIVE DATA RETRIEVAL - FIXED with proper metadata persistence"""
    project = state.get("project")
    if not project:
        logger.warning("No project specified for data retrieval")
        updated_state = state.copy()
        # Even if no project, preserve any existing collaboration metadata
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
            logger.info(f"🚀 Starting collaborative data retrieval for {project}")
            return loop.run_until_complete(
                collaborative_framework.coordinate_agents("jira_data_agent", task_context)
            )
        finally:
            loop.close()
    
    result = run_collaboration()
    logger.info(f"📊 Data collaboration result keys: {list(result.keys())}")
    
    # CRITICAL FIX: Create new state and properly preserve collaboration metadata
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    
    # FIXED: Properly merge collaboration metadata
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(updated_state, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Data node: STORED collaboration metadata with {len(merged_collab.get('collaborating_agents', []))} agents")
    else:
        logger.warning("⚠️ Data node: No collaboration metadata from framework")
    
    return updated_state

@log_aspect
def collaborative_recommendation_node(state: JurixState) -> JurixState:
    """COLLABORATIVE RECOMMENDATION GENERATION - FIXED with metadata preservation"""
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
    logger.info(f"💡 Recommendation collaboration result keys: {list(result.keys())}")
    
    # CRITICAL FIX: Properly preserve and merge collaboration metadata
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["needs_context"] = result.get("needs_context", False)
    
    # FIXED: Intelligent collaboration metadata merging
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(updated_state, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Recommendation node: MERGED collaboration metadata with {len(merged_collab.get('collaborating_agents', []))} total agents")
    
    return updated_state

@log_aspect
def collaborative_retrieval_node(state: JurixState) -> JurixState:
    """COLLABORATIVE ARTICLE RETRIEVAL - FIXED with metadata preservation"""
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
    logger.info(f"📖 Retrieval collaboration result keys: {list(result.keys())}")
    
    # CRITICAL FIX: Preserve collaboration metadata
    updated_state = state.copy()
    updated_state["articles"] = result.get("articles", [])
    
    # FIXED: Merge collaboration metadata properly
    new_collab = result.get("collaboration_metadata", {})
    if new_collab:
        merged_collab = merge_collaboration_metadata(updated_state, new_collab)
        updated_state["collaboration_metadata"] = merged_collab
        logger.info(f"🎼 Retrieval node: MERGED collaboration metadata")
    
    return updated_state

@log_aspect
def collaborative_chat_node(state: JurixState) -> JurixState:
    """COLLABORATIVE RESPONSE GENERATION - FIXED with final metadata preservation"""
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
    logger.info(f"💬 Chat collaboration result keys: {list(result.keys())}")
    
    # CRITICAL FIX: Create final state with complete collaboration metadata preservation
    updated_state = state.copy()
    
    # Update all the regular fields
    updated_state.update({
        "response": result.get("response", "No response generated"),
        "conversation_history": chat_agent.shared_memory.get_conversation(state["conversation_id"]),
        "articles_used": result.get("articles_used", []),
        "tickets": result.get("tickets", state["tickets"]),
        "workflow_status": result.get("workflow_status", "completed")
    })
    
    # CRITICAL FIX: Ensure final collaboration metadata is complete and preserved
    new_collab = result.get("collaboration_metadata", {})
    final_collab = merge_collaboration_metadata(updated_state, new_collab)
    
    if final_collab:
        # Add final workflow completion information
        final_collab["workflow_completed"] = True
        final_collab["final_agent"] = "chat_agent"
        final_collab["final_state_preserved"] = True
        final_collab["total_workflow_agents"] = len(final_collab.get("collaborating_agents", []))
        
        # Store in BOTH fields to ensure it's captured
        updated_state["collaboration_metadata"] = final_collab
        updated_state["final_collaboration_summary"] = final_collab  # Additional backup
        
        logger.info(f"🎉 FINAL CHAT NODE: Stored complete collaboration metadata with {final_collab.get('total_workflow_agents', 0)} agents")
        logger.info(f"🎉 FINAL COLLABORATION SUMMARY: {final_collab}")
    else:
        logger.error("❌ CRITICAL: No final collaboration metadata to store!")
    
    return updated_state

def build_workflow():
    """Build workflow with FIXED collaborative intelligence and metadata persistence"""
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
    """FIXED workflow runner with guaranteed collaboration metadata preservation"""
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
    
    logger.info(f"🚀 Starting FIXED collaborative workflow for: '{query}'")
    
    # Track collaboration metadata throughout execution
    collaboration_trace = []
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            # Track each node's collaboration metadata
            collab_info = node_state.get("collaboration_metadata", {})
            if collab_info:
                logger.info(f"🎼 {node_name} generated collaboration: {collab_info}")
                collaboration_trace.append({
                    "node": node_name,
                    "collaboration": collab_info
                })
            else:
                logger.warning(f"⚠️ {node_name} had no collaboration metadata")
            
            final_state = node_state
    
    # CRITICAL FIX: Ensure final state has collaboration metadata
    if final_state:
        final_collab = final_state.get("collaboration_metadata", {})
        backup_collab = final_state.get("final_collaboration_summary", {})
        
        if final_collab or backup_collab:
            logger.info(f"🎉 WORKFLOW COMPLETE: Final collaboration metadata preserved")
            logger.info(f"🎉 Total agents collaborated: {len((final_collab or backup_collab).get('collaborating_agents', []))}")
        else:
            logger.error(f"❌ CRITICAL: Final state missing collaboration metadata!")
            logger.error(f"❌ Available keys in final state: {list(final_state.keys())}")
            
            # Emergency reconstruction from trace
            if collaboration_trace:
                logger.info(f"🚨 EMERGENCY: Reconstructing collaboration metadata from trace")
                emergency_collab = {
                    "workflow_emergency_reconstruction": True,
                    "collaborating_agents": [],
                    "collaboration_types": [],
                    "nodes_with_collaboration": len(collaboration_trace)
                }
                
                for entry in collaboration_trace:
                    node_collab = entry["collaboration"]
                    agents = node_collab.get("collaborating_agents", [])
                    types = node_collab.get("collaboration_types", [])
                    emergency_collab["collaborating_agents"].extend(agents)
                    emergency_collab["collaboration_types"].extend(types)
                
                # Remove duplicates
                emergency_collab["collaborating_agents"] = list(set(emergency_collab["collaborating_agents"]))
                emergency_collab["collaboration_types"] = list(set(emergency_collab["collaboration_types"]))
                
                final_state["collaboration_metadata"] = emergency_collab
                logger.info(f"🚨 EMERGENCY RECONSTRUCTION: {emergency_collab}")
    
    return final_state or state

# Enhanced: Get collaboration insights
def get_collaboration_insights() -> Dict:
    """Get insights about collaborative performance"""
    return collaborative_framework.get_collaboration_insights()

# NEW: Test collaboration metadata persistence
def test_collaboration_metadata_persistence(query: str = "Give me recommendations for PROJ123") -> Dict:
    """Test function to verify collaboration metadata is preserved"""
    logger.info(f"🧪 TESTING collaboration metadata persistence with query: {query}")
    
    result = run_workflow(query)
    
    test_results = {
        "query": query,
        "has_collaboration_metadata": "collaboration_metadata" in result,
        "has_backup_metadata": "final_collaboration_summary" in result,
        "collaboration_data": result.get("collaboration_metadata", {}),
        "backup_data": result.get("final_collaboration_summary", {}),
        "all_result_keys": list(result.keys()),
        "collab_keys": [k for k in result.keys() if 'collab' in k.lower()]
    }
    
    logger.info(f"🧪 TEST RESULTS: {test_results}")
    return test_results