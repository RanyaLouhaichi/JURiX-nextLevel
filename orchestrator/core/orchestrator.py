# orchestrator/core/orchestrator.py
# Enhanced orchestrator that integrates the collaborative framework
# This transforms your LangGraph nodes from simple agent callers into intelligent collaboration orchestrators

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
from orchestrator.core.collaborative_framework import CollaborativeFramework  # type: ignore # NEW: Import our intelligence coordinator
import functools
import logging
import uuid
import asyncio

# Enhanced logging for collaborative intelligence
logging.basicConfig(
    filename='collaborative_workflow.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_aspect(func):
    """Enhanced logging decorator that tracks collaborative intelligence"""
    @functools.wraps(func)
    def wrapper(state):
        logging.debug(f"🎯 Executing {func.__name__} with collaborative intelligence")
        logging.debug(f"State: {state}")
        result = func(state)
        
        # Log collaboration metadata if present
        collab_metadata = result.get("collaboration_metadata", {})
        if collab_metadata:
            logging.info(f"🤝 Collaboration in {func.__name__}: {collab_metadata}")
        
        logging.debug(f"✅ Completed {func.__name__} with result: {result}")
        return result
    return wrapper

# Initialize shared memory and agents
shared_memory = JurixSharedMemory()
chat_agent = ChatAgent(shared_memory)
retrieval_agent = RetrievalAgent(shared_memory)
recommendation_agent = RecommendationAgent(shared_memory)
jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)

# NEW: Create the agents registry for collaborative framework
agents_registry = {
    "chat_agent": chat_agent,
    "retrieval_agent": retrieval_agent,
    "recommendation_agent": recommendation_agent,
    "jira_data_agent": jira_data_agent
}

# NEW: Initialize the collaborative framework - this is the conductor for your agent orchestra
collaborative_framework = CollaborativeFramework(shared_memory.redis_client, agents_registry)

logging.info("🎭 Enhanced orchestrator initialized with collaborative intelligence")

@log_aspect
def classify_intent_node(state: JurixState) -> JurixState:
    """
    Intent classification remains the same - this node focuses on understanding user intent
    The collaborative intelligence happens in the agent coordination nodes below
    """
    intent_result = classify_intent(state["query"], state["conversation_history"])
    updated_state = state.copy()
    updated_state["intent"] = intent_result
    updated_state["project"] = intent_result.get("project")
    logging.debug(f"Intent classification result: {intent_result}")
    return updated_state

@log_aspect
def jira_data_agent_node(state: JurixState) -> JurixState:
    """
    Enhanced Jira data node that uses collaborative intelligence
    
    Instead of simply calling jira_data_agent.run(), this node now:
    1. Calls the collaborative framework
    2. Allows the framework to coordinate multiple agents if beneficial
    3. Returns enhanced results from potential multi-agent collaboration
    """
    project = state.get("project")
    if not project:
        logging.warning("No project specified for collaborative data retrieval")
        return state.copy()
    
    def run_collaborative_data_retrieval():
        """
        This is where the magic happens - instead of a simple agent call,
        we use the collaborative framework to potentially coordinate multiple agents
        """
        task_context = {
            "project_id": project,
            "time_range": {
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-17T23:59:59Z"
            },
            "analysis_depth": "enhanced",  # Signal that we want comprehensive analysis
            "workflow_context": "langgraph_orchestration"  # Help agents understand the context
        }
        
        # Use collaborative framework instead of direct agent call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                collaborative_framework.coordinate_agents("jira_data_agent", task_context)
            )
            return result
        finally:
            loop.close()
    
    logging.info(f"🚀 Starting collaborative data retrieval for project: {project}")
    result = run_collaborative_data_retrieval()
    
    updated_state = state.copy()
    updated_state["tickets"] = result.get("tickets", [])
    updated_state["additional_metrics"] = result.get("additional_metrics", {})
    updated_state["collaboration_info"] = result.get("collaboration_metadata", {})
    
    # Log collaboration outcomes
    collab_info = updated_state.get("collaboration_info", {})
    if collab_info.get("collaborating_agents"):
        logging.info(f"🎼 Data retrieval involved collaboration with: {collab_info['collaborating_agents']}")
    
    return updated_state

@log_aspect  
def recommendation_agent_node(state: JurixState) -> JurixState:
    """
    Enhanced recommendation node using collaborative intelligence
    
    This is where you'll really see the power of the hybrid architecture.
    The recommendation agent can now collaborate with data agents, productivity agents,
    and validation agents to provide much richer, more comprehensive recommendations.
    """
    
    def run_collaborative_recommendation():
        """
        Transform recommendation generation from a solo performance into an intelligent collaboration
        """
        # Prepare rich context for collaboration
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "articles": state["articles"],
            "project": state["project"],
            "tickets": state["tickets"],
            "workflow_type": "collaborative_orchestration",
            "intent": state["intent"],
            "collaboration_intent": "comprehensive_recommendations"  # Signal our collaborative intent
        }
        
        # Use collaborative framework for intelligent agent coordination
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                collaborative_framework.coordinate_agents("recommendation_agent", task_context)
            )
            return result
        finally:
            loop.close()
    
    logging.info(f"🎯 Starting collaborative recommendation generation for: {state['query']}")
    result = run_collaborative_recommendation()
    
    updated_state = state.copy()
    updated_state["recommendations"] = result.get("recommendations", [])
    updated_state["needs_context"] = result.get("needs_context", False)
    
    # Add collaboration metadata to state for visibility
    updated_state["collaboration_info"] = result.get("collaboration_metadata", {})
    
    # Log the collaborative intelligence outcomes
    collab_info = updated_state.get("collaboration_info", {})
    if collab_info:
        collaborating_agents = collab_info.get("collaborating_agents", [])
        collaboration_quality = collab_info.get("collaboration_quality", 0)
        logging.info(f"🎼 Recommendation generation collaboration summary:")
        logging.info(f"   Collaborating agents: {collaborating_agents}")
        logging.info(f"   Collaboration quality: {collaboration_quality:.2f}")
        logging.info(f"   Recommendations generated: {len(updated_state['recommendations'])}")
    
    return updated_state

@log_aspect
def retrieval_agent_node(state: JurixState) -> JurixState:
    """
    Enhanced retrieval node with collaborative intelligence
    
    Article retrieval can now benefit from collaboration - for example,
    the retrieval agent might collaborate with recommendation agents to
    find not just relevant articles, but articles that support strategic insights.
    """
    def run_collaborative_retrieval():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "intent": state["intent"],
            "collaboration_context": "article_retrieval_with_strategic_enhancement"
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                collaborative_framework.coordinate_agents("retrieval_agent", task_context)
            )
            return result
        finally:
            loop.close()
    
    logging.info(f"📚 Starting collaborative article retrieval")
    result = run_collaborative_retrieval()
    
    updated_state = state.copy()
    updated_state["articles"] = result.get("articles", [])
    updated_state["collaboration_info"] = result.get("collaboration_metadata", {})
    
    # Notify other agents of retrieved articles via shared memory
    shared_memory.store("articles", updated_state["articles"])
    
    return updated_state

@log_aspect
def chat_agent_node(state: JurixState) -> JurixState:
    """
    Enhanced chat agent node with collaborative intelligence
    
    The chat agent is often the final synthesizer in the workflow. With collaborative intelligence,
    it can coordinate with other agents to ensure it has the best possible context for generating
    its response, and can even collaborate during response generation for complex queries.
    """
    def run_collaborative_chat():
        task_context = {
            "session_id": state["conversation_id"],
            "user_prompt": state["query"],
            "articles": state["articles"],
            "recommendations": state["recommendations"],
            "tickets": state["tickets"],
            "intent": state["intent"],
            "collaboration_context": "comprehensive_response_generation"
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                collaborative_framework.coordinate_agents("chat_agent", task_context)
            )
            return result
        finally:
            loop.close()
    
    logging.info(f"💬 Starting collaborative response generation")
    result = run_collaborative_chat()
    
    updated_state = state.copy()
    updated_state.update({
        "response": result.get("response", "No response generated"),
        "conversation_history": chat_agent.shared_memory.get_conversation(state["conversation_id"]),
        "articles_used": result.get("articles_used", []),
        "tickets": result.get("tickets", state["tickets"]),
        "workflow_status": result.get("workflow_status", "completed")
    })
    
    # Handle collaborative feedback loop if more context is needed
    if state.get("needs_context", False):
        updated_state["needs_context"] = False
        logging.info("🔄 Re-entering workflow due to collaborative context enhancement")
    
    # Add final collaboration summary
    updated_state["final_collaboration_summary"] = result.get("collaboration_metadata", {})
    
    logging.info(f"✅ Collaborative workflow completed with status: {updated_state['workflow_status']}")
    return updated_state

def build_workflow():
    """
    Build the enhanced workflow with collaborative intelligence
    
    The workflow structure remains exactly the same as before - this is the beauty
    of the hybrid architecture. Your LangGraph orchestration is preserved, but now
    each node has collaborative intelligence that can coordinate multiple agents.
    """
    workflow = StateGraph(JurixState)
    
    # Add all nodes - same structure as before, enhanced intelligence within nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("jira_data_agent", jira_data_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)
    workflow.add_node("retrieval_agent", retrieval_agent_node)
    workflow.add_node("chat_agent", chat_agent_node)

    workflow.set_entry_point("classify_intent")

    def route(state: JurixState) -> str:
        """
        Enhanced routing that can consider collaborative intelligence outcomes
        """
        intent = state["intent"]["intent"] if "intent" in state and "intent" in state["intent"] else "generic_question"
        needs_context = state.get("needs_context", False)
        
        # Log routing decisions with collaborative context
        collab_info = state.get("collaboration_info", {})
        if collab_info:
            logging.info(f"🎯 Routing with intent: {intent}, collaboration info available: {bool(collab_info)}")
        
        if needs_context:
            return "chat_agent"  # Loop back for collaborative context enhancement
        
        routing = {
            "generic_question": "chat_agent",
            "follow_up": "chat_agent", 
            "article_retrieval": "retrieval_agent",
            "recommendation": "jira_data_agent"
        }
        return routing.get(intent, "chat_agent")

    # Same workflow structure as before - the collaborative intelligence is within the nodes
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
    """
    Enhanced workflow runner that supports collaborative intelligence
    
    This function signature remains exactly the same, maintaining full backward compatibility.
    The enhanced intelligence happens transparently within the workflow execution.
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    
    # Create initial state - same as before
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
    
    # Build and run workflow with collaborative intelligence
    workflow = build_workflow()
    final_state = None
    
    logging.info(f"🚀 Starting collaborative workflow for query: '{query}'")
    
    for event in workflow.stream(state):
        for node_name, node_state in event.items():
            # Enhanced logging for collaborative outcomes
            collab_info = node_state.get("collaboration_info", {})
            if collab_info:
                logging.info(f"🎼 Node {node_name} collaboration: {collab_info}")
            
            final_state = node_state
    
    # Log final collaboration summary
    if final_state:
        final_collab_summary = final_state.get("final_collaboration_summary", {})
        if final_collab_summary:
            logging.info(f"🎉 Workflow collaboration summary: {final_collab_summary}")
    
    logging.info(f"✅ Collaborative workflow completed for: '{query}'")
    return final_state or state

# NEW: Function to get collaborative intelligence insights
def get_collaboration_insights() -> Dict:
    """Get insights about how collaborative intelligence is performing"""
    return collaborative_framework.get_collaboration_insights()

# NEW: Function to get collaboration performance metrics
def get_collaboration_performance() -> Dict:
    """Get performance metrics for collaborative intelligence"""
    return {
        "framework_metrics": collaborative_framework.performance_metrics,
        "agent_collaboration_frequency": get_collaboration_insights().get("agent_collaboration_frequency", {}),
        "collaboration_effectiveness": get_collaboration_insights().get("collaboration_effectiveness", {}),
        "recent_collaborations": get_collaboration_insights().get("total_recent_collaborations", 0)
    }