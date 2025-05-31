# orchestrator/graph/state.py
# FIXED VERSION - Now properly supports collaboration metadata

from typing import Dict, List, Any, TypedDict, Optional

class JurixState(TypedDict):
    # Core workflow fields
    query: str
    intent: Dict[str, Any]
    conversation_id: str
    conversation_history: List[Dict[str, str]]
    articles: List[Dict[str, Any]]
    recommendations: List[str]
    status: str
    response: str
    articles_used: List[Dict[str, Any]]
    workflow_status: str
    next_agent: str
    project: Optional[str]
    
    # Productivity workflow fields
    project_id: str
    time_range: Dict[str, str]
    tickets: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any]
    report: str
    metadata: Dict[str, Any]
    dashboard_id: Optional[str]
    
    # Article generation workflow fields
    ticket_id: str
    article: Dict[str, Any]
    redundant: bool
    refinement_suggestion: Optional[str]
    approved: bool
    refinement_count: int
    has_refined: bool
    iteration_count: int
    workflow_stage: str
    recommendation_id: Optional[str]
    workflow_history: List[Dict[str, Any]]
    error: Optional[str]
    recommendation_status: Optional[str]
    
    # CRITICAL FIX: Add collaboration metadata fields to the state definition
    collaboration_metadata: Optional[Dict[str, Any]]  # Main collaboration metadata
    final_collaboration_summary: Optional[Dict[str, Any]]  # Backup/final summary
    collaboration_insights: Optional[Dict[str, Any]]  # Additional insights
    
    # Optional: Additional collaboration tracking fields
    collaboration_trace: Optional[List[Dict[str, Any]]]  # Track collaboration through workflow
    collaborative_agents_used: Optional[List[str]]  # Simple list of agents that collaborated