from typing import Dict, List, Any, TypedDict, Optional

class JurixState(TypedDict):
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
    project_id: str  # Added
    time_range: Dict[str, str]  # Added
    tickets: List[Dict[str, Any]]  # Added
    metrics: Dict[str, Any]  # Added
    visualization_data: Dict[str, Any]  # Added
    report: str  # Added
    metadata: Dict[str, Any]  # Added
    dashboard_id: Optional[str]  # Added
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
    error: Optional[str]  # Added for error handling
    recommendation_status: Optional[str]  # Added for recommendation agent status