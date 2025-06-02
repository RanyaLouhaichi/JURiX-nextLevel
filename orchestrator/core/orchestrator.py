import sys
import os
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore
from orchestrator.core.intent_router import classify_intent # type: ignore
import functools
import logging
import uuid
import asyncio
from datetime import datetime

from agents.chat_agent import ChatAgent
from agents.retrieval_agent import RetrievalAgent
from agents.recommendation_agent import RecommendationAgent
from agents.jira_data_agent import JiraDataAgent
from agents.productivity_dashboard_agent import ProductivityDashboardAgent
from agents.jira_article_generator_agent import JiraArticleGeneratorAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.predictive_analysis_agent import PredictiveAnalysisAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")

class Orchestrator:
    """Single orchestrator that handles all workflows"""
    
    def __init__(self):
        self.shared_memory = JurixSharedMemory()
        
        self.chat_agent = ChatAgent(self.shared_memory)
        self.retrieval_agent = RetrievalAgent(self.shared_memory)
        self.recommendation_agent = RecommendationAgent(self.shared_memory)
        self.jira_data_agent = JiraDataAgent(redis_client=self.shared_memory.redis_client)
        self.productivity_dashboard_agent = ProductivityDashboardAgent(redis_client=self.shared_memory.redis_client)
        self.jira_article_generator = JiraArticleGeneratorAgent(self.shared_memory)
        self.knowledge_base = KnowledgeBaseAgent(self.shared_memory)
        self.predictive_analysis_agent = PredictiveAnalysisAgent(redis_client=self.shared_memory.redis_client)
        
        self.agents_registry = {
            "chat_agent": self.chat_agent,
            "retrieval_agent": self.retrieval_agent,
            "recommendation_agent": self.recommendation_agent,
            "jira_data_agent": self.jira_data_agent,
            "productivity_dashboard_agent": self.productivity_dashboard_agent,
            "jira_article_generator_agent": self.jira_article_generator,
            "knowledge_base_agent": self.knowledge_base,
            "predictive_analysis_agent": self.predictive_analysis_agent
        }
        
        self.collaborative_framework = CollaborativeFramework(
            self.shared_memory.redis_client, 
            self.agents_registry
        )
        
        logger.info("Orchestrator initialized with all agents including predictive analysis")
    
    def run_workflow(self, query: str, conversation_id: str = None) -> JurixState:
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
            collaborative_agents_used=None,
            predictions=None,
            predictive_insights=None
        )
        
        workflow = self._build_general_workflow()
        final_state = None
        
        logger.info(f"Starting general workflow for: '{query}'")
        
        collaboration_trace = []
        articles_tracking = []
        
        for event in workflow.stream(state):
            for node_name, node_state in event.items():
                collab_info = node_state.get("collaboration_metadata", {})
                articles_count = len(node_state.get("articles", []))
                
                if collab_info:
                    collaboration_trace.append({
                        "node": node_name,
                        "collaboration": collab_info,
                        "articles_count": articles_count
                    })
                
                articles_tracking.append({
                    "node": node_name,
                    "articles_count": articles_count
                })
                
                final_state = node_state
        
        if final_state:
            final_state["collaboration_trace"] = collaboration_trace
            final_state["articles_tracking"] = articles_tracking
        
        return final_state or state
    
    def _build_general_workflow(self):
        workflow = StateGraph(JurixState)
        
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("jira_data_agent", self._collaborative_data_node)
        workflow.add_node("recommendation_agent", self._collaborative_recommendation_node)
        workflow.add_node("retrieval_agent", self._collaborative_retrieval_node)
        workflow.add_node("predictive_analysis_agent", self._collaborative_predictive_node)
        workflow.add_node("chat_agent", self._collaborative_chat_node)

        workflow.set_entry_point("classify_intent")

        def route(state: JurixState) -> str:
            intent = state["intent"]["intent"] if "intent" in state and "intent" in state["intent"] else "generic_question"
            needs_context = state.get("needs_context", False)
            
            logger.info(f"Routing with intent: {intent}")
            
            query_lower = state.get("query", "").lower()
            predictive_keywords = ["will we complete", "sprint completion", "predict", "forecast", "risk", "when will", "probability", "chance of"]
            
            if any(keyword in query_lower for keyword in predictive_keywords):
                logger.info("Routing to predictive analysis based on keywords")
                return "predictive_analysis_agent"
            
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
        workflow.add_edge("predictive_analysis_agent", "chat_agent")
        workflow.add_edge("chat_agent", END)

        return workflow.compile()
    
    def _classify_intent_node(self, state: JurixState) -> JurixState:
        intent_result = classify_intent(state["query"], state["conversation_history"])
        updated_state = state.copy()
        updated_state["intent"] = intent_result
        updated_state["project"] = intent_result.get("project")
        return updated_state
    
    def _collaborative_data_node(self, state: JurixState) -> JurixState:
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
                return loop.run_until_complete(
                    self.collaborative_framework.coordinate_agents("jira_data_agent", task_context)
                )
            finally:
                loop.close()
        
        result = run_collaboration()
        
        updated_state = state.copy()
        updated_state["tickets"] = result.get("tickets", [])
        
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        return updated_state
    
    def _collaborative_recommendation_node(self, state: JurixState) -> JurixState:
        def run_collaboration():
            task_context = {
                "session_id": state["conversation_id"],
                "user_prompt": state["query"],
                "articles": state.get("articles", []),
                "project": state["project"],
                "tickets": state["tickets"],
                "workflow_type": "collaborative_orchestration",
                "intent": state["intent"],
                "predictions": state.get("predictions", {})
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.collaborative_framework.coordinate_agents("recommendation_agent", task_context)
                )
            finally:
                loop.close()
        
        result = run_collaboration()
        
        updated_state = state.copy()
        updated_state["recommendations"] = result.get("recommendations", [])
        updated_state["needs_context"] = result.get("needs_context", False)
        
        if result.get("articles"):
            updated_state["articles"] = result["articles"]
        
        if result.get("articles_from_collaboration"):
            updated_state["articles_from_collaboration"] = result["articles_from_collaboration"]
        
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        return updated_state
    
    def _collaborative_retrieval_node(self, state: JurixState) -> JurixState:
        def run_collaboration():
            task_context = {
                "session_id": state["conversation_id"],
                "user_prompt": state["query"],
                "intent": state["intent"]
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.collaborative_framework.coordinate_agents("retrieval_agent", task_context)
                )
            finally:
                loop.close()
        
        result = run_collaboration()
        
        updated_state = state.copy()
        updated_state["articles"] = result.get("articles", [])
        
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        return updated_state
    
    def _collaborative_predictive_node(self, state: JurixState) -> JurixState:
        logger.info("Running predictive analysis node")
        
        def run_collaboration():
            historical_data = {}
            if state.get("tickets"):
                velocity_history = self._extract_velocity_history(state["tickets"])
                historical_data["velocity_history"] = velocity_history
            
            task_context = {
                "tickets": state.get("tickets", []),
                "metrics": state.get("metrics", {}),
                "historical_data": historical_data,
                "user_query": state["query"],
                "analysis_type": "comprehensive",
                "project": state.get("project", "PROJ123")
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.collaborative_framework.coordinate_agents("predictive_analysis_agent", task_context)
                )
            finally:
                loop.close()
        
        if not state.get("tickets"):
            jira_result = self.jira_data_agent.run({
                "project_id": state.get("project", "PROJ123"),
                "time_range": {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-17T23:59:59Z"
                }
            })
            state["tickets"] = jira_result.get("tickets", [])
        
        result = run_collaboration()
        
        updated_state = state.copy()
        updated_state["predictions"] = result.get("predictions", {})
        updated_state["predictive_insights"] = result.get("predictions", {}).get("natural_language_summary", "")
        
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        return updated_state
    
    def _extract_velocity_history(self, tickets: List[Dict[str, Any]]) -> List[float]:
        weekly_counts = {}
        
        for ticket in tickets:
            if ticket.get("fields", {}).get("status", {}).get("name") == "Done":
                resolution_date = ticket.get("fields", {}).get("resolutiondate")
                if resolution_date:
                    try:
                        date = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
                        week_key = date.strftime("%Y-W%U")
                        weekly_counts[week_key] = weekly_counts.get(week_key, 0) + 1
                    except:
                        pass
        
        if weekly_counts:
            sorted_weeks = sorted(weekly_counts.keys())
            return [weekly_counts[week] for week in sorted_weeks[-5:]]
        
        return [5, 6, 4, 7, 5]
    
    def _collaborative_chat_node(self, state: JurixState) -> JurixState:
        def run_collaboration():
            task_context = {
                "session_id": state["conversation_id"],
                "user_prompt": state["query"],
                "articles": state.get("articles", []),
                "recommendations": state["recommendations"],
                "tickets": state["tickets"],
                "intent": state["intent"],
                "predictions": state.get("predictions", {}),
                "predictive_insights": state.get("predictive_insights", "")
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.collaborative_framework.coordinate_agents("chat_agent", task_context)
                )
            finally:
                loop.close()
        
        result = run_collaboration()
        
        updated_state = state.copy()
        
        updated_state.update({
            "response": result.get("response", "No response generated"),
            "conversation_history": self.chat_agent.shared_memory.get_conversation(state["conversation_id"]),
            "articles_used": result.get("articles_used", []),
            "tickets": result.get("tickets", state["tickets"]),
            "workflow_status": result.get("workflow_status", "completed")
        })
        
        if state.get("articles"):
            updated_state["articles"] = state["articles"]
        
        if result.get("articles"):
            updated_state["articles"] = result["articles"]
        
        new_collab = result.get("collaboration_metadata", {})
        final_collab = self._merge_collaboration_metadata(updated_state, new_collab)
        
        if final_collab:
            final_collab["workflow_completed"] = True
            final_collab["final_agent"] = "chat_agent"
            final_collab["final_state_preserved"] = True
            final_collab["total_workflow_agents"] = len(final_collab.get("collaborating_agents", []))
            final_collab["final_articles_count"] = len(updated_state.get("articles", []))
            
            updated_state["collaboration_metadata"] = final_collab
            updated_state["final_collaboration_summary"] = final_collab
        
        return updated_state
    
    def _merge_collaboration_metadata(self, existing_state: JurixState, new_collab: Dict) -> Dict:
        existing_collab = existing_state.get("collaboration_metadata", {})
        
        if not new_collab:
            return existing_collab
        
        if not existing_collab:
            return new_collab.copy()
        
        merged = existing_collab.copy()
        
        existing_agents = set(merged.get("collaborating_agents", []))
        new_agents = set(new_collab.get("collaborating_agents", []))
        all_agents = list(existing_agents | new_agents)
        
        existing_types = set(merged.get("collaboration_types", []))
        new_types = set(new_collab.get("collaboration_types", []))
        all_types = list(existing_types | new_types)
        
        merged["articles_retrieved"] = merged.get("articles_retrieved", 0) + new_collab.get("articles_retrieved", 0)
        merged["articles_merged"] = merged.get("articles_merged", False) or new_collab.get("articles_merged", False)
        
        merged.update(new_collab)
        merged["collaborating_agents"] = all_agents
        merged["collaboration_types"] = all_types
        merged["total_collaborations"] = len(all_agents)
        merged["workflow_collaboration_complete"] = True
        
        return merged
    
    def run_productivity_workflow(self, project_id: str, time_range: Dict[str, str], conversation_id: str = None) -> JurixState:
        conversation_id = conversation_id or str(uuid.uuid4())
        
        state = JurixState(
            query=f"Generate productivity analysis for {project_id}",
            intent={"intent": "productivity_analysis", "project": project_id},
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
            dashboard_id=None,
            collaboration_metadata=None,
            final_collaboration_summary=None,
            collaboration_insights=None,
            collaboration_trace=None,
            collaborative_agents_used=None,
            predictions=None,
            predictive_insights=None
        )
        
        logger.info(f"Starting productivity workflow for {project_id}")
        
        workflow = self._build_productivity_workflow()
        final_state = state
        
        collaboration_trace = []
        
        for event in workflow.stream(state):
            for node_name, node_state in event.items():
                collab_metadata = node_state.get("collaboration_metadata", {})
                if collab_metadata:
                    collaboration_trace.append({
                        "node": node_name,
                        "collaboration": collab_metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                
                final_state = node_state
        
        final_state["collaboration_trace"] = collaboration_trace
        
        return final_state
    
    def _build_productivity_workflow(self):
        workflow = StateGraph(JurixState)
        
        workflow.add_node("jira_data_agent", self._productivity_jira_data_node)
        workflow.add_node("recommendation_agent", self._productivity_recommendation_node)
        workflow.add_node("predictive_analysis_agent", self._productivity_predictive_node)
        workflow.add_node("productivity_dashboard_agent", self._productivity_dashboard_node)
        
        workflow.set_entry_point("jira_data_agent")
        
        workflow.add_edge("jira_data_agent", "predictive_analysis_agent")
        workflow.add_edge("predictive_analysis_agent", "recommendation_agent")
        workflow.add_edge("recommendation_agent", "productivity_dashboard_agent")
        workflow.add_edge("productivity_dashboard_agent", END)
        
        def handle_error(state: JurixState) -> str:
            if state["workflow_status"] == "failure":
                return END
            return "predictive_analysis_agent"
        
        workflow.add_conditional_edges("jira_data_agent", handle_error)
        
        return workflow.compile()
    
    def _productivity_jira_data_node(self, state: JurixState) -> JurixState:
        input_data = {
            "project_id": state["project_id"],
            "time_range": state["time_range"],
            "analysis_depth": "enhanced",
            "workflow_context": "productivity_analysis"
        }
        
        result = self.jira_data_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["tickets"] = result.get("tickets", [])
        updated_state["workflow_status"] = result.get("workflow_status", "failure")
        updated_state["metadata"] = result.get("metadata", {})
        
        if result.get("collaboration_metadata"):
            updated_state["collaboration_metadata"] = result["collaboration_metadata"]
        
        if updated_state["workflow_status"] == "failure":
            updated_state["error"] = "Failed to retrieve Jira ticket data"
        
        return updated_state
    
    def _productivity_predictive_node(self, state: JurixState) -> JurixState:
        tickets = state["tickets"]
        
        done_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "Done"])
        in_progress = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "In Progress"])
        to_do = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "To Do"])
        
        velocity = done_tickets
        cycle_times = []
        assignee_counts = {}
        
        for ticket in tickets:
            changelog = ticket.get("changelog", {}).get("histories", [])
            start_date = None
            end_date = None
            
            for history in changelog:
                for item in history.get("items", []):
                    if item.get("field") == "status":
                        if item.get("toString") == "In Progress" and not start_date:
                            start_date = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                        elif item.get("toString") == "Done":
                            end_date = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
            
            if start_date and end_date:
                cycle_time = (end_date - start_date).days
                if cycle_time >= 0:
                    cycle_times.append(cycle_time)
            
            assignee_info = ticket.get("fields", {}).get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 5
        
        metrics = {
            "throughput": velocity,
            "cycle_time": avg_cycle_time,
            "workload": assignee_counts,
            "bottlenecks": {"To Do": to_do, "In Progress": in_progress}
        }
        
        state["metrics"] = metrics
        
        velocity_history = self._extract_velocity_history(tickets)
        historical_data = {"velocity_history": velocity_history}
        
        input_data = {
            "tickets": tickets,
            "metrics": metrics,
            "historical_data": historical_data,
            "user_query": "Generate comprehensive predictive analysis for productivity dashboard",
            "analysis_type": "comprehensive",
            "collaboration_context": "productivity_enhancement"
        }
        
        result = self.predictive_analysis_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["predictions"] = result.get("predictions", {})
        updated_state["predictive_insights"] = result.get("predictions", {}).get("natural_language_summary", "")
        
        existing_collab = updated_state.get("collaboration_metadata", {})
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        return updated_state
    
    def _productivity_recommendation_node(self, state: JurixState) -> JurixState:
        project_id = state["project_id"]
        ticket_count = len(state["tickets"])
        predictions = state.get("predictions", {})
        
        prompt = f"Analyze productivity data for {project_id} with {ticket_count} tickets and provide recommendations for improving team efficiency"
        
        if predictions.get("warnings"):
            prompt += f". Pay special attention to these warnings: {[w['message'] for w in predictions['warnings'][:2]]}"
        
        input_data = {
            "session_id": state["conversation_id"],
            "user_prompt": prompt,
            "project": project_id,
            "tickets": state["tickets"],
            "workflow_type": "productivity_analysis",
            "intent": {"intent": "recommendation", "project": project_id},
            "collaboration_context": "productivity_optimization",
            "predictions": predictions
        }
        
        result = self.recommendation_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["recommendations"] = result.get("recommendations", [])
        updated_state["recommendation_status"] = result.get("workflow_status", "failure")
        
        existing_collab = updated_state.get("collaboration_metadata", {})
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        if updated_state["recommendation_status"] == "failure" and updated_state["workflow_status"] == "success":
            updated_state["recommendations"] = [
                "Consider balanced workload distribution among team members",
                "Review tickets in bottleneck stages to identify common blockers",
                "Schedule regular process review meetings to address efficiency issues"
            ]
        
        return updated_state
    
    def _productivity_dashboard_node(self, state: JurixState) -> JurixState:
        input_data = {
            "tickets": state["tickets"],
            "recommendations": state["recommendations"],
            "project_id": state["project_id"],
            "collaboration_context": "comprehensive_analysis",
            "predictions": state.get("predictions", {}),
            "primary_agent_result": {
                "tickets": state["tickets"],
                "recommendations": state["recommendations"],
                "metadata": state.get("metadata", {}),
                "predictions": state.get("predictions", {})
            }
        }
        
        result = self.productivity_dashboard_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["metrics"] = result.get("metrics", {})
        updated_state["visualization_data"] = result.get("visualization_data", {})
        updated_state["report"] = result.get("report", "")
        updated_state["workflow_status"] = result.get("workflow_status", "failure")
        
        existing_collab = updated_state.get("collaboration_metadata", {})
        new_collab = result.get("collaboration_metadata", {})
        if existing_collab or new_collab:
            final_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            final_collab["workflow_completed"] = True
            final_collab["final_agent"] = "productivity_dashboard_agent"
            final_collab["workflow_type"] = "productivity_analysis"
            updated_state["collaboration_metadata"] = final_collab
            updated_state["final_collaboration_summary"] = final_collab
        
        dashboard_id = f"dashboard_{state['project_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.shared_memory.store(dashboard_id, {
            "project_id": state["project_id"],
            "time_range": state["time_range"],
            "metrics": updated_state["metrics"],
            "visualization_data": updated_state["visualization_data"],
            "report": updated_state["report"],
            "recommendations": updated_state["recommendations"],
            "predictions": updated_state.get("predictions", {}),
            "timestamp": datetime.now().isoformat(),
            "collaboration_metadata": updated_state.get("collaboration_metadata", {})
        })
        
        updated_state["dashboard_id"] = dashboard_id
        
        return updated_state
    
    def run_jira_workflow(self, ticket_id: str, conversation_id: str = None, project_id: str = "PROJ123") -> JurixState:
        conversation_id = conversation_id or str(uuid.uuid4())
        
        state = JurixState(
            query=f"Generate comprehensive article for ticket {ticket_id}",
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
            collaboration_metadata=None,
            final_collaboration_summary=None,
            collaboration_insights=None,
            collaboration_trace=None,
            collaborative_agents_used=None,
            predictions=None,
            predictive_insights=None
        )
        
        logger.info(f"Starting jira workflow for ticket {ticket_id}")
        
        recommendation_id, recommendations = self._run_jira_recommendation_agent(state)
        state["recommendation_id"] = recommendation_id
        state["recommendations"] = recommendations
        
        workflow = self._build_jira_workflow()
        final_state = state
        collaboration_trace = []
        
        for event in workflow.stream(state):
            for node_name, node_state in event.items():
                collab_metadata = node_state.get("collaboration_metadata", {})
                if collab_metadata:
                    collaboration_trace.append({
                        "node": node_name,
                        "collaboration": collab_metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                
                final_state = node_state
                
                if (node_name == "jira_article_generator" and 
                    final_state.get("article") and 
                    "provide more project-specific details" in str(final_state.get("recommendations", []))):
                    final_state_dict = dict(final_state)
                    final_state_dict["article"] = final_state.get("article", {})
                    recommendation_id, recommendations = self._run_jira_recommendation_agent(final_state_dict)
                    final_state["recommendation_id"] = recommendation_id
                    final_state["recommendations"] = recommendations
        
        final_state["collaboration_trace"] = collaboration_trace
        
        return final_state
    
    def _build_jira_workflow(self):
        workflow = StateGraph(JurixState)
        
        workflow.add_node("jira_article_generator", self._jira_article_generator_node)
        workflow.add_node("knowledge_base", self._knowledge_base_node)

        workflow.set_entry_point("jira_article_generator")

        def route(state: JurixState) -> str:
            logger.info(f"[ROUTING] Current workflow state analysis:")
            logger.info(f"  - Workflow status: {state.get('workflow_status')}")
            logger.info(f"  - Workflow stage: {state.get('workflow_stage')}")
            logger.info(f"  - Has article: {bool(state.get('article'))}")
            logger.info(f"  - Has refined: {state.get('has_refined', False)}")
            logger.info(f"  - Refinement suggestion: {bool(state.get('refinement_suggestion'))}")
            
            if (state.get("workflow_status") == "failure" or 
                not state.get("article") or 
                state["article"].get("status") == "error"):
                
                state["workflow_stage"] = "terminated_failure"
                state["workflow_history"].append({
                    "step": "workflow_terminated",
                    "article": state["article"],
                    "redundant": state["redundant"],
                    "refinement_suggestion": state["refinement_suggestion"],
                    "workflow_status": state["workflow_status"],
                    "workflow_stage": state["workflow_stage"],
                    "recommendation_id": state["recommendation_id"],
                    "recommendations": state["recommendations"],
                    "autonomous_refinement_done": state.get("autonomous_refinement_done", False),
                    "collaboration_metadata": state.get("collaboration_metadata", {}),
                    "termination_reason": "critical_failure"
                })
                return END
            
            if state.get("workflow_stage") in ["article_generated", "article_refined"]:
                return "knowledge_base"
            
            elif state.get("workflow_stage") == "knowledge_base_evaluated":
                if state.get("refinement_suggestion") and not state.get("has_refined", False):
                    return "jira_article_generator"
                else:
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
                        "collaboration_metadata": state.get("collaboration_metadata", {}),
                        "awaiting_human_approval": True
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
                    "collaboration_metadata": state.get("collaboration_metadata", {}),
                    "human_approved": True
                })
            
            return END

        workflow.add_conditional_edges("jira_article_generator", route)
        workflow.add_conditional_edges("knowledge_base", route)

        return workflow.compile()
    
    def _run_jira_recommendation_agent(self, state: Dict[str, Any]) -> tuple:
        logger.info(f"Running recommendation agent for ticket {state['ticket_id']}")
        
        project_id = state.get("project", "PROJ123")
        
        jira_input = {
            "project_id": project_id,
            "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"}
        }
        
        try:
            ticket_data_result = self.jira_data_agent.run(jira_input)
            tickets = ticket_data_result.get("tickets", [])
            
            input_data = {
                "session_id": state.get("conversation_id", f"rec_{state['ticket_id']}"),
                "user_prompt": f"Analyze resolved ticket {state['ticket_id']} and provide strategic recommendations for process improvement and knowledge sharing",
                "articles": [state.get("article", {})] if state.get("article") else [],
                "project": project_id,
                "tickets": tickets,
                "workflow_type": "knowledge_base_creation",
                "intent": {"intent": "strategic_recommendations"}
            }
            
            result = self.recommendation_agent.run(input_data)
            recommendations = result.get("recommendations", [])
            
            if not recommendations or (len(recommendations) == 1 and "provide more project-specific details" in recommendations[0].lower()):
                target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
                if target_ticket:
                    ticket_fields = target_ticket.get("fields", {})
                    enhanced_prompt = (
                        f"Based on the successful resolution of ticket {state['ticket_id']} - {ticket_fields.get('summary', 'No summary')}, "
                        f"provide strategic recommendations for preventing similar issues, improving team processes, and sharing knowledge. "
                        f"Status: {ticket_fields.get('status', {}).get('name', 'Unknown')}. "
                        f"Focus on actionable insights that can improve project efficiency and team learning."
                    )
                    
                    enhanced_input = input_data.copy()
                    enhanced_input["user_prompt"] = enhanced_prompt
                    enhanced_result = self.recommendation_agent.run(enhanced_input)
                    recommendations = enhanced_result.get("recommendations", [])
            
            if not recommendations:
                target_ticket = next((t for t in tickets if t.get("key") == state['ticket_id']), None)
                if target_ticket:
                    ticket_summary = target_ticket.get("fields", {}).get("summary", "this ticket")
                    recommendations = [
                        f"Document the resolution pattern from {state['ticket_id']} ({ticket_summary}) for future team reference and knowledge sharing.",
                        f"Implement proactive monitoring to detect similar issues early, based on the resolution approach used in {state['ticket_id']}.",
                        f"Create automated testing scenarios that cover the use case resolved in {state['ticket_id']} to prevent regression.",
                        f"Schedule a team knowledge-sharing session about the {state['ticket_id']} resolution methodology and lessons learned."
                    ]
                else:
                    recommendations = [
                        "Document the resolution methodology for future team reference and knowledge base enhancement.",
                        "Implement monitoring and alerting to detect similar issues proactively before they impact users.",
                        "Develop automated tests to prevent regression of this issue type in future releases.",
                        "Share resolution knowledge with the team through documentation and training sessions."
                    ]
        
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            recommendations = [
                f"Review and document the resolution approach used in {state.get('ticket_id', 'this ticket')} for team knowledge sharing.",
                "Implement preventive measures based on the root cause analysis of this issue.",
                "Create automated monitoring to detect similar issues early in the future.",
                "Schedule team discussion about applying this resolution pattern to similar scenarios."
            ]
        
        recommendation_id = self._store_recommendations(state["ticket_id"], recommendations)
        return recommendation_id, recommendations
    
    def _store_recommendations(self, ticket_id: str, recommendations: List[str]) -> str:
        recommendation_id = f"rec_{ticket_id}_{str(uuid.uuid4())}"
        self.shared_memory.store(recommendation_id, {"ticket_id": ticket_id, "recommendations": recommendations})
        logger.info(f"Stored recommendations with ID {recommendation_id}")
        return recommendation_id
    
    def _jira_article_generator_node(self, state: JurixState) -> JurixState:
        input_data = {
            "ticket_id": state["ticket_id"],
            "refinement_suggestion": state.get("refinement_suggestion"),
        }
        
        result = self.jira_article_generator.run(input_data)
        
        updated_state = state.copy()
        updated_state["article"] = result.get("article", {})
        updated_state["workflow_status"] = result.get("workflow_status", "failure")
        updated_state["autonomous_refinement_done"] = result.get("autonomous_refinement_done", False)
        
        collaboration_metadata = result.get("collaboration_metadata", {})
        if collaboration_metadata:
            updated_state["collaboration_metadata"] = collaboration_metadata
            updated_state["final_collaboration_summary"] = collaboration_metadata
        
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
            "collaboration_metadata": collaboration_metadata,
            "collaboration_applied": bool(collaboration_metadata.get("collaborating_agents"))
        }
        updated_state["workflow_history"].append(history_entry)
        
        return updated_state
    
    def _knowledge_base_node(self, state: JurixState) -> JurixState:
        input_data = {
            "article": state["article"],
            "validation_context": "comprehensive_quality_assurance",
            "previous_collaboration": state.get("collaboration_metadata", {})
        }
        
        result = self.knowledge_base.run(input_data)
        
        updated_state = state.copy()
        updated_state["redundant"] = result.get("redundant", False)
        
        if not state.get("has_refined", False):
            kb_suggestion = result.get("refinement_suggestion")
            
            if not kb_suggestion or len(kb_suggestion) < 50:
                collaboration_applied = bool(state.get("collaboration_metadata", {}).get("collaborating_agents"))
                
                if collaboration_applied:
                    updated_state["refinement_suggestion"] = (
                        "Enhance the article by adding specific technical implementation details, "
                        "including code examples where applicable, quantifiable metrics showing the impact "
                        "of the resolution, and a detailed timeline of the resolution process. "
                        "Also expand the 'Related Knowledge' section with specific references to similar "
                        "issues and their resolution patterns."
                    )
                else:
                    updated_state["refinement_suggestion"] = (
                        "Significantly enhance the article by adding comprehensive technical details, "
                        "specific performance metrics before and after the resolution, detailed step-by-step "
                        "implementation instructions, code examples, and strategic recommendations for "
                        "preventing similar issues. Include a troubleshooting section for common variations."
                    )
            else:
                updated_state["refinement_suggestion"] = kb_suggestion
                
            updated_state["refinement_count"] = 1
        else:
            updated_state["refinement_suggestion"] = None
            updated_state["refinement_count"] = state.get("refinement_count", 0)
        
        updated_state["workflow_status"] = result.get("workflow_status", "failure")
        updated_state["iteration_count"] = state.get("iteration_count", 0) + 1
        updated_state["workflow_stage"] = "knowledge_base_evaluated"
        
        existing_collab = updated_state.get("collaboration_metadata", {})
        new_collab = result.get("collaboration_metadata", {})
        
        if existing_collab or new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            merged_collab["validation_completed"] = True
            updated_state["collaboration_metadata"] = merged_collab
            updated_state["final_collaboration_summary"] = merged_collab
        
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
            "collaboration_metadata": updated_state.get("collaboration_metadata", {}),
            "validation_quality": "enhanced"
        }
        updated_state["workflow_history"].append(history_entry)
        
        return updated_state
    
    def run_predictive_workflow(self, project_id: str, analysis_type: str = "comprehensive", 
                              conversation_id: str = None) -> JurixState:
        conversation_id = conversation_id or str(uuid.uuid4())
        
        state = JurixState(
            query=f"Generate predictive analysis for {project_id}",
            intent={"intent": "predictive_analysis", "project": project_id},
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
            time_range={
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-17T23:59:59Z"
            },
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
            dashboard_id=None,
            collaboration_metadata=None,
            final_collaboration_summary=None,
            collaboration_insights=None,
            collaboration_trace=None,
            collaborative_agents_used=None,
            predictions=None,
            predictive_insights=None,
            analysis_type=analysis_type
        )
        
        logger.info(f"Starting predictive workflow for {project_id}")
        
        workflow = self._build_predictive_workflow()
        final_state = state
        
        collaboration_trace = []
        
        for event in workflow.stream(state):
            for node_name, node_state in event.items():
                collab_metadata = node_state.get("collaboration_metadata", {})
                if collab_metadata:
                    collaboration_trace.append({
                        "node": node_name,
                        "collaboration": collab_metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                
                final_state = node_state
        
        final_state["collaboration_trace"] = collaboration_trace
        
        return final_state
    
    def _build_predictive_workflow(self):
        workflow = StateGraph(JurixState)
        
        workflow.add_node("jira_data_agent", self._predictive_jira_data_node)
        workflow.add_node("predictive_analysis_agent", self._predictive_analysis_node)
        workflow.add_node("recommendation_agent", self._predictive_recommendation_node)
        
        workflow.set_entry_point("jira_data_agent")
        
        workflow.add_edge("jira_data_agent", "predictive_analysis_agent")
        workflow.add_edge("predictive_analysis_agent", "recommendation_agent")
        workflow.add_edge("recommendation_agent", END)
        
        return workflow.compile()
    
    def _predictive_jira_data_node(self, state: JurixState) -> JurixState:
        input_data = {
            "project_id": state["project_id"],
            "time_range": state["time_range"],
            "analysis_depth": "enhanced",
            "workflow_context": "predictive_analysis"
        }
        
        result = self.jira_data_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["tickets"] = result.get("tickets", [])
        updated_state["workflow_status"] = result.get("workflow_status", "failure")
        updated_state["metadata"] = result.get("metadata", {})
        
        if result.get("collaboration_metadata"):
            updated_state["collaboration_metadata"] = result["collaboration_metadata"]
        
        return updated_state
    
    def _predictive_analysis_node(self, state: JurixState) -> JurixState:
        tickets = state["tickets"]
        
        done_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "Done"])
        in_progress = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "In Progress"])
        to_do = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "To Do"])
        
        velocity = done_tickets
        cycle_times = []
        assignee_counts = {}
        
        for ticket in tickets:
            changelog = ticket.get("changelog", {}).get("histories", [])
            start_date = None
            end_date = None
            
            for history in changelog:
                for item in history.get("items", []):
                    if item.get("field") == "status":
                        if item.get("toString") == "In Progress" and not start_date:
                            start_date = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                        elif item.get("toString") == "Done":
                            end_date = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
            
            if start_date and end_date:
                cycle_time = (end_date - start_date).days
                if cycle_time >= 0:
                    cycle_times.append(cycle_time)
            
            assignee_info = ticket.get("fields", {}).get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 5
        
        metrics = {
            "throughput": velocity,
            "cycle_time": avg_cycle_time,
            "workload": assignee_counts,
            "bottlenecks": {"To Do": to_do, "In Progress": in_progress}
        }
        
        velocity_history = self._extract_velocity_history(tickets)
        historical_data = {"velocity_history": velocity_history}
        
        input_data = {
            "tickets": tickets,
            "metrics": metrics,
            "historical_data": historical_data,
            "user_query": state.get("query", "Generate comprehensive predictive analysis"),
            "analysis_type": state.get("analysis_type", "comprehensive"),
            "collaboration_context": "predictive_intelligence"
        }
        
        result = self.predictive_analysis_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["predictions"] = result.get("predictions", {})
        updated_state["predictive_insights"] = result.get("predictions", {}).get("natural_language_summary", "")
        updated_state["metrics"] = metrics
        
        existing_collab = updated_state.get("collaboration_metadata", {})
        new_collab = result.get("collaboration_metadata", {})
        if new_collab:
            merged_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            updated_state["collaboration_metadata"] = merged_collab
        
        return updated_state
    
    def _predictive_recommendation_node(self, state: JurixState) -> JurixState:
        predictions = state.get("predictions", {})
        warnings = predictions.get("warnings", [])
        
        prompt = f"Based on predictive analysis for {state['project_id']}, provide specific recommendations to address identified risks and improve outcomes."
        
        if warnings:
            prompt += f" Critical warnings: {[w['message'] for w in warnings[:3]]}"
        
        if predictions.get("sprint_completion", {}).get("recommended_actions"):
            prompt += f" Consider these predictive recommendations: {predictions['sprint_completion']['recommended_actions'][:2]}"
        
        input_data = {
            "session_id": state["conversation_id"],
            "user_prompt": prompt,
            "project": state["project_id"],
            "tickets": state["tickets"],
            "workflow_type": "predictive_recommendations",
            "intent": {"intent": "recommendation", "project": state["project_id"]},
            "collaboration_context": "preventive_action_planning",
            "predictions": predictions
        }
        
        result = self.recommendation_agent.run(input_data)
        
        updated_state = state.copy()
        updated_state["recommendations"] = result.get("recommendations", [])
        updated_state["workflow_status"] = result.get("workflow_status", "failure")
        
        existing_collab = updated_state.get("collaboration_metadata", {})
        new_collab = result.get("collaboration_metadata", {})
        if existing_collab or new_collab:
            final_collab = self._merge_collaboration_metadata(updated_state, new_collab)
            final_collab["workflow_completed"] = True
            final_collab["final_agent"] = "recommendation_agent"
            final_collab["workflow_type"] = "predictive_analysis"
            updated_state["collaboration_metadata"] = final_collab
            updated_state["final_collaboration_summary"] = final_collab
        
        return updated_state

orchestrator = Orchestrator()

def run_workflow(query: str, conversation_id: str = None) -> JurixState:
    return orchestrator.run_workflow(query, conversation_id)

def run_productivity_workflow(project_id: str, time_range: Dict[str, str], conversation_id: str = None) -> JurixState:
    return orchestrator.run_productivity_workflow(project_id, time_range, conversation_id)

def run_jira_workflow(ticket_id: str, conversation_id: str = None, project_id: str = "PROJ123") -> JurixState:
    return orchestrator.run_jira_workflow(ticket_id, conversation_id, project_id)

def run_predictive_workflow(project_id: str, analysis_type: str = "comprehensive", conversation_id: str = None) -> JurixState:
    return orchestrator.run_predictive_workflow(project_id, analysis_type, conversation_id)