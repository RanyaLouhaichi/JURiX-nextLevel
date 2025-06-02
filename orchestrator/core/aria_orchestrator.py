# orchestrator/core/aria_orchestrator.py
# FIXED VERSION - Handles async/sync properly

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import uuid
from dataclasses import dataclass
from enum import Enum
import threading

from orchestrator.core.orchestrator import Orchestrator # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore

class ARIAPersonality(Enum):
    HELPFUL = "helpful"
    ANALYTICAL = "analytical"
    PROACTIVE = "proactive"
    CREATIVE = "creative"

@dataclass
class ARIAContext:
    """ARIA's understanding of the current situation"""
    user_intent: str
    workspace: str  # "jira" or "confluence"
    current_project: Optional[str]
    current_ticket: Optional[str]
    user_history: List[Dict[str, Any]]
    active_insights: List[Dict[str, Any]]
    mood: ARIAPersonality

class ARIA:
    """
    ARIA - Artificial Reasoning Intelligence Assistant
    Your AI Team Member that lives in Jira and Confluence
    """
    
    def __init__(self):
        self.orchestrator = Orchestrator()
        self.logger = logging.getLogger("ARIA")
        self.personality = ARIAPersonality.HELPFUL
        
        # ARIA's memory of interactions
        self.conversation_contexts = {}
        self.active_monitoring = {}
        self.predictive_insights = []
        
        # Real-time event handlers
        self.event_handlers = {
            "ticket_resolved": self._handle_ticket_resolution,
        }
        
        self.logger.info("🤖 ARIA initialized - Your AI team member is ready!")
    
    def introduce_myself(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """ARIA introduces herself to the team (synchronous version)"""
        return {
            "avatar": "aria_animated.gif",  # Future: animated avatar
            "message": "Hi! I'm ARIA, your AI team member. I live right here in Jira and Confluence to help you work smarter. I can analyze your projects, create documentation, predict issues, and answer any questions you have!",
            "capabilities": [
                "Real-time project analytics",
                "Automatic documentation",
                "Predictive insights",
                "Natural conversation",
                "Proactive assistance"
            ],
            "status": "active",
            "current_mood": self.personality.value
        }
    
    def process_interaction(self, user_input: str, context: ARIAContext) -> Dict[str, Any]:
        """
        ARIA processes any interaction from Jira/Confluence (synchronous version)
        """
        self.logger.info(f"🤖 ARIA processing: {user_input} in {context.workspace}")
        
        # Determine what the user needs
        intent = self._analyze_intent(user_input, context)
        
        # Route to appropriate workflow
        if intent == "dashboard_request":
            return self._provide_live_dashboard(context)
        elif intent == "ticket_help":
            return self._assist_with_ticket(context)
        elif intent == "documentation":
            return self._handle_documentation(context)
        elif intent == "prediction":
            return self._provide_predictions(context)
        else:
            return self._general_conversation(user_input, context)
    
    def _provide_live_dashboard(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA provides live dashboard data for Jira plugin"""
        self.logger.info(f"📊 ARIA generating live dashboard for {context.current_project}")
        
        # Use productivity workflow
        state = self.orchestrator.run_productivity_workflow(
            context.current_project or "PROJ123",
            {
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-17T23:59:59Z"
            }
        )
        
        # Transform for plugin display
        dashboard_data = {
            "type": "live_dashboard",
            "data": {
                "sprint_health": self._calculate_sprint_health(state),
                "live_metrics": {
                    "velocity": state["metrics"].get("throughput", 0),
                    "cycle_time": state["metrics"].get("cycle_time", 0),
                    "bottlenecks": state["metrics"].get("bottlenecks", {}),
                    "team_workload": state["metrics"].get("workload", {})
                },
                "predictions": self._generate_predictions(state),
                "visualizations": state.get("visualization_data", {}),
                "recommendations": state.get("recommendations", [])
            },
            "aria_insights": [
                {
                    "type": "observation",
                    "message": f"I notice your team's velocity is {state['metrics'].get('throughput', 0)} tickets/week",
                    "sentiment": "positive" if state['metrics'].get('throughput', 0) > 5 else "concern"
                }
            ],
            "refresh_interval": 30,  # Real-time update every 30s
            "aria_message": "Here's your real-time dashboard! I'll keep it updated every 30 seconds. Ask me anything about what you see!"
        }
        
        return dashboard_data
    
    def _handle_ticket_resolution(self, ticket_id: str, context: ARIAContext) -> Dict[str, Any]:
        """ARIA handles automatic documentation when ticket is resolved"""
        self.logger.info(f"📝 ARIA handling ticket resolution for {ticket_id}")
        
        # Run article generation workflow
        state = self.orchestrator.run_jira_workflow(ticket_id)
        
        # Return success notification
        return {
            "type": "notification",
            "stage": "complete",
            "message": f"✅ Documentation created! I've posted it to Confluence.",
            "avatar_state": "happy",
            "actions": [
                {
                    "label": "View Documentation",
                    "url": f"/confluence/articles/{state.get('article', {}).get('id', '')}"
                },
                {
                    "label": "Ask me about it",
                    "action": "chat_with_aria"
                }
            ],
            "article_preview": state.get("article", {}).get("content", "")[:200] + "...",
            "collaboration_applied": bool(state.get("collaboration_metadata"))
        }
    
    def _assist_with_ticket(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA assists with ticket-related questions"""
        ticket_id = context.current_ticket or "TICKET-001"
        
        # Get ticket data
        jira_data = self.orchestrator.jira_data_agent.run({
            "project_id": context.current_project or "PROJ123",
            "time_range": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-12-31T23:59:59Z"
            }
        })
        
        tickets = jira_data.get("tickets", [])
        target_ticket = next((t for t in tickets if t.get("key") == ticket_id), None)
        
        if target_ticket:
            return {
                "type": "ticket_assistance",
                "message": f"I found information about {ticket_id}. This ticket is about '{target_ticket.get('fields', {}).get('summary', 'No summary')}' and is currently {target_ticket.get('fields', {}).get('status', {}).get('name', 'Unknown')}.",
                "ticket_data": {
                    "key": ticket_id,
                    "summary": target_ticket.get('fields', {}).get('summary'),
                    "status": target_ticket.get('fields', {}).get('status', {}).get('name'),
                    "assignee": target_ticket.get('fields', {}).get('assignee', {}).get('displayName') if target_ticket.get('fields', {}).get('assignee') else "Unassigned"
                },
                "suggestions": [
                    "Would you like me to analyze similar tickets?",
                    "Should I create documentation for this?",
                    "Want to see the team's productivity metrics?"
                ]
            }
        else:
            return {
                "type": "ticket_assistance",
                "message": f"I couldn't find {ticket_id}. Would you like me to search for similar tickets or help you create a new one?",
                "suggestions": ["Search similar tickets", "Create new ticket", "Show all tickets"]
            }
    
    def _handle_documentation(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA handles documentation requests"""
        return {
            "type": "documentation_help",
            "message": "I can help you with documentation! What would you like to do?",
            "options": [
                {
                    "label": "Create documentation from ticket",
                    "action": "create_from_ticket"
                },
                {
                    "label": "Find existing documentation",
                    "action": "search_docs"
                },
                {
                    "label": "Improve current page",
                    "action": "improve_page"
                },
                {
                    "label": "Generate documentation report",
                    "action": "doc_report"
                }
            ],
            "recent_docs": [
                "API Migration Guide",
                "Deployment Best Practices",
                "Team Onboarding"
            ]
        }
    
    def _provide_predictions(self, context: ARIAContext) -> Dict[str, Any]:
        """ARIA provides predictive insights"""
        # Get current state
        state = self.orchestrator.run_productivity_workflow(
            context.current_project or "PROJ123",
            {
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-17T23:59:59Z"
            }
        )
        
        predictions = self._generate_predictions(state)
        
        return {
            "type": "predictions",
            "message": "Based on my analysis of your project patterns, here's what I predict:",
            "predictions": predictions,
            "confidence_level": "high",
            "aria_advice": "I recommend addressing the bottlenecks first to improve sprint completion likelihood."
        }
    
    def _calculate_sprint_health(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sprint health metrics"""
        metrics = state.get("metrics", {})
        
        # Simple health calculation
        health_score = 0
        if metrics.get("throughput", 0) > 5:
            health_score += 40
        if metrics.get("cycle_time", 0) < 5:
            health_score += 30
        if len(metrics.get("bottlenecks", {})) < 3:
            health_score += 30
        
        return {
            "score": health_score,
            "percentage": f"{health_score}%",
            "status": "healthy" if health_score > 70 else "at_risk" if health_score > 40 else "critical",
            "factors": {
                "velocity": "good" if metrics.get("throughput", 0) > 5 else "low",
                "cycle_time": "good" if metrics.get("cycle_time", 0) < 5 else "high",
                "bottlenecks": len(metrics.get("bottlenecks", {}))
            }
        }
    
    def _generate_predictions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictive insights"""
        predictions = []
        metrics = state.get("metrics", {})
        
        # Sprint completion prediction
        if metrics.get("throughput", 0) < 5:
            predictions.append({
                "type": "sprint_risk",
                "confidence": 0.85,
                "message": "Sprint completion at risk - current velocity is below target",
                "suggestion": "Consider redistributing 2-3 tickets to next sprint"
            })
        
        # Bottleneck prediction
        bottlenecks = metrics.get("bottlenecks", {})
        for status, count in bottlenecks.items():
            if count > 3:
                predictions.append({
                    "type": "bottleneck",
                    "confidence": 0.9,
                    "message": f"{count} tickets stuck in {status}",
                    "suggestion": f"Review {status} tickets for blockers"
                })
        
        # Team workload prediction
        workload = metrics.get("workload", {})
        if workload:
            max_load = max(workload.values()) if workload.values() else 0
            min_load = min(workload.values()) if workload.values() else 0
            if max_load > min_load * 2 and max_load > 5:
                predictions.append({
                    "type": "workload_imbalance",
                    "confidence": 0.8,
                    "message": "Significant workload imbalance detected",
                    "suggestion": "Redistribute tickets for better team balance"
                })
        
        return predictions
    
    def _general_conversation(self, user_input: str, context: ARIAContext) -> Dict[str, Any]:
        """ARIA has a general conversation using enhanced ChatAgent"""
        conversation_id = context.user_history[-1].get("conversation_id") if context.user_history else str(uuid.uuid4())
        
        # Run general workflow
        state = self.orchestrator.run_workflow(user_input, conversation_id)
        
        return {
            "type": "chat_response",
            "message": state.get("response", "I'm not sure how to help with that. Can you provide more details?"),
            "avatar_state": "talking",
            "context_used": {
                "articles": len(state.get("articles", [])),
                "recommendations": len(state.get("recommendations", [])),
                "collaboration": bool(state.get("collaboration_metadata"))
            },
            "suggested_actions": self._suggest_followup_actions(state)
        }
    
    def _suggest_followup_actions(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest follow-up actions based on conversation"""
        actions = []
        
        if state.get("recommendations"):
            actions.append({
                "label": "View Recommendations",
                "action": "show_recommendations"
            })
        
        if state.get("articles"):
            actions.append({
                "label": "Related Articles",
                "action": "show_articles"
            })
        
        actions.append({
            "label": "Show Dashboard",
            "action": "show_dashboard"
        })
        
        return actions
    
    # In aria_orchestrator.py, simplify the _analyze_intent method:

    def _analyze_intent(self, user_input: str, context: ARIAContext) -> str:
        """Analyze user intent from input"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["dashboard", "metrics", "analytics", "status"]):
            return "dashboard_request"
        elif "ticket" in input_lower and any(word in input_lower for word in ["help", "status", "info"]):
            return "ticket_help"
        elif any(phrase in input_lower for phrase in ["create documentation", "improve documentation", "write article"]):
            return "documentation"
        else:
            return "general"
    
    def start_monitoring(self, project_id: str):
        """ARIA starts monitoring a project proactively"""
        self.active_monitoring[project_id] = {
            "started": datetime.now(),
            "last_check": datetime.now(),
            "insights_generated": 0
        }
        
        # Run in background thread
        thread = threading.Thread(target=self._monitor_project_sync, args=(project_id,))
        thread.daemon = True
        thread.start()
    
    def _monitor_project_sync(self, project_id: str):
        """Background monitoring for proactive insights (synchronous)"""
        import time
        
        while project_id in self.active_monitoring:
            time.sleep(30)  # Check every 30 seconds
            
            # Run analysis
            state = self.orchestrator.run_productivity_workflow(
                project_id,
                {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-17T23:59:59Z"
                }
            )
            
            # Check for issues
            insights = self._check_for_issues(state)
            if insights:
                self.predictive_insights.extend(insights)
                self.active_monitoring[project_id]["insights_generated"] += len(insights)
    
    def _check_for_issues(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for issues in project state"""
        insights = []
        metrics = state.get("metrics", {})
        
        # Check for sudden velocity drop
        if metrics.get("throughput", 0) < 3:
            insights.append({
                "type": "velocity_alert",
                "severity": "high",
                "message": "Team velocity has dropped below critical threshold",
                "suggestion": "Check for blockers or team availability issues"
            })
        
        # Check for bottleneck growth
        bottlenecks = metrics.get("bottlenecks", {})
        critical_bottlenecks = [status for status, count in bottlenecks.items() if count > 5]
        if critical_bottlenecks:
            insights.append({
                "type": "bottleneck_alert",
                "severity": "medium",
                "message": f"Critical bottlenecks in: {', '.join(critical_bottlenecks)}",
                "suggestion": "Immediate review needed for stuck tickets"
            })
        
        return insights

# Create global ARIA instance
aria = ARIA()