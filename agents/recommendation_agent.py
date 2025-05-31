from datetime import datetime
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
import json
import threading
import time
import logging

logging.basicConfig(
    filename='recommendation_agent.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class RecommendationAgent(BaseAgent):
    OBJECTIVE = "Provide actionable recommendations for Agile and software development queries, tailored to the workflow context"

    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="recommendation_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        
        self.mental_state.capabilities = [
            AgentCapability.PROVIDE_RECOMMENDATIONS
        ]
        
        self.mental_state.obligations.extend([
            "generate_recommendations",
            "check_for_updates"
        ])
        
        self.log(f"Initialized RecommendationAgent with Redis shared memory")
        
        # Start the monitoring thread
        self.monitoring_thread = threading.Thread(target=self._check_for_updates_loop, daemon=True)
        self.monitoring_thread.start()

    def _check_for_updates(self):
        try:
            project_id = self.mental_state.get_belief("project")
            if not project_id:
                self.log("No project ID available for updates check")
                return
                
            # Check Redis for ticket updates
            if self.shared_memory.has_ticket_updates(project_id):
                self.log(f"Detected changes for project {project_id}")
                tickets = self.shared_memory.get_tickets(project_id)
                
                if tickets:
                    self.mental_state.add_belief("tickets", tickets, 0.9, "redis_update")
                    self.log(f"Updated tickets for project {project_id}: {len(tickets)} tickets")
                    
                    recommendations = self._generate_recommendations()
                    self.mental_state.add_belief("recommendations", recommendations, 0.9, "auto_update")
                    self.log(f"Generated new recommendations for project {project_id}: {recommendations}")
                    
                    # Mark updates as processed
                    self.shared_memory.mark_updates_processed(project_id)
            
        except Exception as e:
            self.log(f"[ERROR] Failed to check for updates: {str(e)}")

    def _check_for_updates_loop(self):
        while True:
            try:
                self._check_for_updates()
                self.log("Sleeping for 10 seconds before next update check")
                time.sleep(10)
            except Exception as e:
                self.log(f"[ERROR] Update loop error: {str(e)}")
                time.sleep(30)

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        articles = input_data.get("articles", [])
        project = input_data.get("project")
        tickets = input_data.get("tickets", [])
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        workflow_type = input_data.get("workflow_type", "prompting")
        intent = input_data.get("intent", {}).get("intent", "generic_question")

        self.log(f"Perceiving input data: session_id={session_id}, user_prompt={user_prompt}, project={project}, tickets_count={len(tickets)}, workflow_type={workflow_type}")
        self.log(f"[DEBUG] Tickets received: {tickets}")

        # Store beliefs about the current request
        self.mental_state.add_belief("session_id", session_id, 0.9, "input")
        self.mental_state.add_belief("user_prompt", user_prompt, 0.9, "input")
        self.mental_state.add_belief("conversation_history", history, 0.8, "memory")
        self.mental_state.add_belief("articles", articles, 0.9, "input")
        self.mental_state.add_belief("project", project, 0.9, "input")
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("workflow_type", workflow_type, 0.9, "input")
        self.mental_state.add_belief("intent", intent, 0.9, "input")

    def _generate_recommendations(self) -> List[str]:
        user_prompt = self.mental_state.get_belief("user_prompt")
        project = self.mental_state.get_belief("project")
        conversation_history = self.mental_state.get_belief("conversation_history")
        articles = self.mental_state.get_belief("articles")
        tickets = self.mental_state.get_belief("tickets") or []

        self.log(f"Generating recommendations for project {project} with {len(tickets)} tickets")

        # Check if we have sufficient context
        if not tickets:
            self.log("[DEBUG] Insufficient ticket data, requesting more context")
            # Store a flag in Redis instead of the old memory dict
            self.shared_memory.store("needs_context", {"value": True, "timestamp": datetime.now().isoformat()})
            return ["Please provide more project-specific details for better recommendations."]

        # Build context from available data
        article_context = ""
        if articles:
            article_summaries = [f"[Article {i+1}] {article.get('title', 'No Title')}: {article.get('content', '')[:200]}..." for i, article in enumerate(articles)]
            article_context = "\n".join(article_summaries)
            self.log(f"Article context: {article_context}")

        history_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]
            history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
            self.log(f"History context: {history_context}")

        project_context = ""
        if project and tickets:
            status_distribution = {"To Do": 0, "In Progress": 0, "Done": 0}
            cycle_times = []
            for ticket in tickets:
                status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                if status in status_distribution:
                    status_distribution[status] += 1
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
            avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
            project_context = (
                f"Project: {project}\n"
                f"Status Distribution: {status_distribution}\n"
                f"Cycle Time (In Progress to Done): {avg_cycle_time} days"
            )
            self.log(f"Project context: {project_context}")
        else:
            project_context = "No project-specific ticket data available."
            self.log("No tickets available for project context")

        prompt_template = f"""<|system|>You are an AI specialized in Agile methodologies and software development best practices. 
        Generate 3-5 specific, actionable recommendations related to this query: "{user_prompt}".
        
        Use project-specific data to tailor your recommendations, focusing on velocity, bottlenecks, resource allocation, or process improvements.
        
        Provide detailed explanations for each recommendation, formatted as separate strings in a list, without numbering or bullet points.
        Adopt a conversational tone, explaining the reasoning behind each suggestion.
        
        Relevant context from conversation:
        {history_context}
        
        Relevant articles:
        {article_context}
        
        Project-specific data:
        {project_context}
        
        Return ONLY the list of recommendations.<|assistant|>"""

        try:
            response = self.model_manager.generate_response(prompt_template)
            self.log(f"[DEBUG] Raw LLM recommendation response: {response}")
            
            recommendations = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-") and not line.startswith("*"):
                    if line[0].isdigit() and len(line) > 2 and line[1] in [')', '.', ':']:
                        line = line[2:].strip()
                    recommendations.append(line)
            
            if not recommendations:
                recommendations = ["Consider reviewing Agile best practices for general improvements."]
            self.log(f"[DEBUG] Processed recommendations: {recommendations}")
            return recommendations
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate recommendations with LLM: {e}")
            return [
                "Consider automating testing to reduce delays in the 'In Progress' phase, as this can streamline workflows.",
                "Schedule a team training session on CI/CD optimizations to boost efficiency.",
                "Review workload distribution to balance the 'To Do' and 'In Progress' stages."
            ]

    def _act(self) -> Dict[str, Any]:
        try:
            # Store the recommendation request as an experience
            user_prompt = self.mental_state.get_belief("user_prompt")
            project = self.mental_state.get_belief("project")
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Asked to provide recommendations for project {project}",
                    outcome="generating_recommendations",
                    confidence=0.8,
                    metadata={
                        "project": project,
                        "prompt": user_prompt[:100] if user_prompt else ""
                    }
                )
            
            recommendations = self._generate_recommendations()
            self.log(f"Action completed, returning {len(recommendations)} recommendations")
            
            # Store the successful recommendation generation as an experience
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Generated {len(recommendations)} recommendations for {project}",
                    outcome="recommendations_generated",
                    confidence=0.9,
                    metadata={
                        "project": project,
                        "recommendation_count": len(recommendations),
                        "recommendations": recommendations[:3]  # Store first 3 as sample
                    }
                )
            
            # Check if we need more context (from Redis)
            needs_context_data = self.shared_memory.get("needs_context")
            needs_context = needs_context_data.get("value", False) if needs_context_data else False
            
            # Record the decision
            decision = {
                "action": "generate_recommendations",
                "recommendation_count": len(recommendations),
                "context_sufficient": not needs_context,
                "project": self.mental_state.get_belief("project"),
                "reasoning": f"Generated {len(recommendations)} recommendations based on available context"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "recommendations": recommendations,
                "workflow_status": "success",
                "needs_context": needs_context
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to generate recommendations: {e}")
            
            # Store the failure as an experience
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed to generate recommendations",
                    outcome=f"Error: {str(e)}",
                    confidence=0.2,
                    metadata={"error_type": type(e).__name__}
                )
            
            return {
                "recommendations": [],
                "workflow_status": "failure",
                "needs_context": True
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        # Enhanced reflection with recommendation quality analysis
        recommendations = action_result.get("recommendations", [])
        quality_score = min(len(recommendations) / 3.0, 1.0)  # Target 3+ recommendations
        
        reflection = {
            "operation": "recommendation_generation",
            "success": action_result.get("workflow_status") == "success",
            "recommendation_count": len(recommendations),
            "quality_score": quality_score,
            "context_availability": not action_result.get("needs_context", True),
            "performance_notes": f"Generated {len(recommendations)} recommendations with quality score {quality_score}"
        }
        self.mental_state.add_reflection(reflection)
        
        self.log(f"Rethink completed: {reflection}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)