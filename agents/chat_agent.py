from typing import Dict, Any, Optional, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from agents.jira_data_agent import JiraDataAgent
from agents.recommendation_agent import RecommendationAgent
import logging
import uuid
from datetime import datetime

class ChatAgent(BaseAgent):
    OBJECTIVE = "Coordinate conversation, maintain context, and deliver responses for Agile and software development queries"
    
    def __init__(self, shared_memory: JurixSharedMemory):
        # IMPORTANT: Initialize with redis_client to enable semantic memory
        super().__init__(name="chat_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        self.jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
        self.recommendation_agent = RecommendationAgent(shared_memory)
        
        self.mental_state.capabilities = [
            AgentCapability.GENERATE_RESPONSE,
            AgentCapability.MAINTAIN_CONVERSATION,
            AgentCapability.COORDINATE_AGENTS
        ]
        
        self.mental_state.obligations.extend([
            "generate_response",
            "coordinate_agents",
            "maintain_conversation_context"
        ])
        
        # Log if semantic memory is available
        if hasattr(self.mental_state, 'vector_memory') and self.mental_state.vector_memory:
            self.log("✅ Semantic memory enabled for ChatAgent")
        else:
            self.log("⚠️ Semantic memory not available for ChatAgent")

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        articles = input_data.get("articles", [])
        recommendations = input_data.get("recommendations", [])
        tickets = input_data.get("tickets", [])
        
        self.log(f"[DEBUG] Articles received: {len(articles)} | Recommendations: {len(recommendations)} | Tickets: {len(tickets)}")
        
        # Store beliefs about the current interaction
        self.mental_state.add_belief("session_id", session_id, 0.9, "input")
        self.mental_state.add_belief("user_prompt", user_prompt, 0.9, "input")
        self.mental_state.add_belief("conversation_history", history, 0.8, "memory")
        self.mental_state.add_belief("articles", articles, 0.9, "input")
        self.mental_state.add_belief("recommendations", recommendations, 0.9, "input")
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("intent", input_data.get("intent", {"intent": "generic_question"}), 0.9, "input")

    def _generate_response(self) -> str:
        intent = self.mental_state.get_belief("intent")["intent"]
        prompt = self.mental_state.get_belief("user_prompt")
        history = self.mental_state.get_belief("conversation_history")
        articles = self.mental_state.get_belief("articles")
        recommendations = self.mental_state.get_belief("recommendations")
        tickets = self.mental_state.get_belief("tickets")
        
        # Enhanced context building with cognitive awareness
        history_context = ""
        if history:
            recent_history = history[-5:]
            history_context = "\n".join([f"{entry['role'].upper()}: {entry['content']}" for entry in recent_history])
        else:
            history_context = "No previous context"
        
        article_context = "\n\n".join([f"[Article {i+1}]\nTitle: {a.get('title', 'N/A')}\nContent: {a.get('content', 'N/A')[:200]}..." for i, a in enumerate(articles)]) if articles else ""
        rec_context = "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else ""
        ticket_context = "\n".join([f"Ticket {t.get('key', 'N/A')}: {t.get('fields', {}).get('summary', 'N/A')}" for t in tickets]) if tickets else ""
        
        # Build enhanced prompt with cognitive context
        prompt_template = f"""You are a conversational AI assistant specializing in Agile methodology and software development topics.

CONVERSATION HISTORY:
{history_context}

CURRENT USER QUERY: {prompt}

IMPORTANT: Pay attention to conversation history. For follow-up questions, relate to previous topics discussed.
"""
        
        # Add contextual information if available
        if articles:
            prompt_template += f"\nRELEVANT ARTICLES:\n{article_context}\n"
        if recommendations:
            prompt_template += f"\nRECOMMENDATIONS:\n{rec_context}\n"
        if tickets:
            prompt_template += f"\nRELEVANT TICKETS:\n{ticket_context}\n"
        
        prompt_template += "\nProvide a helpful, accurate response based on the context above. End with 'Anything else I can help with?'"
        
        self.log(f"[DEBUG] Using cognitive model routing for conversational response")
        
        try:
            # Use cognitive model routing for better conversational responses
            response = self.model_manager.generate_for_agent(
                agent_name=self.name,
                prompt=prompt_template,
                agent_capabilities=[cap.value for cap in self.mental_state.capabilities],
                task_type="conversation"
            )
            
            if not response:
                return "No valid response from model. Anything else I can help with?"
            
            self.log(f"[DEBUG] Generated cognitive response: {len(response)} characters")
            return response.strip()
            
        except Exception as e:
            self.log(f"[ERROR] Cognitive generation error: {str(e)}")
            # Fallback to simple generation
            try:
                response = self.model_manager.generate_response(prompt_template, use_cognitive_routing=False)
                return response.strip()
            except Exception as fallback_error:
                self.log(f"[ERROR] Fallback generation error: {str(fallback_error)}")
                return f"Error generating response: {str(e)}. Anything else I can help with?"

    def _act(self) -> Dict[str, Any]:
        try:
            session_id = self.mental_state.get_belief("session_id")
            user_prompt = self.mental_state.get_belief("user_prompt")
            intent = self.mental_state.get_belief("intent")["intent"]
            project = self.mental_state.get_belief("intent").get("project")
            
            if not project:
                self.log("[WARNING] No project specified in intent, fetching tickets for all projects")
            
            # Store this interaction as an experience in semantic memory
            if hasattr(self.mental_state, 'add_experience') and self.mental_state.vector_memory:
                experience_content = f"User asked for {intent} about: {user_prompt}"
                if project:
                    experience_content += f" for project {project}"
                
                self.mental_state.add_experience(
                    experience_description=experience_content,
                    outcome="processing_request",
                    confidence=0.8,
                    metadata={"intent": intent, "project": project, "user_query": user_prompt[:100]}
                )
                self.log("✅ Stored experience in semantic memory")
            else:
                self.log("⚠️ Semantic memory not available - cannot store experience")
            
            if intent == "recommendation" and not self.mental_state.get_belief("tickets"):
                self.log(f"[DEBUG] Requesting ticket context from JiraDataAgent for project {project}")
                jira_input = {
                    "project_id": project,
                    "time_range": {"start": "2025-05-01T00:00:00Z", "end": "2025-05-17T23:59:59Z"}
                }
                tickets_result = self.jira_data_agent.run(jira_input)
                tickets = tickets_result.get("tickets", [])
                self.mental_state.add_belief("tickets", tickets, 0.9, "jira_data_agent")
                self.log(f"[DEBUG] Received {len(tickets)} tickets from JiraDataAgent")

            if intent == "recommendation" and self.mental_state.get_belief("tickets"):
                self.log(f"[DEBUG] Requesting recommendations from RecommendationAgent")
                rec_input = {
                    "session_id": session_id,
                    "user_prompt": user_prompt,
                    "articles": self.mental_state.get_belief("articles"),
                    "project": project,
                    "tickets": self.mental_state.get_belief("tickets"),
                    "workflow_type": "prompting",
                    "intent": self.mental_state.get_belief("intent")
                }
                rec_result = self.recommendation_agent.run(rec_input)
                recommendations = rec_result.get("recommendations", [])
                self.mental_state.add_belief("recommendations", recommendations, 0.9, "recommendation_agent")
                self.log(f"[DEBUG] Received {len(recommendations)} recommendations from RecommendationAgent")

            # Get semantic context for better responses
            semantic_context = ""
            if hasattr(self.mental_state, 'get_semantic_context'):
                semantic_context = self.mental_state.get_semantic_context(user_prompt)
                self.log(f"[DEBUG] Retrieved semantic context: {len(semantic_context)} characters")

            response = self._generate_response()
            self.shared_memory.add_interaction(session_id, "user", user_prompt)
            self.shared_memory.add_interaction(session_id, "assistant", response)
            
            # Record the successful interaction as an experience
            if hasattr(self.mental_state, 'add_experience') and self.mental_state.vector_memory:
                success_content = f"Successfully answered {intent} question about: {user_prompt[:100]}"
                if project:
                    success_content += f" for project {project}"
                
                self.mental_state.add_experience(
                    experience_description=success_content,
                    outcome=f"Generated {len(response)} character response",
                    confidence=0.9,
                    metadata={
                        "intent": intent,
                        "project": project,
                        "response_length": len(response),
                        "articles_used": len(self.mental_state.get_belief("articles") or []),
                        "recommendations_used": len(self.mental_state.get_belief("recommendations") or []),
                        "success": True
                    }
                )
                self.log("✅ Stored successful interaction in semantic memory")
            else:
                self.log("⚠️ Cannot store interaction - semantic memory not available")
            
            # Record the decision
            decision = {
                "action": "generate_response",
                "intent": intent,
                "response_length": len(response),
                "used_articles": len(self.mental_state.get_belief("articles") or []),
                "used_recommendations": len(self.mental_state.get_belief("recommendations") or []),
                "used_tickets": len(self.mental_state.get_belief("tickets") or []),
                "reasoning": f"Processed {intent} query and generated contextual response"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "response": response,
                "articles_used": [{"title": a.get("title", "N/A"), "relevance": a.get("relevance_score", 0)} for a in self.mental_state.get_belief("articles") or []],
                "tickets": self.mental_state.get_belief("tickets") or [],
                "workflow_status": "completed",
                "next_agent": None
            }
        except Exception as e:
            self.log(f"[ERROR] Act error: {str(e)}")
            
            # Record the failure as an experience
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed to process {user_prompt[:50]}",
                    outcome=f"Error: {str(e)}",
                    confidence=0.3,
                    metadata={"error_type": type(e).__name__}
                )
            
            return {
                "error": str(e),
                "workflow_status": "error",
                "response": f"Error processing request: {str(e)}. Please try again. Anything else I can help with?",
                "tickets": self.mental_state.get_belief("tickets") or []
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        # Enhanced reflection with performance analysis
        response_quality = len(action_result.get("response", "")) > 50  # Basic quality check
        
        reflection = {
            "interaction_type": "chat_response",
            "response_quality": response_quality,
            "articles_utilized": len(action_result.get("articles_used", [])),
            "context_maintained": bool(self.mental_state.get_belief("conversation_history")),
            "performance_notes": f"Generated response with {len(action_result.get('response', ''))} characters"
        }
        self.mental_state.add_reflection(reflection)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)