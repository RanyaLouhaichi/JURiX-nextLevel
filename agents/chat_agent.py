# agents/chat_agent.py
# Enhanced ChatAgent that can intelligently coordinate with other agents
# This agent now acts as a conversation orchestrator that can recognize when
# it needs additional expertise and request collaboration

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
    OBJECTIVE = "Coordinate conversation, maintain context, and deliver intelligent responses by collaborating with specialized agents when needed"
    
    def __init__(self, shared_memory: JurixSharedMemory):
        # Initialize with redis_client to enable semantic memory and collaboration
        super().__init__(name="chat_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        self.jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
        self.recommendation_agent = RecommendationAgent(shared_memory)
        
        # Define what this agent can do
        self.mental_state.capabilities = [
            AgentCapability.GENERATE_RESPONSE,
            AgentCapability.MAINTAIN_CONVERSATION,
            AgentCapability.COORDINATE_AGENTS
        ]
        
        # Define what this agent is responsible for
        self.mental_state.obligations.extend([
            "generate_response",
            "coordinate_agents", 
            "maintain_conversation_context",
            "assess_collaboration_needs",  # New collaborative responsibility
            "synthesize_multi_agent_results"  # New collaborative responsibility
        ])
        
        # Log collaborative capabilities
        if hasattr(self.mental_state, 'vector_memory') and self.mental_state.vector_memory:
            self.log("✅ ChatAgent initialized with semantic memory and collaboration capabilities")
        else:
            self.log("⚠️ ChatAgent initialized without semantic memory - limited collaboration abilities")

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        """Enhanced perception that includes collaborative context awareness"""
        super()._perceive(input_data)
        
        # Store core conversation elements
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        articles = input_data.get("articles", [])
        recommendations = input_data.get("recommendations", [])
        tickets = input_data.get("tickets", [])
        intent = input_data.get("intent", {"intent": "generic_question"})

        self.log(f"[PERCEPTION] Processing query: '{user_prompt}' with {len(articles)} articles, {len(recommendations)} recommendations, {len(tickets)} tickets")
        
        # Store beliefs with confidence based on data quality
        self.mental_state.add_belief("session_id", session_id, 0.9, "input")
        self.mental_state.add_belief("user_prompt", user_prompt, 0.9, "input")
        self.mental_state.add_belief("conversation_history", history, 0.8, "memory")
        self.mental_state.add_belief("articles", articles, 0.9, "input")
        self.mental_state.add_belief("recommendations", recommendations, 0.9, "input")
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("intent", intent, 0.9, "input")
        
        # NEW: Check if this is a collaborative request from another agent
        if input_data.get("collaboration_purpose"):
            self.mental_state.add_belief(
                "collaboration_context", 
                input_data.get("collaboration_purpose"), 
                0.9, 
                "collaboration"
            )
            self.log(f"[COLLABORATION] Received collaborative request: {input_data.get('collaboration_purpose')}")
        
        # NEW: Store primary agent results if this is a collaboration
        if input_data.get("primary_agent_result"):
            self.mental_state.add_belief(
                "primary_agent_context",
                input_data.get("primary_agent_result"),
                0.8,
                "collaboration"
            )
            
        # NEW: Assess if we need collaboration for this request
        self._assess_collaboration_needs(input_data)

    def _assess_collaboration_needs(self, input_data: Dict[str, Any]) -> None:
        """
        Intelligent assessment of whether this agent needs help from other agents
        This is the key innovation - agents become self-aware about their limitations
        """
        user_prompt = input_data.get("user_prompt", "").lower()
        intent = input_data.get("intent", {}).get("intent", "generic_question")
        articles = input_data.get("articles", [])
        recommendations = input_data.get("recommendations", [])
        tickets = input_data.get("tickets", [])
        
        # Check if we have insufficient context for a good response
        context_quality = self._assess_context_quality(input_data)
        
        if context_quality < 0.6:  # Low quality context
            self.log("[COLLABORATION NEED] Insufficient context detected")
            
            # If intent is recommendation but we have no recommendations, request help
            if intent == "recommendation" and not recommendations:
                self.mental_state.request_collaboration(
                    agent_type="recommendation_agent",
                    reasoning_type="strategic_reasoning",
                    context={
                        "reason": "missing_recommendations", 
                        "user_query": user_prompt,
                        "available_tickets": len(tickets)
                    }
                )
                
            # If query mentions specific projects but we have no ticket data, request help
            if any(keyword in user_prompt for keyword in ["project", "ticket", "jira"]) and not tickets:
                self.mental_state.request_collaboration(
                    agent_type="jira_data_agent",
                    reasoning_type="data_analysis",
                    context={
                        "reason": "missing_project_data",
                        "user_query": user_prompt
                    }
                )
        
        # Check if query complexity exceeds our confidence threshold
        query_complexity = self._assess_query_complexity(user_prompt)
        if query_complexity > 0.7 and not self._has_sufficient_expertise(user_prompt):
            self.log("[COLLABORATION NEED] Complex query detected, may need specialist help")
            
            # For productivity/analytics queries, suggest dashboard agent collaboration
            if any(keyword in user_prompt for keyword in ["productivity", "analytics", "dashboard", "metrics", "performance"]):
                self.mental_state.request_collaboration(
                    agent_type="productivity_dashboard_agent",
                    reasoning_type="data_analysis",
                    context={
                        "reason": "analytics_expertise_needed",
                        "complexity_score": query_complexity
                    }
                )

    def _assess_context_quality(self, input_data: Dict[str, Any]) -> float:
        """Assess the quality of available context for generating a good response"""
        quality_score = 0.0
        
        # Check conversation history
        history = input_data.get("conversation_history", [])
        if history:
            quality_score += 0.2
            
        # Check available articles
        articles = input_data.get("articles", [])
        if articles:
            quality_score += 0.3
            
        # Check recommendations
        recommendations = input_data.get("recommendations", [])
        if recommendations:
            quality_score += 0.3
            
        # Check ticket data relevance
        tickets = input_data.get("tickets", [])
        intent = input_data.get("intent", {}).get("intent", "")
        if tickets and intent in ["recommendation", "project_analysis"]:
            quality_score += 0.2
            
        return min(quality_score, 1.0)

    def _assess_query_complexity(self, user_prompt: str) -> float:
        """Assess how complex a query is to determine if collaboration is needed"""
        complexity_indicators = [
            "analyze", "compare", "evaluate", "recommend", "optimize", 
            "forecast", "predict", "trend", "pattern", "correlation",
            "bottleneck", "efficiency", "performance", "metrics"
        ]
        
        prompt_lower = user_prompt.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in prompt_lower)
        
        # Normalize to 0-1 scale
        return min(complexity_score / len(complexity_indicators), 1.0)

    def _has_sufficient_expertise(self, user_prompt: str) -> bool:
        """Check if this agent has sufficient past experience with similar queries"""
        if not hasattr(self.mental_state, 'recall_similar_experiences'):
            return False
            
        try:
            similar_experiences = self.mental_state.recall_similar_experiences(user_prompt, max_results=3)
            successful_experiences = [exp for exp in similar_experiences if exp.confidence > 0.7]
            return len(successful_experiences) >= 2
        except Exception as e:
            self.log(f"[ERROR] Failed to check expertise: {e}")
            return False

    def _generate_response(self) -> str:
        """Enhanced response generation that can incorporate collaborative results"""
        intent = self.mental_state.get_belief("intent")["intent"]
        prompt = self.mental_state.get_belief("user_prompt")
        history = self.mental_state.get_belief("conversation_history")
        articles = self.mental_state.get_belief("articles")
        recommendations = self.mental_state.get_belief("recommendations")
        tickets = self.mental_state.get_belief("tickets")
        
        # NEW: Check if we have collaborative context to incorporate
        collaboration_context = self.mental_state.get_belief("collaboration_context")
        primary_agent_context = self.mental_state.get_belief("primary_agent_context")
        
        # Enhanced context building with collaborative awareness
        history_context = ""
        if history:
            recent_history = history[-5:]
            history_context = "\n".join([f"{entry['role'].upper()}: {entry['content']}" for entry in recent_history])
        else:
            history_context = "No previous context"
        
        article_context = "\n\n".join([
            f"[Article {i+1}]\nTitle: {a.get('title', 'N/A')}\nContent: {a.get('content', 'N/A')[:200]}..." 
            for i, a in enumerate(articles)
        ]) if articles else ""
        
        rec_context = "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else ""
        ticket_context = "\n".join([
            f"Ticket {t.get('key', 'N/A')}: {t.get('fields', {}).get('summary', 'N/A')}" 
            for t in tickets
        ]) if tickets else ""
        
        # NEW: Build collaborative context
        collaborative_context = ""
        if collaboration_context or primary_agent_context:
            collaborative_context = "\n\nCOLLABORATIVE CONTEXT:\n"
            if collaboration_context:
                collaborative_context += f"Collaboration Purpose: {collaboration_context}\n"
            if primary_agent_context:
                collaborative_context += f"Previous Analysis: {str(primary_agent_context)[:200]}...\n"

        # Build enhanced prompt with cognitive and collaborative awareness
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
        if collaborative_context:
            prompt_template += collaborative_context
        
        prompt_template += "\nProvide a helpful, accurate response based on the context above. End with 'Anything else I can help with?'"
        
        self.log(f"[RESPONSE GENERATION] Using cognitive model routing for enhanced response")
        
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
            
            self.log(f"[RESPONSE] Generated {len(response)} character response")
            return response.strip()
            
        except Exception as e:
            self.log(f"[ERROR] Response generation failed: {str(e)}")
            # Fallback to simple generation
            try:
                response = self.model_manager.generate_response(prompt_template, use_cognitive_routing=False)
                return response.strip()
            except Exception as fallback_error:
                self.log(f"[ERROR] Fallback generation error: {str(fallback_error)}")
                return f"Error generating response: {str(e)}. Anything else I can help with?"

    def _act(self) -> Dict[str, Any]:
        """Enhanced action method that handles both regular and collaborative workflows"""
        try:
            session_id = self.mental_state.get_belief("session_id")
            user_prompt = self.mental_state.get_belief("user_prompt")
            intent = self.mental_state.get_belief("intent")["intent"]
            project = self.mental_state.get_belief("intent").get("project")
            
            # NEW: Check if this is a collaborative response
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            is_collaborative = bool(collaboration_context)
            
            if not project:
                self.log("[WARNING] No project specified in intent")
            
            # Store this interaction as an experience in semantic memory
            if hasattr(self.mental_state, 'add_experience') and self.mental_state.vector_memory:
                experience_content = f"{'Collaborative ' if is_collaborative else ''}response for {intent}: {user_prompt}"
                if project:
                    experience_content += f" for project {project}"
                
                self.mental_state.add_experience(
                    experience_description=experience_content,
                    outcome="processing_chat_request",
                    confidence=0.8,
                    metadata={
                        "intent": intent, 
                        "project": project, 
                        "user_query": user_prompt[:100],
                        "is_collaborative": is_collaborative,
                        "collaboration_context": collaboration_context
                    }
                )
                self.log("✅ Stored interaction experience in semantic memory")

            # Handle collaborative data requests
            if intent == "recommendation" and not self.mental_state.get_belief("tickets"):
                self.log(f"[COLLABORATION] Requesting ticket context from JiraDataAgent for project {project}")
                jira_input = {
                    "project_id": project,
                    "time_range": {"start": "2025-05-01T00:00:00Z", "end": "2025-05-17T23:59:59Z"}
                }
                tickets_result = self.jira_data_agent.run(jira_input)
                tickets = tickets_result.get("tickets", [])
                self.mental_state.add_belief("tickets", tickets, 0.9, "jira_data_agent")
                self.log(f"[COLLABORATION] Received {len(tickets)} tickets from JiraDataAgent")

            # Handle collaborative recommendations
            if intent == "recommendation" and self.mental_state.get_belief("tickets"):
                self.log(f"[COLLABORATION] Requesting recommendations from RecommendationAgent")
                rec_input = {
                    "session_id": session_id,
                    "user_prompt": user_prompt,
                    "articles": self.mental_state.get_belief("articles"),
                    "project": project,
                    "tickets": self.mental_state.get_belief("tickets"),
                    "workflow_type": "collaborative_chat",
                    "intent": self.mental_state.get_belief("intent")
                }
                rec_result = self.recommendation_agent.run(rec_input)
                recommendations = rec_result.get("recommendations", [])
                self.mental_state.add_belief("recommendations", recommendations, 0.9, "recommendation_agent")
                self.log(f"[COLLABORATION] Received {len(recommendations)} recommendations")

            # Get semantic context for better responses
            semantic_context = ""
            if hasattr(self.mental_state, 'get_semantic_context'):
                semantic_context = self.mental_state.get_semantic_context(user_prompt)
                self.log(f"[SEMANTIC] Retrieved {len(semantic_context)} characters of context")

            # Generate the response
            response = self._generate_response()
            
            # Update conversation history
            self.shared_memory.add_interaction(session_id, "user", user_prompt)
            self.shared_memory.add_interaction(session_id, "assistant", response)
            
            # Record successful interaction as experience
            if hasattr(self.mental_state, 'add_experience') and self.mental_state.vector_memory:
                success_content = f"Successfully {'collaborated on' if is_collaborative else 'handled'} {intent} query: {user_prompt[:100]}"
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
                        "success": True,
                        "collaborative": is_collaborative
                    }
                )
                self.log("✅ Stored successful interaction in semantic memory")
            
            # Record the decision with collaborative context
            decision = {
                "action": "generate_response",
                "intent": intent,
                "response_length": len(response),
                "used_articles": len(self.mental_state.get_belief("articles") or []),
                "used_recommendations": len(self.mental_state.get_belief("recommendations") or []),
                "used_tickets": len(self.mental_state.get_belief("tickets") or []),
                "collaborative": is_collaborative,
                "collaboration_requests": len(self.mental_state.collaborative_requests),
                "reasoning": f"{'Collaborative ' if is_collaborative else ''}response for {intent} query"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "response": response,
                "articles_used": [
                    {"title": a.get("title", "N/A"), "relevance": a.get("relevance_score", 0)} 
                    for a in self.mental_state.get_belief("articles") or []
                ],
                "tickets": self.mental_state.get_belief("tickets") or [],
                "workflow_status": "completed",
                "collaboration_metadata": {
                    "is_collaborative": is_collaborative,
                    "collaboration_context": collaboration_context,
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests)
                },
                "next_agent": None
            }
            
        except Exception as e:
            self.log(f"[ERROR] Action failed: {str(e)}")
            
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
                "tickets": self.mental_state.get_belief("tickets") or [],
                "collaboration_metadata": {
                    "is_collaborative": False,
                    "error_occurred": True
                }
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        """Enhanced reflection that includes collaborative performance analysis"""
        super()._rethink(action_result)
        
        # Analyze response quality and collaborative effectiveness
        response_quality = len(action_result.get("response", "")) > 50
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        was_collaborative = collaboration_metadata.get("is_collaborative", False)
        
        # Enhanced reflection with collaborative analysis
        reflection = {
            "interaction_type": "chat_response",
            "response_quality": response_quality,
            "articles_utilized": len(action_result.get("articles_used", [])),
            "context_maintained": bool(self.mental_state.get_belief("conversation_history")),
            "collaborative_interaction": was_collaborative,
            "collaboration_requests_made": collaboration_metadata.get("collaboration_requests_made", 0),
            "performance_notes": f"Generated {'collaborative ' if was_collaborative else ''}response with {len(action_result.get('response', ''))} characters"
        }
        self.mental_state.add_reflection(reflection)
        
        # Learn from collaborative outcomes
        if was_collaborative:
            collaboration_success = action_result.get("workflow_status") == "completed"
            self.mental_state.add_experience(
                experience_description=f"Collaborative interaction {'succeeded' if collaboration_success else 'failed'}",
                outcome=f"collaboration_{'success' if collaboration_success else 'failure'}",
                confidence=0.8 if collaboration_success else 0.4,
                metadata={
                    "collaboration_type": collaboration_metadata.get("collaboration_context"),
                    "response_quality": response_quality
                }
            )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point - processes both regular and collaborative requests"""
        return self.process(input_data)