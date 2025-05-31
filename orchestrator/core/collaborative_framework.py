# orchestrator/core/collaborative_framework.py
# This is the intelligence coordinator that transforms your system from sequential to collaborative
# Think of this as the "conductor" that helps your agents work together intelligently

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json
import redis
import time

from agents.base_agent import BaseAgent, AgentCapability

class CollaborationStrategy(Enum):
    """Different ways agents can collaborate - like different musical arrangements"""
    SEQUENTIAL = "sequential"      # One agent after another (like a relay race)
    PARALLEL = "parallel"         # Multiple agents at once (like a choir)
    ADAPTIVE = "adaptive"         # Dynamically determine the best approach (like jazz improvisation)

class CollaborationNeed(Enum):
    """What type of help an agent might need - the 'instruments' in our orchestra"""
    DATA_ANALYSIS = "data_analysis"           # Need number crunching and metrics
    CONTENT_GENERATION = "content_generation" # Need writing and response creation
    VALIDATION = "validation"                 # Need quality checking and review
    CONTEXT_ENRICHMENT = "context_enrichment" # Need additional background information
    STRATEGIC_REASONING = "strategic_reasoning" # Need high-level planning and recommendations

@dataclass
class CollaborationRequest:
    """A request for help between agents - like passing a note between musicians"""
    task_id: str
    requesting_agent: str
    needed_capability: CollaborationNeed
    context: Dict[str, Any]
    priority: int = 5  # 1-10 scale, 10 being most urgent
    reasoning: str = ""  # Why this collaboration is needed

class CollaborativeFramework:
    """
    The intelligent conductor that orchestrates agent collaboration
    
    This framework sits between your LangGraph orchestrator and your agents,
    transforming simple agent calls into intelligent collaboration sessions.
    Your LangGraph workflows remain unchanged, but gain collaborative intelligence.
    """
    
    def __init__(self, redis_client: redis.Redis, agents_registry: Dict[str, BaseAgent]):
        self.redis_client = redis_client
        self.agents_registry = agents_registry
        self.logger = logging.getLogger("CollaborativeFramework")
        
        # Map collaboration needs to agents that can fulfill them
        # This is like knowing which musicians can play which instruments
        self.capability_map = self._build_capability_map()
        
        # Track collaboration history for learning and optimization
        self.collaboration_history_key = "collaboration_history"
        
        # Performance tracking for continuous improvement
        self.performance_metrics = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "avg_collaboration_time": 0.0,
            "collaboration_patterns": {}
        }
        
        self.logger.info("🎭 Collaborative Framework initialized - ready to orchestrate intelligent agent coordination")

    def _build_capability_map(self) -> Dict[CollaborationNeed, List[str]]:
        """
        Build a map of which agents can help with which types of collaboration
        
        This is like creating a directory of which musicians can play which instruments.
        As your system grows, this map helps the framework know who to call for what.
        """
        capability_map = {
            # Data analysis needs - agents that are good with numbers and patterns
            CollaborationNeed.DATA_ANALYSIS: [
                "jira_data_agent",           # Raw data retrieval and filtering
                "productivity_dashboard_agent" # Analytics and insights
            ],
            
            # Content generation needs - agents that create responses and content
            CollaborationNeed.CONTENT_GENERATION: [
                "chat_agent",                    # Conversational responses
                "jira_article_generator_agent"   # Article and documentation creation
            ],
            
            # Validation needs - agents that can review and improve quality
            CollaborationNeed.VALIDATION: [
                "knowledge_base_agent"  # Quality review and refinement suggestions
            ],
            
            # Context enrichment needs - agents that add background and depth
            CollaborationNeed.CONTEXT_ENRICHMENT: [
                "retrieval_agent",      # Find relevant articles and documentation
                "recommendation_agent"  # Add strategic insights and suggestions
            ],
            
            # Strategic reasoning needs - agents that think at a high level
            CollaborationNeed.STRATEGIC_REASONING: [
                "recommendation_agent",          # Strategic advice and planning
                "productivity_dashboard_agent"   # Performance analysis and optimization
            ]
        }
        
        self.logger.info(f"🗺️ Built capability map with {len(capability_map)} collaboration types")
        return capability_map

    async def coordinate_agents(self, primary_agent_id: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        This is the main method that transforms simple agent calls into intelligent collaboration
        
        Your LangGraph nodes call this method instead of agent.run(), and it:
        1. Runs the primary agent
        2. Analyzes if collaboration would help
        3. Coordinates additional agents if beneficial
        4. Synthesizes all results into an enhanced outcome
        
        It's like having a conductor who can turn a solo performance into a symphony.
        """
        collaboration_start_time = time.time()
        self.logger.info(f"🎯 Starting intelligent coordination for primary agent: {primary_agent_id}")
        
        # Step 1: Run the primary agent and capture its results
        primary_agent = self.agents_registry.get(primary_agent_id)
        if not primary_agent:
            self.logger.error(f"❌ Primary agent {primary_agent_id} not found in registry")
            return {"error": f"Agent {primary_agent_id} not available", "workflow_status": "error"}
        
        self.logger.info(f"🏃 Running primary agent: {primary_agent_id}")
        primary_result = primary_agent.run(task_context)
        
        # Step 2: Analyze the primary agent's mental state for collaboration opportunities
        collaboration_needs = self._analyze_collaboration_needs(primary_agent, task_context, primary_result)
        
        # Step 3: If collaboration is beneficial, orchestrate it
        if collaboration_needs:
            self.logger.info(f"🤝 Collaboration opportunities detected: {len(collaboration_needs)} requests")
            enhanced_result = await self._orchestrate_collaboration(
                primary_agent_id, primary_result, collaboration_needs, task_context
            )
        else:
            self.logger.info(f"✅ Primary agent {primary_agent_id} achieved good results independently")
            enhanced_result = primary_result
            enhanced_result["collaboration_metadata"] = {
                "collaboration_attempted": False,
                "reason": "no_collaboration_needed"
            }
        
        # Step 4: Track performance for continuous improvement
        collaboration_time = time.time() - collaboration_start_time
        self._track_collaboration_performance(primary_agent_id, collaboration_needs, enhanced_result, collaboration_time)
        
        return enhanced_result

    def _analyze_collaboration_needs(self, agent: BaseAgent, context: Dict[str, Any], 
                                   result: Dict[str, Any]) -> List[CollaborationRequest]:
        """
        Analyze if an agent's work could benefit from collaboration
        
        This is like a conductor listening to a solo performance and recognizing
        when additional instruments would make it richer and more complete.
        
        The analysis looks at:
        - Agent's confidence in its results
        - Complexity of the request
        - Quality indicators in the output
        - Explicit collaboration requests from the agent's mental state
        """
        needs = []
        agent_name = agent.name
        
        self.logger.info(f"🔍 Analyzing collaboration needs for {agent_name}")
        
        # Check 1: Does the agent have low confidence in its results?
        if hasattr(agent.mental_state, 'beliefs'):
            # Look for confidence-related beliefs that suggest uncertainty
            low_confidence_beliefs = []
            for key, belief in agent.mental_state.beliefs.items():
                if hasattr(belief, 'get_current_confidence'):
                    confidence = belief.get_current_confidence()
                    if confidence < 0.6 and 'result' in key.lower():
                        low_confidence_beliefs.append((key, confidence))
            
            if low_confidence_beliefs:
                self.logger.info(f"📉 Low confidence detected in {len(low_confidence_beliefs)} beliefs")
                needs.append(CollaborationRequest(
                    task_id=f"validation_{datetime.now().strftime('%H%M%S')}",
                    requesting_agent=agent_name,
                    needed_capability=CollaborationNeed.VALIDATION,
                    context=context,
                    reasoning=f"Agent has low confidence in {len(low_confidence_beliefs)} key results"
                ))

        # Check 2: Has the agent explicitly requested collaboration?
        if hasattr(agent.mental_state, 'collaborative_requests'):
            recent_requests = agent.mental_state.collaborative_requests[-3:]  # Last 3 requests
            for req in recent_requests:
                collaboration_type = self._map_reasoning_to_collaboration(req.get('reasoning_type', ''))
                if collaboration_type:
                    self.logger.info(f"🤝 Explicit collaboration request: {req.get('agent_type')} for {collaboration_type.value}")
                    needs.append(CollaborationRequest(
                        task_id=f"explicit_{datetime.now().strftime('%H%M%S')}",
                        requesting_agent=agent_name,
                        needed_capability=collaboration_type,
                        context=req.get('context', {}),
                        reasoning=f"Explicit request for {req.get('agent_type')} collaboration"
                    ))

        # Check 3: Does the result quality suggest collaboration could help?
        result_quality_score = self._assess_result_quality(result, context)
        if result_quality_score < 0.7:
            self.logger.info(f"📊 Result quality score {result_quality_score:.2f} suggests collaboration could help")
            
            # Determine what type of collaboration might improve quality
            if result.get('workflow_status') == 'partial_success':
                needs.append(CollaborationRequest(
                    task_id=f"enhancement_{datetime.now().strftime('%H%M%S')}",
                    requesting_agent=agent_name,
                    needed_capability=CollaborationNeed.CONTEXT_ENRICHMENT,
                    context=context,
                    reasoning=f"Result quality {result_quality_score:.2f} could be enhanced with additional context"
                ))

        # Check 4: Is this a complex request that could benefit from multiple perspectives?
        complexity_score = self._assess_request_complexity(context)
        if complexity_score > 0.7 and len(needs) == 0:  # High complexity but no other collaboration identified
            self.logger.info(f"🧩 High complexity request (score: {complexity_score:.2f}) suggests strategic collaboration")
            needs.append(CollaborationRequest(
                task_id=f"complexity_{datetime.now().strftime('%H%M%S')}",
                requesting_agent=agent_name,
                needed_capability=CollaborationNeed.STRATEGIC_REASONING,
                context=context,
                reasoning=f"High complexity request (score: {complexity_score:.2f}) benefits from strategic analysis"
            ))

        self.logger.info(f"🎯 Identified {len(needs)} collaboration opportunities for {agent_name}")
        return needs

    def _map_reasoning_to_collaboration(self, reasoning_type: str) -> Optional[CollaborationNeed]:
        """Map agent reasoning types to collaboration needs"""
        mapping = {
            "data_analysis": CollaborationNeed.DATA_ANALYSIS,
            "strategic_reasoning": CollaborationNeed.STRATEGIC_REASONING,
            "validation": CollaborationNeed.VALIDATION,
            "context_enrichment": CollaborationNeed.CONTEXT_ENRICHMENT,
            "content_generation": CollaborationNeed.CONTENT_GENERATION
        }
        return mapping.get(reasoning_type.lower())

    def _assess_result_quality(self, result: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Assess the quality of an agent's result to determine if collaboration could improve it
        
        This is like a music critic evaluating a performance and suggesting where
        additional instruments might enhance the overall experience.
        """
        quality_score = 0.0
        
        # Factor 1: Workflow status (40% of score)
        if result.get("workflow_status") == "success":
            quality_score += 0.4
        elif result.get("workflow_status") == "completed":
            quality_score += 0.4
        elif result.get("workflow_status") == "partial_success":
            quality_score += 0.2
        
        # Factor 2: Content richness (30% of score)
        response = result.get("response", "")
        if response and len(response) > 100:  # Substantial response
            quality_score += 0.3
        elif response and len(response) > 50:  # Moderate response
            quality_score += 0.2
        elif response:  # Some response
            quality_score += 0.1
        
        # Factor 3: Supporting data availability (20% of score)
        if result.get("articles_used") or result.get("recommendations") or result.get("tickets"):
            quality_score += 0.2
        
        # Factor 4: Error indicators (10% of score, negative impact)
        if result.get("error") or "error" in result.get("workflow_status", "").lower():
            quality_score -= 0.1
        else:
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))

    def _assess_request_complexity(self, context: Dict[str, Any]) -> float:
        """
        Assess how complex a request is to determine if multi-agent collaboration would help
        
        Complex requests are like complex musical pieces - they benefit from multiple
        instruments playing together rather than a single solo performance.
        """
        complexity_score = 0.0
        
        # Check query complexity
        user_prompt = context.get("user_prompt", "")
        if user_prompt:
            # Length indicates complexity
            word_count = len(user_prompt.split())
            complexity_score += min(word_count / 20.0, 0.3)  # Up to 0.3 for length
            
            # Keywords that suggest complexity
            complex_keywords = [
                "analyze", "compare", "evaluate", "recommend", "optimize", 
                "forecast", "predict", "trend", "pattern", "correlation",
                "bottleneck", "efficiency", "performance", "metrics",
                "strategy", "planning", "roadmap"
            ]
            
            prompt_lower = user_prompt.lower()
            keyword_matches = sum(1 for keyword in complex_keywords if keyword in prompt_lower)
            complexity_score += min(keyword_matches / len(complex_keywords), 0.4)  # Up to 0.4 for keywords
        
        # Check context richness (more context = more complex analysis possible)
        context_factors = [
            context.get("articles", []),
            context.get("tickets", []),
            context.get("conversation_history", []),
            context.get("recommendations", [])
        ]
        
        non_empty_contexts = sum(1 for factor in context_factors if factor)
        complexity_score += min(non_empty_contexts / len(context_factors), 0.3)  # Up to 0.3 for context
        
        return min(complexity_score, 1.0)

    async def _orchestrate_collaboration(self, primary_agent_id: str, primary_result: Dict[str, Any],
                                       needs: List[CollaborationRequest], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the actual collaboration between agents
        
        This is where the magic happens - like a conductor bringing different sections
        of the orchestra together to create a richer, more complete performance.
        """
        self.logger.info(f"🎼 Orchestrating collaboration for {len(needs)} identified needs")
        
        enhanced_result = primary_result.copy()
        collaboration_metadata = {
            "primary_agent": primary_agent_id,
            "collaborating_agents": [],
            "collaboration_types": [],
            "collaboration_quality": 0.0,
            "start_time": datetime.now().isoformat()
        }

        successful_collaborations = 0
        
        for i, need in enumerate(needs):
            self.logger.info(f"🎯 Processing collaboration need {i+1}/{len(needs)}: {need.needed_capability.value}")
            
            # Find the best agent for this collaboration need
            suitable_agents = self.capability_map.get(need.needed_capability, [])
            available_agents = [
                agent_id for agent_id in suitable_agents 
                if agent_id != primary_agent_id and agent_id in self.agents_registry
            ]
            
            if not available_agents:
                self.logger.warning(f"⚠️ No available agents for {need.needed_capability.value}")
                continue

            # Select the best available agent (could be enhanced with performance history)
            collaborating_agent_id = available_agents[0]  # For now, pick the first available
            collaborating_agent = self.agents_registry[collaborating_agent_id]
            
            self.logger.info(f"🤝 Collaborating with {collaborating_agent_id} for {need.needed_capability.value}")
            
            # Prepare enhanced context for the collaborating agent
            enhanced_context = self._prepare_collaboration_context(
                context, primary_result, need, collaboration_metadata
            )
            
            try:
                # Run the collaborating agent
                collaboration_result = collaborating_agent.run(enhanced_context)
                
                # Merge results intelligently based on collaboration type
                enhanced_result = self._merge_results(enhanced_result, collaboration_result, need)
                
                # Track successful collaboration
                successful_collaborations += 1
                collaboration_metadata["collaborating_agents"].append(collaborating_agent_id)
                collaboration_metadata["collaboration_types"].append(need.needed_capability.value)
                
                self.logger.info(f"✅ Successful collaboration with {collaborating_agent_id}")
                
            except Exception as e:
                self.logger.error(f"❌ Collaboration with {collaborating_agent_id} failed: {str(e)}")
                continue

        # Calculate collaboration quality score
        collaboration_metadata["collaboration_quality"] = successful_collaborations / len(needs) if needs else 0.0
        collaboration_metadata["end_time"] = datetime.now().isoformat()
        collaboration_metadata["successful_collaborations"] = successful_collaborations
        collaboration_metadata["total_collaboration_attempts"] = len(needs)

        enhanced_result["collaboration_metadata"] = collaboration_metadata
        
        # Store collaboration outcome for learning
        self._store_collaboration_outcome(primary_agent_id, collaboration_metadata, enhanced_result)
        
        self.logger.info(f"🎉 Collaboration complete: {successful_collaborations}/{len(needs)} successful")
        return enhanced_result

    def _prepare_collaboration_context(self, original_context: Dict[str, Any], 
                                     primary_result: Dict[str, Any],
                                     need: CollaborationRequest,
                                     collaboration_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the right context for each collaborating agent
        
        This is like giving each musician the right sheet music and background
        about what the other instruments are playing.
        """
        enhanced_context = original_context.copy()
        
        # Add results from primary agent so collaborating agent understands the context
        enhanced_context["primary_agent_result"] = primary_result
        enhanced_context["collaboration_purpose"] = need.needed_capability.value
        enhanced_context["collaboration_reasoning"] = need.reasoning
        
        # Add metadata about the collaboration session
        enhanced_context["collaboration_session"] = {
            "primary_agent": collaboration_metadata["primary_agent"],
            "collaboration_id": f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "collaboration_type": need.needed_capability.value
        }
        
        return enhanced_context

    def _merge_results(self, primary_result: Dict[str, Any], 
                      collaboration_result: Dict[str, Any],
                      need: CollaborationRequest) -> Dict[str, Any]:
        """
        Intelligently merge results from different agents based on collaboration type
        
        This is like mixing different instrumental tracks to create a harmonious
        final composition where each part enhances the whole.
        """
        merged = primary_result.copy()
        
        if need.needed_capability == CollaborationNeed.DATA_ANALYSIS:
            # Enhance with additional data and metrics
            if "tickets" in collaboration_result:
                merged["tickets"] = collaboration_result["tickets"]
            if "metrics" in collaboration_result:
                merged["additional_metrics"] = collaboration_result["metrics"]
            if "analysis_confidence" in collaboration_result:
                merged["analysis_confidence"] = collaboration_result["analysis_confidence"]
                
        elif need.needed_capability == CollaborationNeed.CONTEXT_ENRICHMENT:
            # Add recommendations, articles, or additional context
            if "recommendations" in collaboration_result:
                existing_recs = merged.get("recommendations", [])
                new_recs = collaboration_result["recommendations"]
                # Merge recommendations, avoiding duplicates
                all_recs = existing_recs + [rec for rec in new_recs if rec not in existing_recs]
                merged["recommendations"] = all_recs
            if "articles" in collaboration_result:
                merged["related_articles"] = collaboration_result["articles"]
                
        elif need.needed_capability == CollaborationNeed.VALIDATION:
            # Add validation results and quality improvements
            merged["validation_result"] = collaboration_result
            if collaboration_result.get("refinement_suggestion"):
                merged["suggested_improvements"] = collaboration_result["refinement_suggestion"]
            if collaboration_result.get("quality_score"):
                merged["quality_assessment"] = collaboration_result["quality_score"]
                
        elif need.needed_capability == CollaborationNeed.STRATEGIC_REASONING:
            # Enhance with strategic insights and high-level recommendations
            if "recommendations" in collaboration_result:
                existing_recs = merged.get("recommendations", [])
                strategic_recs = collaboration_result["recommendations"]
                merged["strategic_recommendations"] = strategic_recs
                merged["recommendations"] = existing_recs + strategic_recs
            if "strategic_insights" in collaboration_result:
                merged["strategic_insights"] = collaboration_result["strategic_insights"]
                
        elif need.needed_capability == CollaborationNeed.CONTENT_GENERATION:
            # Enhance content quality and richness
            if collaboration_result.get("response"):
                # If the collaboration produced a better response, use it
                if len(collaboration_result["response"]) > len(merged.get("response", "")):
                    merged["enhanced_response"] = collaboration_result["response"]

        return merged

    def _store_collaboration_outcome(self, primary_agent: str, metadata: Dict[str, Any], 
                                   result: Dict[str, Any]):
        """Store collaboration outcomes for future learning and optimization"""
        outcome = {
            "timestamp": datetime.now().isoformat(),
            "primary_agent": primary_agent,
            "collaborating_agents": metadata.get("collaborating_agents", []),
            "collaboration_types": metadata.get("collaboration_types", []),
            "collaboration_quality": metadata.get("collaboration_quality", 0.0),
            "success_indicators": {
                "workflow_status": result.get("workflow_status"),
                "has_recommendations": bool(result.get("recommendations")),
                "has_validation": bool(result.get("validation_result")),
                "response_quality": len(result.get("response", "")) > 100
            }
        }
        
        try:
            self.redis_client.lpush(self.collaboration_history_key, json.dumps(outcome, default=str))
            self.redis_client.ltrim(self.collaboration_history_key, 0, 99)  # Keep last 100
            self.logger.info(f"📝 Stored collaboration outcome for future learning")
        except Exception as e:
            self.logger.error(f"❌ Failed to store collaboration outcome: {e}")

    def _track_collaboration_performance(self, primary_agent_id: str, needs: List[CollaborationRequest],
                                       result: Dict[str, Any], collaboration_time: float):
        """Track performance metrics for continuous improvement"""
        self.performance_metrics["total_collaborations"] += 1
        
        if result.get("collaboration_metadata", {}).get("collaboration_quality", 0) > 0.5:
            self.performance_metrics["successful_collaborations"] += 1
        
        # Update average collaboration time
        current_avg = self.performance_metrics["avg_collaboration_time"]
        total_collabs = self.performance_metrics["total_collaborations"]
        new_avg = ((current_avg * (total_collabs - 1)) + collaboration_time) / total_collabs
        self.performance_metrics["avg_collaboration_time"] = new_avg
        
        # Track collaboration patterns
        pattern_key = f"{primary_agent_id}_{len(needs)}_needs"
        if pattern_key not in self.performance_metrics["collaboration_patterns"]:
            self.performance_metrics["collaboration_patterns"][pattern_key] = {"count": 0, "success_rate": 0.0}
        
        pattern_data = self.performance_metrics["collaboration_patterns"][pattern_key]
        pattern_data["count"] += 1
        
        # Update success rate for this pattern
        if result.get("collaboration_metadata", {}).get("collaboration_quality", 0) > 0.5:
            old_success_rate = pattern_data["success_rate"]
            pattern_data["success_rate"] = ((old_success_rate * (pattern_data["count"] - 1)) + 1.0) / pattern_data["count"]

    def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get insights about collaboration patterns and effectiveness"""
        try:
            # Get recent collaboration history
            recent_history = self.redis_client.lrange(self.collaboration_history_key, 0, 19)  # Last 20
            
            if not recent_history:
                return {"status": "no_collaborations", "insights": "No collaboration history available"}
            
            collaborations = [json.loads(collab) for collab in recent_history]
            
            # Analyze patterns
            agent_collaboration_frequency = {}
            collaboration_type_effectiveness = {}
            
            for collab in collaborations:
                primary = collab.get("primary_agent", "unknown")
                agent_collaboration_frequency[primary] = agent_collaboration_frequency.get(primary, 0) + 1
                
                for collab_type in collab.get("collaboration_types", []):
                    if collab_type not in collaboration_type_effectiveness:
                        collaboration_type_effectiveness[collab_type] = {"total": 0, "successful": 0}
                    
                    collaboration_type_effectiveness[collab_type]["total"] += 1
                    if collab.get("collaboration_quality", 0) > 0.5:
                        collaboration_type_effectiveness[collab_type]["successful"] += 1
            
            # Calculate effectiveness rates
            for collab_type, stats in collaboration_type_effectiveness.items():
                if stats["total"] > 0:
                    stats["effectiveness_rate"] = stats["successful"] / stats["total"]
            
            return {
                "total_recent_collaborations": len(collaborations),
                "agent_collaboration_frequency": agent_collaboration_frequency,
                "collaboration_effectiveness": collaboration_type_effectiveness,
                "average_collaboration_quality": sum(c.get("collaboration_quality", 0) for c in collaborations) / len(collaborations),
                "performance_metrics": self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collaboration insights: {e}")
            return {"error": str(e)}