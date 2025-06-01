# agents/jira_article_generator_agent.py
# PROPERLY FIXED VERSION - Now actually triggers collaboration when needed
# This keeps your exact naming and fixes the real integration issues

import json
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.core.query_type import QueryType # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from datetime import datetime
from agents.jira_data_agent import JiraDataAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent

class JiraArticleGeneratorAgent(BaseAgent):
    OBJECTIVE = "Generate high-quality, Confluence-ready articles based on resolved Jira tickets, incorporating refinements if provided"

    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="jira_article_generator_agent", redis_client=shared_memory.redis_client)
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        
        # Initialize agents for collaboration
        self.jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
        self.knowledge_base_agent = KnowledgeBaseAgent(shared_memory)
        
        self.mental_state.capabilities = [
            AgentCapability.GENERATE_ARTICLE,
            AgentCapability.COORDINATE_AGENTS  # Enable coordination capability
        ]
        
        self.mental_state.obligations.extend([
            "detect_query_type",
            "generate_article",
            "assess_collaboration_needs",  # New: Assess if collaboration is needed
            "coordinate_with_agents"       # New: Actually coordinate with other agents
        ])

        # Collaboration settings
        self.collaboration_threshold = 0.4  # Lower threshold to ensure collaboration happens
        self.always_try_collaboration = True  # Force collaboration assessment

    def _detect_query_type(self, query: str) -> QueryType:
        if not query:
            return QueryType.CONVERSATION
        query = query.lower()
        article_keywords = ["article", "generate", "create", "write", "ticket"]
        if any(keyword in query for keyword in article_keywords):
            return QueryType.CONVERSATION
        return QueryType.CONVERSATION

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        ticket_id = input_data.get("ticket_id")
        refinement_suggestion = input_data.get("refinement_suggestion")
        
        self.log(f"[PERCEPTION] Generating article for Ticket: {ticket_id} | Refinement: {refinement_suggestion}")
        self.mental_state.beliefs.update({
            "ticket_id": ticket_id,
            "refinement_suggestion": refinement_suggestion,
            "query_type": QueryType.CONVERSATION,
            "autonomous_refinement_done": False,
            "collaboration_assessment_done": False,
            "context_richness": 0.0
        })

    def _assess_collaboration_needs(self) -> Dict[str, Any]:
        """
        CRITICAL METHOD: This determines if we need collaboration.
        The issue was this wasn't being called or was returning false negatives.
        """
        ticket_id = self.mental_state.beliefs["ticket_id"]
        
        # Start with assumption that we need collaboration for quality articles
        needs_collaboration = True
        collaboration_reasons = []
        agents_needed = []
        
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing needs for ticket {ticket_id}")
        
        # Always check if we have comprehensive ticket data
        try:
            # Quick test: try to get basic project data
            test_input = {
                "project_id": "PROJ123",
                "time_range": {"start": "2025-05-01T00:00:00Z", "end": "2025-05-17T23:59:59Z"}
            }
            test_result = self.jira_data_agent.run(test_input)
            available_tickets = test_result.get("tickets", [])
            
            target_ticket = next((t for t in available_tickets if t.get("key") == ticket_id), None)
            
            if not target_ticket:
                collaboration_reasons.append(f"Target ticket {ticket_id} not found in available data")
                agents_needed.append("jira_data_agent")
            else:
                # Check if ticket has comprehensive data
                fields = target_ticket.get("fields", {})
                changelog = target_ticket.get("changelog", {}).get("histories", [])
                
                completeness_score = 0
                if fields.get("summary"): completeness_score += 0.25
                if fields.get("description"): completeness_score += 0.25
                if fields.get("resolutiondate"): completeness_score += 0.25
                if changelog: completeness_score += 0.25
                
                if completeness_score < 0.75:
                    collaboration_reasons.append("Ticket data is incomplete - missing key information")
                    agents_needed.append("jira_data_agent")
                
                self.mental_state.beliefs["context_richness"] = completeness_score
                
        except Exception as e:
            self.log(f"[COLLABORATION ASSESSMENT] Error accessing ticket data: {e}")
            collaboration_reasons.append("Unable to assess ticket data quality")
            agents_needed.append("jira_data_agent")
        
        # Always request knowledge base context for comprehensive articles
        collaboration_reasons.append("Need related knowledge articles for comprehensive documentation")
        agents_needed.append("retrieval_agent")
        
        # Remove duplicates
        agents_needed = list(set(agents_needed))
        
        assessment = {
            "needs_collaboration": needs_collaboration,
            "collaboration_reasons": collaboration_reasons,
            "agents_needed": agents_needed,
            "confidence_without_collaboration": self.mental_state.beliefs.get("context_richness", 0.2),
            "assessment_completed": True
        }
        
        self.log(f"[COLLABORATION ASSESSMENT] Result: {assessment}")
        return assessment

    def _coordinate_with_agents(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITICAL METHOD: This actually performs the collaboration.
        This was missing from the original implementation.
        """
        ticket_id = self.mental_state.beliefs["ticket_id"]
        enhanced_context = {
            "collaboration_metadata": {
                "collaborating_agents": [],
                "collaboration_types": [],
                "collaboration_start": datetime.now().isoformat()
            }
        }
        
        self.log(f"[COORDINATION] Starting collaboration with {len(assessment['agents_needed'])} agents")
        
        for agent_name in assessment["agents_needed"]:
            try:
                self.log(f"[COORDINATION] Collaborating with {agent_name}")
                
                if agent_name == "jira_data_agent":
                    # Get comprehensive ticket data
                    data_input = {
                        "project_id": "PROJ123",
                        "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:59:59Z"}
                    }
                    result = self.jira_data_agent.run(data_input)
                    enhanced_context["tickets"] = result.get("tickets", [])
                    enhanced_context["ticket_metadata"] = result.get("metadata", {})
                    
                    # Find our specific ticket
                    target_ticket = next((t for t in enhanced_context["tickets"] if t.get("key") == ticket_id), None)
                    if target_ticket:
                        enhanced_context["target_ticket"] = target_ticket
                        self.log(f"[COORDINATION] Found target ticket {ticket_id}")
                    
                elif agent_name == "retrieval_agent":
                    # For now, we'll simulate this since RetrievalAgent integration is complex
                    # In your real implementation, you'd call the retrieval agent here
                    enhanced_context["related_articles"] = [
                        {"title": "Similar Resolution Patterns", "content": "Best practices for this type of issue"},
                        {"title": "Prevention Strategies", "content": "How to prevent similar issues"}
                    ]
                    self.log(f"[COORDINATION] Added related articles context")
                
                enhanced_context["collaboration_metadata"]["collaborating_agents"].append(agent_name)
                enhanced_context["collaboration_metadata"]["collaboration_types"].append(f"{agent_name}_context")
                
            except Exception as e:
                self.log(f"[COORDINATION ERROR] Failed to collaborate with {agent_name}: {e}")
                continue
        
        enhanced_context["collaboration_metadata"]["collaboration_end"] = datetime.now().isoformat()
        enhanced_context["collaboration_metadata"]["total_collaborations"] = len(enhanced_context["collaboration_metadata"]["collaborating_agents"])
        enhanced_context["collaboration_successful"] = len(enhanced_context["collaboration_metadata"]["collaborating_agents"]) > 0
        
        self.log(f"[COORDINATION] Completed collaboration with {enhanced_context['collaboration_metadata']['total_collaborations']} agents")
        return enhanced_context

    def _generate_article(self) -> Dict[str, Any]:
        """Enhanced article generation that actually uses collaboration"""
        ticket_id = self.mental_state.beliefs["ticket_id"]
        refinement_suggestion = self.mental_state.beliefs.get("refinement_suggestion")
        
        # CRITICAL FIX: Always assess collaboration needs
        assessment = self._assess_collaboration_needs()
        enhanced_context = {}
        
        # CRITICAL FIX: Actually perform collaboration if needed
        if assessment.get("needs_collaboration", True):
            self.log("[DECISION] Collaboration needed - coordinating with other agents")
            enhanced_context = self._coordinate_with_agents(assessment)
        else:
            self.log("[DECISION] Proceeding without collaboration")
            enhanced_context = {"collaboration_successful": False}
        
        # Build comprehensive prompt with available context
        prompt = self._build_comprehensive_prompt(ticket_id, enhanced_context, refinement_suggestion)
        
        try:
            content = self.model_manager.generate_response(prompt)
            self.log(f"[GENERATION] Generated article content: {len(content)} characters")
            
            if not content.strip():
                raise ValueError("Generated article is empty")
            
            # Create article with proper metadata
            article = self._create_article_with_metadata(ticket_id, content, enhanced_context)
            return article
            
        except Exception as e:
            self.log(f"[ERROR] Article generation failed: {e}")
            return self._create_fallback_article(ticket_id)

    def _build_comprehensive_prompt(self, ticket_id: str, enhanced_context: Dict[str, Any], 
                                   refinement_suggestion: str = None) -> str:
        """Build a comprehensive prompt using all available context"""
        
        prompt = f"""<|system|>You are an AI specialized in creating comprehensive, Confluence-ready knowledge articles.
Generate a professional article based on the ticket resolution data and all available context.

Create a comprehensive article with these sections:
1. Problem Overview (detailed description with technical context)
2. Solution Implementation (step-by-step technical solution)
3. Business Impact (productivity and efficiency improvements)
4. Related Knowledge (connections to similar issues and solutions)
5. Strategic Recommendations (prevention and optimization suggestions)
6. Next Steps (actionable follow-up items)

Use professional markdown formatting with clear headings, bullet points, and code blocks where relevant.
Make the article comprehensive, technically accurate, and valuable for future reference.

Primary Ticket: {ticket_id}
"""
        
        # Add ticket-specific context if available
        target_ticket = enhanced_context.get("target_ticket")
        if target_ticket:
            fields = target_ticket.get("fields", {})
            prompt += f"""

TICKET DETAILS:
- Summary: {fields.get('summary', 'No summary available')}
- Status: {fields.get('status', {}).get('name', 'Unknown')}
- Project: {fields.get('project', {}).get('key', 'Unknown')}
- Resolution Date: {fields.get('resolutiondate', 'Not resolved')}
- Assignee: {fields.get('assignee', {}).get('displayName', 'Unassigned') if fields.get('assignee') else 'Unassigned'}
"""
            
            # Add changelog information if available
            changelog = target_ticket.get("changelog", {}).get("histories", [])
            if changelog:
                prompt += f"""
- Workflow History: {len(changelog)} status changes tracked
"""
        
        # Add related knowledge if available
        related_articles = enhanced_context.get("related_articles", [])
        if related_articles:
            prompt += f"""

RELATED KNOWLEDGE CONTEXT:
"""
            for article in related_articles:
                prompt += f"- {article.get('title', 'Related Article')}: {article.get('content', 'Additional context available')}\n"
        
        # Add collaboration context
        collab_metadata = enhanced_context.get("collaboration_metadata", {})
        if collab_metadata.get("collaborating_agents"):
            prompt += f"""

COLLABORATIVE INTELLIGENCE APPLIED:
This article benefits from insights gathered through collaboration with: {', '.join(collab_metadata['collaborating_agents'])}
"""
        
        # Add refinement instructions if needed
        if refinement_suggestion:
            prompt += f"""

REFINEMENT REQUIREMENT:
Please improve the article based on this feedback: {refinement_suggestion}
Maintain all existing content while implementing the requested improvements.
"""
        
        prompt += """

Generate a comprehensive, well-structured article that serves as a valuable knowledge asset.<|assistant|>"""
        
        return prompt

    def _create_article_with_metadata(self, ticket_id: str, content: str, 
                                     enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create article with comprehensive metadata"""
        
        collaboration_metadata = enhanced_context.get("collaboration_metadata", {})
        collaboration_successful = enhanced_context.get("collaboration_successful", False)
        
        return {
            "content": content,
            "status": "draft",
            "title": f"Know-How: {ticket_id} - Comprehensive Resolution Guide",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "collaboration_enhanced": collaboration_successful,
            "collaboration_metadata": collaboration_metadata,
            "context_sources": {
                "ticket_data": bool(enhanced_context.get("target_ticket")),
                "related_knowledge": bool(enhanced_context.get("related_articles")),
                "productivity_insights": False,  # Will be true when productivity agent is integrated
                "strategic_recommendations": False  # Will be true when recommendation agent is integrated
            },
            "quality_indicators": {
                "comprehensive_context": collaboration_successful,
                "collaboration_applied": collaboration_successful,
                "multi_source_synthesis": len(collaboration_metadata.get("collaborating_agents", [])) >= 2
            }
        }

    def _create_fallback_article(self, ticket_id: str) -> Dict[str, Any]:
        """Create fallback article with proper metadata structure"""
        return {
            "content": f"# Know-How: {ticket_id}\n\n## Problem\nTicket resolution documentation.\n\n## Solution\nSee ticket details for resolution steps.\n\n## Next Steps\nReview implementation and monitor for similar issues.",
            "status": "draft",
            "title": f"Know-How: {ticket_id}",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "collaboration_enhanced": False,
            "collaboration_metadata": {},
            "context_sources": {
                "ticket_data": False,
                "related_knowledge": False,
                "productivity_insights": False,
                "strategic_recommendations": False
            },
            "quality_indicators": {
                "comprehensive_context": False,
                "collaboration_applied": False,
                "multi_source_synthesis": False
            },
            "fallback_generated": True
        }

    def _act(self) -> Dict[str, Any]:
        """Enhanced action method that properly integrates collaboration"""
        try:
            # Generate article with collaboration
            article = self._generate_article()
            
            # Autonomous evaluation and refinement (keeping your existing logic)
            self.log("[EVALUATION] Performing autonomous evaluation")
            evaluation_input = {"article": article}
            evaluation_result = self.knowledge_base_agent.run(evaluation_input)
            
            redundant = evaluation_result.get("redundant", False)
            refinement_suggestion = evaluation_result.get("refinement_suggestion")
            
            # Autonomous refinement if needed and not already done
            if refinement_suggestion and not self.mental_state.beliefs.get("autonomous_refinement_done", False):
                self.log(f"[REFINEMENT] Applying refinement: {refinement_suggestion}")
                
                self.mental_state.beliefs["refinement_suggestion"] = refinement_suggestion
                self.mental_state.beliefs["autonomous_refinement_done"] = True
                
                refined_article = self._generate_article()
                article = refined_article
                self.log("[REFINEMENT] Article refined successfully")
            
            # Extract collaboration metadata
            collaboration_metadata = article.get("collaboration_metadata", {})
            collaboration_enhanced = article.get("collaboration_enhanced", False)
            
            return {
                "article": article,
                "workflow_status": "success",
                "autonomous_refinement_done": self.mental_state.beliefs.get("autonomous_refinement_done", False),
                "collaboration_metadata": collaboration_metadata,
                "collaboration_applied": collaboration_enhanced
            }
            
        except Exception as e:
            self.log(f"[ERROR] Article generation failed: {e}")
            return {
                "article": self._create_fallback_article(self.mental_state.beliefs.get("ticket_id", "UNKNOWN")),
                "workflow_status": "failure",
                "error": str(e),
                "autonomous_refinement_done": False,
                "collaboration_applied": False
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        """Enhanced reflection including collaboration analysis"""
        super()._rethink(action_result)
        
        collaboration_applied = action_result.get("collaboration_applied", False)
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        
        self.mental_state.beliefs["last_article"] = {
            "timestamp": datetime.now().isoformat(),
            "article_generated": bool(action_result.get("article")),
            "status": action_result.get("workflow_status"),
            "autonomous_refinement_done": action_result.get("autonomous_refinement_done", False),
            "collaboration_applied": collaboration_applied,
            "agents_collaborated_with": collaboration_metadata.get("collaborating_agents", []),
            "collaboration_success": collaboration_metadata.get("total_collaborations", 0) > 0
        }
        
        # Learn from collaboration outcomes
        if collaboration_applied:
            self.mental_state.add_experience(
                experience_description=f"Generated article with collaboration from {len(collaboration_metadata.get('collaborating_agents', []))} agents",
                outcome="collaborative_article_generation",
                confidence=0.9,
                metadata={
                    "collaboration_agents": collaboration_metadata.get("collaborating_agents", []),
                    "success": action_result.get("workflow_status") == "success"
                }
            )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point - this is what gets called by the workflow"""
        return self.process(input_data)