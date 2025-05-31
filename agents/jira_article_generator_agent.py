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
        super().__init__(name="jira_article_generator_agent")
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        self.jira_data_agent = JiraDataAgent(mock_data_path="data/mock_jira_data.json") 
        self.knowledge_base_agent = KnowledgeBaseAgent(shared_memory)
        
        self.mental_state.capabilities = [
            AgentCapability.GENERATE_ARTICLE
        ]
        
        self.mental_state.obligations.extend([
            "detect_query_type",
            "generate_article"
        ])

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
        
        self.log(f"[DEBUG] Perceiving article generation request for Ticket: {ticket_id} | Refinement: {refinement_suggestion}")
        self.mental_state.beliefs.update({
            "ticket_id": ticket_id,
            "refinement_suggestion": refinement_suggestion,
            "query_type": QueryType.CONVERSATION,
            "autonomous_refinement_done": False  
        })

    def _generate_article(self) -> Dict[str, Any]:
        ticket_id = self.mental_state.beliefs["ticket_id"]
        refinement_suggestion = self.mental_state.beliefs.get("refinement_suggestion")
        
        
        input_data = {
            "project_id": "PROJ123",  
            "time_range": {"start": "2025-05-01T00:00:00Z", "end": "2025-05-15T23:59:59Z"}
        }
        ticket_data_result = self.jira_data_agent.run(input_data)
        tickets = ticket_data_result.get("tickets", [])
        
        ticket_context = "No ticket-specific data available."
        for ticket in tickets:
            if ticket.get("key") == ticket_id:
                fields = ticket.get("fields", {})
                changelog = ticket.get("changelog", {}).get("histories", [])
                resolution = fields.get("resolutiondate") and "Resolved" or "In Progress"
                ticket_context = (
                    f"Ticket ID: {ticket_id}\n"
                    f"Project: {fields.get('project', {}).get('key', 'Unknown Project')}\n"
                    f"Summary: {fields.get('summary', 'No summary')}\n"
                    f"Status: {fields.get('status', {}).get('name', 'Unknown')}\n"
                    f"Resolution: {resolution}\n"
                    f"Created: {fields.get('created', 'Unknown')}\n"
                    f"Updated: {fields.get('updated', 'Unknown')}\n"
                    f"Changelog: {json.dumps(changelog) if changelog else 'No changelog'}"
                )
                break
        
        # Enhanced prompt for higher quality initial article generation
        prompt_template = f"""<|system|>You are an AI specialized in creating high-quality, Confluence-ready markdown articles for resolved Jira tickets.
        Generate a professional article based on the ticket data provided. 
        
        Include the following sections with detailed content:
        1. Problem (clearly describe the issue with relevant technical details)
        2. Resolution (provide detailed technical solution with step-by-step explanation)
        3. Key Improvements (list specific measurable improvements with metrics)
        4. Implementation Details (include technical specifics that would be valuable for future reference)
        5. Project Impact (explain wider benefits to the project)
        6. Next Steps (actionable recommendations)
        
        Use professional markdown formatting with clear headings (use # for main headings, ## for subheadings), bullet points for improvements, code blocks where relevant, and a technical but accessible tone. 
        Be specific and detailed rather than generic. Include realistic data points, metrics, and technical specifications.
        Ensure the article is actionable, technically precise, and ready for Confluence.
        
        Ticket-specific data:
        {ticket_context}
        
        Return the article content as a complete markdown document.<|assistant|>"""
        
       
        if refinement_suggestion:
            prompt_template += f"\nRevise the previous article based on this specific suggestion: {refinement_suggestion}. Make targeted improvements while maintaining all the existing content and structure."
        
        self.log(f"[DEBUG] Article generation prompt: {prompt_template[:500]}...")
        
        try:
            
            content = self.model_manager.generate_response(prompt_template)
            self.log(f"[DEBUG] Raw LLM article response: {content}")
            
            if not content.strip():
                raise ValueError("Generated article is empty")
            
            fields = next((t.get("fields", {}) for t in tickets if t.get("key") == ticket_id), {})
            project = fields.get("project", {}).get("key", "Unknown Project")
            title = f"Know-How: {fields.get('summary', 'Unknown Issue')} in Project {project}"
            article = {
                "content": content,
                "status": "draft",
                "title": title,
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat()
            }
            
            self.log(f"[DEBUG] Generated article: {article}")
            return article
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate article with LLM: {e}")
            
            fields = next((t.get("fields", {}) for t in tickets if t.get("key") == ticket_id), {})
            project = fields.get("project", {}).get("key", "Unknown Project")
            return {
                "content": f"# Know-How: {fields.get('summary', 'Unknown Issue')} in Project {project}\n## Problem\nNo detailed problem data available.\n## Resolution\nNo detailed resolution available.\n## Key Improvements\n- No improvements recorded.\n## Project Impact\nNo impact data available.\n## Next Steps\nReview ticket details for further action.",
                "status": "draft",
                "title": f"Know-How: {fields.get('summary', 'Unknown Issue')} in Project {project}",
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat()
            }

    def _act(self) -> Dict[str, Any]:
        try:
            article = self._generate_article()
            
        
            self.log("[DEBUG] Autonomously evaluating the generated article")
            evaluation_input = {"article": article}
            evaluation_result = self.knowledge_base_agent.run(evaluation_input)
            
            self.log(f"[DEBUG] Evaluation result: {evaluation_result}")
            redundant = evaluation_result.get("redundant", False)
            refinement_suggestion = evaluation_result.get("refinement_suggestion")
            
           
            if refinement_suggestion and not self.mental_state.beliefs.get("autonomous_refinement_done", False):
                self.log(f"[DEBUG] Refinement suggested: {refinement_suggestion}. Triggering autonomous refinement.")
                
                # Update beliefs for refinement
                self.mental_state.beliefs["refinement_suggestion"] = refinement_suggestion
                self.mental_state.beliefs["autonomous_refinement_done"] = True  # Limit to one cycle
                
                
                refined_article = self._generate_article()
                article = refined_article
                self.log("[DEBUG] Article refined autonomously")
            
            else:
                self.log("[DEBUG] No refinement needed or refinement cycle limit reached")
            
            return {
                "article": article,
                "workflow_status": "success",
                "autonomous_refinement_done": self.mental_state.beliefs.get("autonomous_refinement_done", False)
            }
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate or refine article: {e}")
            return {
                "article": {},
                "workflow_status": "failure",
                "autonomous_refinement_done": self.mental_state.beliefs.get("autonomous_refinement_done", False)
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        self.mental_state.beliefs["last_article"] = {
            "timestamp": datetime.now().isoformat(),
            "article_generated": bool(action_result.get("article")),
            "status": action_result.get("workflow_status"),
            "autonomous_refinement_done": action_result.get("autonomous_refinement_done", False)
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)