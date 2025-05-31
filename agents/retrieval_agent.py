from typing import Dict, Any, Optional, List
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.core.model_manager import ModelManager # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from datetime import datetime
from orchestrator.core.query_type import QueryType # type: ignore
import chromadb  # type: ignore

class RetrievalAgent(BaseAgent):
    OBJECTIVE = "Retrieve relevant Confluence articles for Agile and software development queries using semantic search"
    
    def __init__(self, shared_memory: JurixSharedMemory):
        super().__init__(name="retrieval_agent")
        self.shared_memory = shared_memory
        self.model_manager = ModelManager()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="c:/Users/tlouh/Desktop/JURIX/chromadb_data")

        self.collection = self.client.get_or_create_collection(name="confluence_articles")
        
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT
        ]
        
        self.mental_state.obligations.extend([
            "detect_query_type",
            "retrieve_articles",
            "rank_relevance"
        ])

    def _detect_query_type(self, query: str) -> QueryType:
        if not query:
            return QueryType.CONVERSATION
        query = query.lower()
        search_keywords = ["find", "search", "show me", "look for", "articles about", "documentation", "give me"]
        if any(keyword in query for keyword in search_keywords):
            return QueryType.SEARCH
        return QueryType.CONVERSATION

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        session_id = input_data.get("session_id")
        user_prompt = input_data.get("user_prompt")
        history = self.shared_memory.get_conversation(session_id) if session_id else []
        
        self.log(f"[DEBUG] Perceiving retrieval request for: {user_prompt}")
        self.mental_state.beliefs.update({
            "session_id": session_id,
            "user_prompt": user_prompt,
            "conversation_history": history,
            "query_type": self._detect_query_type(user_prompt) if user_prompt else None
        })

    def _retrieve_articles(self) -> List[Dict[str, Any]]:
        user_prompt = self.mental_state.beliefs["user_prompt"]
        query_embedding = self.model.encode(user_prompt).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3  
        )

        articles = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            articles.append({"content": doc, "metadata": meta})
        return articles

    def _act(self) -> Dict[str, Any]:
        try:
            articles = self._retrieve_articles()
            return {"articles": articles, "workflow_status": "success"}
        except Exception as e:
            self.log(f"[ERROR] Failed to retrieve articles: {e}")
            return {"articles": [], "workflow_status": "failure"}

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        self.mental_state.beliefs["last_retrieval"] = {
            "timestamp": datetime.now().isoformat(),
            "articles_retrieved": len(action_result.get("articles", [])),
            "status": action_result.get("workflow_status")
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)