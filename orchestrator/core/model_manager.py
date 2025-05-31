import ollama
import redis
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from orchestrator.core.cognitive_model_manager import CognitiveModelManager, ReasoningType # type: ignore

class ModelManager:
    """Enhanced Model Manager with cognitive specialization"""
    
    def __init__(self, model_name: str = "mistral", redis_client: Optional[redis.Redis] = None):
        self.default_model = model_name
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.logger = logging.getLogger("ModelManager")
        
        # Initialize cognitive model manager
        self.cognitive_manager = CognitiveModelManager(self.redis_client)
        
        # Track agent context for better model selection
        self.current_agent_context = {}
        
        self.logger.info(f"Enhanced ModelManager initialized with default model: {model_name}")

    def set_agent_context(self, agent_name: str, agent_capabilities: list = None, 
                         task_type: str = None):
        """Set context about the current agent for better model selection"""
        self.current_agent_context = {
            "agent_name": agent_name,
            "agent_capabilities": agent_capabilities or [],
            "task_type": task_type,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.debug(f"Set agent context: {agent_name}")

    def generate_response(self, prompt: str, reasoning_type: Optional[ReasoningType] = None,
                         force_model: Optional[str] = None, use_cognitive_routing: bool = True) -> str:
        """
        Generate response with cognitive model routing
        
        Args:
            prompt: The input prompt
            reasoning_type: Specific reasoning type (auto-detected if None)
            force_model: Force specific model (bypasses cognitive routing)
            use_cognitive_routing: Whether to use cognitive specialization
        """
        self.logger.info(f"Generating response with cognitive routing: {use_cognitive_routing}")
        
        if use_cognitive_routing and not force_model:
            # Use cognitive model manager for specialized routing
            return self.cognitive_manager.generate_response(
                prompt=prompt,
                reasoning_type=reasoning_type,
                context=self.current_agent_context,
                force_model=force_model
            )
        else:
            # Fall back to original simple generation
            return self._simple_generate(prompt, force_model or self.default_model)

    def _simple_generate(self, prompt: str, model: str) -> str:
        """Simple generation without cognitive routing (backward compatibility)"""
        self.logger.info(f"Generating response with model: {model}")
        try:
            response = ollama.generate(model=model, prompt=prompt)
            return response["response"]
        except Exception as e:
            self.logger.error(f"Simple generation failed with {model}: {str(e)}")
            raise Exception(f"Model generation failed: {str(e)}")

    def generate_for_agent(self, agent_name: str, prompt: str, 
                          agent_capabilities: list = None, task_type: str = None) -> str:
        """Generate response optimized for specific agent type"""
        
        # Set agent context
        self.set_agent_context(agent_name, agent_capabilities, task_type)
        
        # Auto-detect reasoning type based on agent
        reasoning_type = self._get_reasoning_type_for_agent(agent_name, task_type)
        
        self.logger.info(f"Generating response for {agent_name} using {reasoning_type.value} reasoning")
        
        return self.cognitive_manager.generate_response(
            prompt=prompt,
            reasoning_type=reasoning_type,
            context=self.current_agent_context
        )

    def _get_reasoning_type_for_agent(self, agent_name: str, task_type: str = None) -> ReasoningType:
        """Map agent types to reasoning types"""
        agent_lower = agent_name.lower()
        task_lower = (task_type or "").lower()
        
        # Agent-specific mapping
        if "chat" in agent_lower:
            return ReasoningType.CONVERSATIONAL
        elif "jira_data" in agent_lower or "data" in agent_lower:
            return ReasoningType.DATA_ANALYSIS
        elif "recommendation" in agent_lower:
            return ReasoningType.STRATEGIC_REASONING
        elif "article" in agent_lower or "generator" in agent_lower:
            return ReasoningType.CREATIVE_WRITING
        elif "knowledge" in agent_lower or "evaluate" in agent_lower:
            return ReasoningType.LOGICAL_REASONING
        elif "productivity" in agent_lower:
            return ReasoningType.TEMPORAL_ANALYSIS
        
        # Task-specific mapping
        if "analysis" in task_lower:
            return ReasoningType.DATA_ANALYSIS
        elif "recommendation" in task_lower:
            return ReasoningType.STRATEGIC_REASONING
        elif "generation" in task_lower or "writing" in task_lower:
            return ReasoningType.CREATIVE_WRITING
        
        # Default
        return ReasoningType.CONVERSATIONAL

    def get_cognitive_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from cognitive model manager"""
        return self.cognitive_manager.get_model_performance_stats()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        try:
            cache_keys = self.redis_client.keys("model_cache:*")
            total_cached_responses = len(cache_keys)
            
            # Get cache hit statistics from recent performance data
            stats = self.cognitive_manager.get_model_performance_stats()
            
            return {
                "total_cached_responses": total_cached_responses,
                "cache_keys_sample": cache_keys[:10],  # First 10 keys as sample
                "model_performance": stats
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    def clear_cognitive_cache(self, reasoning_type: Optional[ReasoningType] = None):
        """Clear cognitive model cache"""
        self.cognitive_manager.clear_cache(reasoning_type)

    def benchmark_models(self, test_prompts: Dict[ReasoningType, str]) -> Dict[str, Any]:
        """Benchmark different models across reasoning types"""
        results = {}
        
        for reasoning_type, prompt in test_prompts.items():
            self.logger.info(f"Benchmarking {reasoning_type.value}")
            
            start_time = datetime.now()
            try:
                response = self.cognitive_manager.generate_response(
                    prompt=prompt,
                    reasoning_type=reasoning_type
                )
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                results[reasoning_type.value] = {
                    "success": True,
                    "response_time": response_time,
                    "response_length": len(response),
                    "model_used": self.cognitive_manager.specialized_models.get(reasoning_type, "unknown")
                }
                
            except Exception as e:
                results[reasoning_type.value] = {
                    "success": False,
                    "error": str(e),
                    "response_time": 0
                }
        
        return results

    # Backward compatibility methods
    def generate_response_legacy(self, prompt: str) -> str:
        """Legacy method for backward compatibility"""
        return self._simple_generate(prompt, self.default_model)