import sys
import os
import logging
import redis
import json

# Add the JURIX root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from orchestrator.core.orchestrator import run_workflow # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
import uuid

def setup_logging():
    """Setup enhanced logging for Redis integration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('workflow_debug.log'),
            logging.StreamHandler()
        ]
    )

def test_redis_connection():
    """Test Redis connection before starting the workflow"""
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connection successful!")
        return True
    except redis.ConnectionError:
        print("❌ Redis connection failed!")
        print("Please make sure Redis is running:")
        print("  Docker: docker start redis-stack")
        print("  Native: redis-server")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False

def interactive_workflow():
    """Enhanced interactive workflow with Redis"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger("Main")
    
    # Test Redis connection
    if not test_redis_connection():
        return
    
    # Initialize shared memory
    try:
        shared_memory = JurixSharedMemory()
        logger.info("Shared memory initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize shared memory: {e}")
        return
    
    conversation_id = str(uuid.uuid4())
    logger.info(f"Starting interactive workflow with conversation ID: {conversation_id}")
    
    print("\n🤖 JURIX Interactive Workflow (Redis-Enhanced)")
    print("=" * 50)
    print("Enter your questions about Agile, software development, or project management.")
    print("Special commands:")
    print("  'stats' - Show Redis memory statistics")
    print("  'mental' - Show agent mental states")
    print("  'models' - Show model performance statistics")
    print("  'memory' - Show semantic memory analysis")
    print("  'workflows' - Show workflow intelligence")
    print("  'search <query>' - Search semantic memories")
    print("  'cleanup' - Clean up old workflows")
    print("  'clear' - Clear conversation history")
    print("  'quit' - Exit")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n🎯 Your question > ").strip()
            
            if query.lower() == "quit":
                print("👋 Goodbye!")
                break
                
            elif query.lower() == "stats":
                # Show Redis statistics
                stats = shared_memory.get_memory_stats()
                print(f"\n📊 Redis Memory Statistics:")
                print(f"   Used Memory: {stats.get('used_memory_human', 'N/A')}")
                print(f"   Total Keys: {stats.get('total_keys', 'N/A')}")
                print(f"   Connected Clients: {stats.get('connected_clients', 'N/A')}")
            elif query.lower() == "memory":
                # Show semantic memory statistics
                print("\n🧠 Semantic Memory Analysis:")
                try:
                    from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
                    
                    vector_memory = VectorMemoryManager(shared_memory.redis_client)
                    insights = vector_memory.get_memory_insights()
                    
                    print(f"   📊 Total Memories: {insights.get('total_memories', 0)}")
                    print(f"   🤖 Memories by Agent:")
                    for agent_id, count in insights.get('memories_by_agent', {}).items():
                        print(f"      {agent_id}: {count} memories")
                    
                    print(f"   📝 Memories by Type:")
                    for mem_type, count in insights.get('memories_by_type', {}).items():
                        print(f"      {mem_type}: {count} memories")
                    
                    # Show vector indices
                    indices = insights.get('memory_indices', {})
                    print(f"   🔍 Vector Indices:")
                    print(f"      Agent indices: {indices.get('agent_indices', 0)}")
                    print(f"      Type indices: {indices.get('type_indices', 0)}")
                    
                except Exception as e:
                    print(f"   ❌ Error retrieving memory stats: {e}")
            elif query.lower() == "workflows":
                # Show workflow performance and insights
                print("\n🔄 Workflow Intelligence:")
                try:
                    from orchestrator.memory.persistent_langraph_state import LangGraphRedisManager # type: ignore
                    
                    workflow_manager = LangGraphRedisManager(shared_memory.redis_client)
                    insights = workflow_manager.get_workflow_insights()
                    
                    if insights.get("global_performance"):
                        perf = insights["global_performance"]
                        print(f"   📊 Global Performance:")
                        print(f"      Total Workflows: {perf.get('total_workflows', 0)}")
                        print(f"      Successful: {perf.get('successful_workflows', 0)}")
                        print(f"      Failed: {perf.get('failed_workflows', 0)}")
                        print(f"      Avg Execution Time: {perf.get('avg_execution_time', 0):.2f}s")
                        
                        if perf.get('total_workflows', 0) > 0:
                            success_rate = (perf.get('successful_workflows', 0) / perf.get('total_workflows', 1)) * 100
                            print(f"      Success Rate: {success_rate:.1f}%")
                    
                    print(f"   🏃 Active Workflows: {insights.get('active_workflows_count', 0)}")
                    
                    print(f"   📈 Workflows by Type:")
                    for workflow_type in ["general_orchestration", "productivity_analysis", "jira_article_generation"]:
                        count = insights.get(f"{workflow_type}_workflows", 0)
                        print(f"      {workflow_type}: {count}")
                    
                    if insights.get("recent_patterns"):
                        recent = insights["recent_patterns"]
                        print(f"   🎯 Recent Patterns:")
                        print(f"      Recent Workflows: {recent.get('total_recent', 0)}")
                        print(f"      Recent Success Rate: {recent.get('success_rate', 0):.1f}%")
                    
                except Exception as e:
                    print(f"   ❌ Error retrieving workflow stats: {e}")
                continue
                
            elif query.lower() == "cleanup":
                # Clean up old workflows
                print("\n🧹 Cleaning up old workflows...")
                try:
                    from orchestrator.memory.persistent_langraph_state import LangGraphRedisManager # type: ignore
                    
                    workflow_manager = LangGraphRedisManager(shared_memory.redis_client)
                    cleaned_count = workflow_manager.cleanup_old_workflows(max_age_days=7)
                    print(f"   ✅ Cleaned up {cleaned_count} old workflows")
                    
                except Exception as e:
                    print(f"   ❌ Cleanup error: {e}")
                continue
                
            elif query.lower().startswith("search "):
                # Search semantic memory
                search_query = query[7:]  # Remove "search " prefix
                print(f"\n🔍 Searching memories for: '{search_query}'")
                try:
                    from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
                    
                    vector_memory = VectorMemoryManager(shared_memory.redis_client)
                    results = vector_memory.search_memories(search_query, max_results=5)
                    
                    if results:
                        print(f"   Found {len(results)} relevant memories:")
                        for i, memory in enumerate(results, 1):
                            print(f"   {i}. [{memory.memory_type.value}] {memory.content[:100]}...")
                            print(f"      Agent: {memory.agent_id} | Confidence: {memory.confidence:.2f}")
                            print(f"      Last accessed: {memory.access_count} times")
                            print()
                    else:
                        print("   No relevant memories found")
                        
                except Exception as e:
                    print(f"   ❌ Search error: {e}")
                continue
                
            elif query.lower() == "clear":
                # Clear conversation history
                shared_memory.redis_client.delete(f"conversation:{conversation_id}")
                print("🗑️ Conversation history cleared!")
                conversation_id = str(uuid.uuid4())  # New conversation ID
                continue
                
            elif query.lower() == "mental":
                # Show agent mental states
                print("\n🧠 Agent Mental States:")
                try:
                    # Get all agent mental states from Redis
                    redis_client = shared_memory.redis_client
                    mental_state_keys = redis_client.keys("mental_state:*")
                    
                    if not mental_state_keys:
                        print("   No active agent mental states found")
                    else:
                        for key in mental_state_keys:
                            agent_data = redis_client.get(key)
                            if agent_data:
                                try:
                                    state = json.loads(agent_data)
                                    agent_id = state.get("agent_id", "unknown")
                                    beliefs_count = len(state.get("beliefs", {}))
                                    decisions_count = len(state.get("decisions", []))
                                    capabilities = state.get("capabilities", [])
                                    
                                    print(f"   🤖 Agent: {agent_id}")
                                    print(f"      Capabilities: {', '.join(capabilities)}")
                                    print(f"      Beliefs: {beliefs_count}")
                                    print(f"      Decisions: {decisions_count}")
                                    
                                    # Show some recent beliefs
                                    beliefs = state.get("beliefs", {})
                                    if beliefs:
                                        print("      Recent beliefs:")
                                        for belief_key, belief_data in list(beliefs.items())[:3]:
                                            confidence = belief_data.get("confidence", 0)
                                            print(f"        - {belief_key}: {confidence:.2f} confidence")
                                    print()
                                    
                                except json.JSONDecodeError:
                                    print(f"   ⚠️ Could not parse state for {key}")
                    
                    # Show Redis memory usage
                    stats = shared_memory.get_memory_stats()
                    print(f"   📊 Redis Status:")
                    print(f"      Total Keys: {stats.get('total_keys', 'N/A')}")
                    print(f"      Memory Used: {stats.get('used_memory_human', 'N/A')}")
                    
                except Exception as e:
                    print(f"   ❌ Error retrieving mental states: {e}")
                continue
                
            elif query.lower() == "models":
                # Show model performance statistics
                print("\n🧠 Model Performance Statistics:")
                try:
                    # This would require importing the orchestrator components
                    from orchestrator.core.model_manager import ModelManager # type: ignore
                    
                    # Create a temporary model manager to get stats
                    temp_model_manager = ModelManager(redis_client=shared_memory.redis_client)
                    
                    # Get cognitive performance stats
                    perf_stats = temp_model_manager.get_cognitive_performance_stats()
                    cache_stats = temp_model_manager.get_cache_stats()
                    
                    if perf_stats:
                        print("   🎯 Reasoning Type Performance:")
                        for key, stats in perf_stats.items():
                            reasoning_type = stats.get('reasoning_type', 'unknown')
                            model = stats.get('model', 'unknown')
                            success_rate = stats.get('success_rate', 0)
                            avg_time = stats.get('avg_response_time', 0)
                            total_requests = stats.get('total_requests', 0)
                            
                            print(f"      {reasoning_type} ({model}):")
                            print(f"        Requests: {total_requests}")
                            print(f"        Success Rate: {success_rate:.1%}")
                            print(f"        Avg Response Time: {avg_time:.2f}s")
                    
                    print(f"\n   💾 Cache Performance:")
                    print(f"      Cached Responses: {cache_stats.get('total_cached_responses', 0)}")
                    
                except Exception as e:
                    print(f"   ❌ Error retrieving model stats: {e}")
                continue

            elif query.lower() == "hybrid":
                # Show hybrid architecture status
                print("\n🔄 Hybrid Architecture Status:")
                try:
                    # Check if collaborative framework is working
                    from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore
                    print("   ✅ Collaborative Framework: Available")
                    
                    # Show recent collaborations
                    recent_collabs = shared_memory.redis_client.lrange("collaboration_history", 0, 4)
                    if recent_collabs:
                        print(f"   📊 Recent Collaborations: {len(recent_collabs)}")
                        for i, collab_data in enumerate(recent_collabs, 1):
                            collab = json.loads(collab_data)
                            primary = collab.get("primary_agent", "unknown")
                            collaborators = collab.get("collaborating_agents", [])
                            print(f"      {i}. {primary} + {collaborators}")
                    else:
                        print("   📊 No recent collaborations found")
                        
                except Exception as e:
                    print(f"   ❌ Hybrid Architecture Error: {e}")
                continue

            # Add this as a new command in your interactive_workflow function
            elif query.lower() == "testcollab":
                # Test collaboration
                test_query = "Give me productivity recommendations for project PROJ123"
                print(f"🧪 Testing collaboration with: {test_query}")
                
                final_state = run_workflow(test_query)
                
                # Check collaboration results
                collab_meta = final_state.get("collaboration_metadata", {})
                final_collab = final_state.get("final_collaboration_summary", {})
                
                if collab_meta or final_collab:
                    print("🎉 COLLABORATION IS WORKING!")
                    print(f"Collaborating agents: {collab_meta.get('collaborating_agents', [])}")
                else:
                    print("❌ Collaboration not detected")

            elif query.lower() == "debug":
                # Debug workflow step by step
                test_query = "Give me recommendations for PROJ123"
                
                print("🔍 Step-by-step workflow debug:")
                result = run_workflow(test_query)
                
                print("📊 Complete result structure:")
                import json
                print(json.dumps(list(result.keys()), indent=2))
                
                print("\n🔎 All keys containing 'collab':")
                collab_keys = [k for k in result.keys() if 'collab' in k.lower()]
                for key in collab_keys:
                    print(f"   {key}: {result[key]}")
                
                continue                
            if not query:
                continue
            
            # Process the query
            logger.info(f"Processing query: {query}")
            final_state = run_workflow(query, conversation_id)
            
            # Log the final state
            logger.info(f"Final state: {final_state}")
            
            # Extract and display response
            if "response" in final_state:
                response = final_state["response"]
                logger.info(f"Response: {response}")
            else:
                response = "No response generated"
                logger.warning("No response found in final_state")
            
            # Display results
            print(f"\n🎯 Intent: {final_state.get('intent', {}).get('intent', 'Unknown')}")
            print(f"🤖 Response: {response}")
            
            # Show additional information if available
            articles = final_state.get('articles', [])
            if articles:
                print(f"📚 Articles found: {len(articles)}")
                
            recommendations = final_state.get('recommendations', [])
            if recommendations:
                print(f"💡 Recommendations: {len(recommendations)}")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec[:100]}...")
            
            tickets = final_state.get('tickets', [])
            if tickets:
                print(f"🎫 Tickets retrieved: {len(tickets)}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
            
        except Exception as e:
            logger.error(f"Error in interactive workflow: {str(e)}")
            print(f"❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

def run_single_query(query: str):
    """Run a single query for testing"""
    setup_logging()
    
    if not test_redis_connection():
        return
    
    try:
        conversation_id = str(uuid.uuid4())
        final_state = run_workflow(query, conversation_id)
        
        print(f"Query: {query}")
        print(f"Intent: {final_state.get('intent', {}).get('intent', 'Unknown')}")
        print(f"Response: {final_state.get('response', 'No response')}")
        print(f"Articles: {len(final_state.get('articles', []))}")
        print(f"Recommendations: {len(final_state.get('recommendations', []))}")
        print(f"Tickets: {len(final_state.get('tickets', []))}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        # Run single query mode
        query = " ".join(sys.argv[1:])
        run_single_query(query)
    else:
        # Run interactive mode
        interactive_workflow()