#!/usr/bin/env python3

import sys
import os
import redis

# Add the JURIX root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_fixed_agents():
    print("🔧 Testing Fixed Agent Integration")
    print("=" * 40)
    
    # Test Redis connection
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"❌ Redis failed: {e}")
        return False
    
    # Test 1: Create BaseAgent with vector memory
    print("\n1. Testing Enhanced BaseAgent...")
    try:
        from agents.base_agent import BaseAgent
        agent = BaseAgent("test_agent", redis_client=redis_client)
        
        # Check if vector memory is available
        has_vector_memory = hasattr(agent.mental_state, 'vector_memory')
        vector_memory_works = has_vector_memory and agent.mental_state.vector_memory is not None
        
        print(f"   Has vector_memory: {has_vector_memory}")
        print(f"   Vector memory works: {vector_memory_works}")
        
        if vector_memory_works:
            print("   ✅ BaseAgent has working semantic memory")
        else:
            print("   ❌ BaseAgent semantic memory failed")
            return False
            
    except Exception as e:
        print(f"   ❌ BaseAgent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Test adding experience
    print("\n2. Testing experience storage...")
    try:
        agent.mental_state.add_experience(
            experience_description="User asked for advice on PROJ123 velocity optimization and bottleneck identification",
            outcome="provided_comprehensive_advice",
            confidence=0.9,
            metadata={"project": "PROJ123", "topics": ["velocity", "bottlenecks", "advice"]}
        )
        print("   ✅ Experience stored successfully")
    except Exception as e:
        print(f"   ❌ Experience storage failed: {e}")
        return False
    
    # Test 3: Test memory search  
    print("\n3. Testing memory search...")
    try:
        memories = agent.mental_state.recall_similar_experiences("PROJ123 advice", max_results=5)
        print(f"   ✅ Found {len(memories)} similar experiences")
        
        for i, memory in enumerate(memories, 1):
            print(f"      {i}. {memory.content[:60]}...")
            
    except Exception as e:
        print(f"   ❌ Memory search failed: {e}")
        return False
    
    # Test 4: Test ChatAgent specifically
    print("\n4. Testing ChatAgent...")
    try:
        from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
        from agents.chat_agent import ChatAgent
        
        shared_memory = JurixSharedMemory()
        chat_agent = ChatAgent(shared_memory)
        
        # Check if chat agent has semantic memory
        has_vector_memory = hasattr(chat_agent.mental_state, 'vector_memory')
        vector_memory_works = has_vector_memory and chat_agent.mental_state.vector_memory is not None
        
        print(f"   ChatAgent has vector_memory: {has_vector_memory}")
        print(f"   ChatAgent vector memory works: {vector_memory_works}")
        
        if vector_memory_works:
            print("   ✅ ChatAgent has working semantic memory")
            
            # Test adding experience through ChatAgent
            chat_agent.mental_state.add_experience(
                experience_description="ChatAgent test: handling PROJ123 discussion about sprint optimization",
                outcome="test_successful", 
                confidence=0.8,
                metadata={"agent": "chat_agent", "project": "PROJ123"}
            )
            print("   ✅ ChatAgent experience storage works")
            
        else:
            print("   ❌ ChatAgent semantic memory failed")
            return False
            
    except Exception as e:
        print(f"   ❌ ChatAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Search across all agents
    print("\n5. Testing cross-agent search...")
    try:
        from orchestrator.memory.vector_memory_manager import VectorMemoryManager # type: ignore
        
        vm = VectorMemoryManager(redis_client)
        all_memories = vm.search_memories("PROJ123", max_results=10)
        
        print(f"   ✅ Found {len(all_memories)} total memories about PROJ123")
        for i, memory in enumerate(all_memories, 1):
            print(f"      {i}. [{memory.agent_id}] {memory.content[:50]}...")
            
    except Exception as e:
        print(f"   ❌ Cross-agent search failed: {e}")
        return False
    
    print("\n🎉 All agent tests passed! Semantic memory is fully working.")
    return True

if __name__ == "__main__":
    success = test_fixed_agents()
    if success:
        print("\n🚀 Ready to test! Run:")
        print("python main.py")
        print("Then try: 'advice on PROJ123' followed by 'search PROJ123'")
    else:
        print("\n❌ Fix needed. Check errors above.")