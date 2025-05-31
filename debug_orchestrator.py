# trace_metadata.py
# Let's trace exactly where the collaboration metadata gets lost

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def trace_node_execution():
    """Trace a single node execution to see metadata flow"""
    print("🔍 TRACING COLLABORATION METADATA FLOW")
    print("=" * 60)
    
    try:
        from orchestrator.graph.state import JurixState # type: ignore
        import orchestrator.core.orchestrator as orch # type: ignore
        
        # Create test state
        test_state = JurixState(
            query="Give me recommendations for PROJ123",
            intent={"intent": "recommendation", "project": "PROJ123"},
            conversation_id="test_123",
            conversation_history=[],
            articles=[],
            recommendations=[],
            tickets=[],
            status="pending",
            response="",
            articles_used=[],
            workflow_status="",
            next_agent="",
            project="PROJ123"
        )
        
        print(f"📝 BEFORE collaborative_data_node:")
        print(f"   State keys: {list(test_state.keys())}")
        print(f"   Collab keys: {[k for k in test_state.keys() if 'collab' in k.lower()]}")
        
        # Execute the collaborative data node
        print(f"\n🎯 EXECUTING collaborative_data_node...")
        result_state = orch.collaborative_data_node(test_state)
        
        print(f"\n📝 AFTER collaborative_data_node:")
        print(f"   State keys: {list(result_state.keys())}")
        print(f"   Collab keys: {[k for k in result_state.keys() if 'collab' in k.lower()]}")
        
        # Check specific metadata
        collab_meta = result_state.get("collaboration_metadata")
        if collab_meta:
            print(f"   🎉 FOUND collaboration_metadata: {collab_meta}")
        else:
            print(f"   ❌ NO collaboration_metadata found")
        
        # Check all possible variations
        possible_keys = [
            "collaboration_metadata", "collab_metadata", "collaborative_metadata",
            "final_collaboration_summary", "collaboration_info"
        ]
        
        print(f"\n🔎 Checking all possible metadata keys:")
        for key in possible_keys:
            value = result_state.get(key)
            if value:
                print(f"   ✅ {key}: {value}")
            else:
                print(f"   ❌ {key}: None")
        
        return result_state
        
    except Exception as e:
        print(f"❌ Node execution trace failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

def trace_collaborative_framework_direct():
    """Test CollaborativeFramework directly"""
    print("\n🧪 TESTING COLLABORATIVE FRAMEWORK DIRECTLY")
    print("=" * 60)
    
    try:
        import asyncio
        from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
        from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore
        from agents.chat_agent import ChatAgent
        from agents.jira_data_agent import JiraDataAgent
        
        # Setup
        shared_memory = JurixSharedMemory()
        chat_agent = ChatAgent(shared_memory)
        jira_data_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
        
        agents_registry = {
            "chat_agent": chat_agent,
            "jira_data_agent": jira_data_agent
        }
        
        collaborative_framework = CollaborativeFramework(shared_memory.redis_client, agents_registry)
        
        # Test direct coordination
        async def test_direct():
            task_context = {
                "project_id": "PROJ123",
                "time_range": {
                    "start": "2025-05-01T00:00:00Z",
                    "end": "2025-05-17T23:59:59Z"
                },
                "user_query": "test query"
            }
            
            print("🚀 Calling coordinate_agents directly...")
            result = await collaborative_framework.coordinate_agents("jira_data_agent", task_context)
            return result
        
        # Execute
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            direct_result = loop.run_until_complete(test_direct())
            
            print(f"📊 DIRECT FRAMEWORK RESULT:")
            print(f"   Result type: {type(direct_result)}")
            print(f"   Result keys: {list(direct_result.keys()) if isinstance(direct_result, dict) else 'Not a dict'}")
            
            # Check for collaboration metadata
            if isinstance(direct_result, dict):
                collab_meta = direct_result.get("collaboration_metadata")
                if collab_meta:
                    print(f"   🎉 DIRECT collaboration_metadata: {collab_meta}")
                    return True
                else:
                    print(f"   ❌ NO collaboration_metadata in direct result")
                    
                    # Show all keys with 'collab' in them
                    collab_keys = [k for k in direct_result.keys() if 'collab' in k.lower()]
                    if collab_keys:
                        print(f"   🔍 Found collab-related keys: {collab_keys}")
                        for key in collab_keys:
                            print(f"      {key}: {direct_result[key]}")
            
            return False
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Direct framework test failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def check_collaborative_framework_methods():
    """Check what the CollaborativeFramework actually returns"""
    print("\n🔬 INSPECTING COLLABORATIVE FRAMEWORK METHODS")
    print("=" * 60)
    
    try:
        from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore
        from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
        
        # Create minimal framework
        shared_memory = JurixSharedMemory()
        framework = CollaborativeFramework(shared_memory.redis_client, {})
        
        # Check what methods exist
        methods = [method for method in dir(framework) if not method.startswith('_')]
        print(f"📝 Available methods: {methods}")
        
        # Check the coordinate_agents method signature
        import inspect
        sig = inspect.signature(framework.coordinate_agents)
        print(f"🔍 coordinate_agents signature: {sig}")
        
        # Look at the source code
        try:
            source = inspect.getsource(framework.coordinate_agents)
            print(f"\n📄 coordinate_agents source (first 10 lines):")
            for i, line in enumerate(source.split('\n')[:10]):
                print(f"   {i+1}: {line}")
        except:
            print("   ❌ Cannot get source code")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework inspection failed: {e}")
        return False

def trace_workflow_stream():
    """Trace the workflow.stream() to see where metadata gets lost"""
    print("\n🌊 TRACING WORKFLOW STREAM")
    print("=" * 60)
    
    try:
        from orchestrator.core.orchestrator import run_workflow # type: ignore
        import orchestrator.core.orchestrator as orch # type: ignore
        
        # Monkey patch the collaborative nodes to add logging
        original_collaborative_data_node = orch.collaborative_data_node
        
        def logged_collaborative_data_node(state):
            print(f"🔥 ENTERING collaborative_data_node")
            result = original_collaborative_data_node(state)
            print(f"🔥 EXITING collaborative_data_node")
            print(f"   Result keys: {list(result.keys())}")
            
            collab_meta = result.get("collaboration_metadata")
            if collab_meta:
                print(f"   🎉 Node HAS collaboration_metadata: {collab_meta}")
            else:
                print(f"   ❌ Node has NO collaboration_metadata")
            
            return result
        
        # Replace temporarily
        orch.collaborative_data_node = logged_collaborative_data_node
        
        # Run workflow
        print("🚀 Running workflow with logging...")
        final_result = run_workflow("Give me recommendations for PROJ123")
        
        # Restore original
        orch.collaborative_data_node = original_collaborative_data_node
        
        print(f"\n📊 FINAL WORKFLOW RESULT:")
        print(f"   Final keys: {list(final_result.keys())}")
        
        collab_keys = [k for k in final_result.keys() if 'collab' in k.lower()]
        if collab_keys:
            print(f"   🎉 Final collab keys: {collab_keys}")
        else:
            print(f"   ❌ NO collab keys in final result")
        
        return final_result
        
    except Exception as e:
        print(f"❌ Workflow stream trace failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

def run_complete_trace():
    """Run complete tracing to find where metadata disappears"""
    print("🕵️ COMPLETE COLLABORATION METADATA TRACE")
    print("=" * 70)
    
    # Step 1: Test direct framework
    print("STEP 1: Direct Framework Test")
    direct_works = trace_collaborative_framework_direct()
    
    # Step 2: Test individual node
    print("\nSTEP 2: Individual Node Test")
    node_result = trace_node_execution()
    
    # Step 3: Inspect framework methods
    print("\nSTEP 3: Framework Method Inspection")
    framework_ok = check_collaborative_framework_methods()
    
    # Step 4: Trace workflow stream
    print("\nSTEP 4: Workflow Stream Trace")
    workflow_result = trace_workflow_stream()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 TRACE SUMMARY")
    print("=" * 70)
    
    print(f"✅ Direct framework works: {direct_works}")
    print(f"✅ Individual node works: {node_result is not None}")
    print(f"✅ Framework methods OK: {framework_ok}")
    print(f"✅ Workflow completes: {workflow_result is not None}")
    
    if direct_works and node_result and not workflow_result:
        print("\n🔍 DIAGNOSIS: Workflow stream is losing metadata")
        print("   SOLUTION: Check LangGraph state handling")
    elif not direct_works:
        print("\n🔍 DIAGNOSIS: CollaborativeFramework doesn't return metadata")
        print("   SOLUTION: Fix CollaborativeFramework.coordinate_agents()")
    else:
        print("\n🤔 DIAGNOSIS: Need deeper investigation")

if __name__ == "__main__":
    run_complete_trace()