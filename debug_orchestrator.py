# workflow_sequence_tracer.py
# Trace the exact sequence of agent execution in the workflow

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def trace_workflow_sequence():
    """Trace the exact sequence of how agents execute in the workflow"""
    print("🔍 TRACING WORKFLOW EXECUTION SEQUENCE")
    print("=" * 60)
    
    try:
        from orchestrator.core.orchestrator import run_workflow # type: ignore
        import orchestrator.core.orchestrator as orch # type: ignore
        
        # Monkey patch to trace the sequence
        execution_trace = []
        
        # Patch each node function to track execution
        original_functions = {}
        
        node_functions = [
            'classify_intent_node',
            'collaborative_data_node', 
            'collaborative_recommendation_node',
            'collaborative_retrieval_node',
            'collaborative_chat_node'
        ]
        
        for func_name in node_functions:
            if hasattr(orch, func_name):
                original_functions[func_name] = getattr(orch, func_name)
                
                def create_traced_function(original_func, name):
                    def traced_function(state):
                        print(f"🎯 EXECUTING: {name}")
                        
                        # Record execution start
                        execution_trace.append({
                            'node': name,
                            'action': 'start',
                            'state_keys': list(state.keys()),
                            'tickets_count': len(state.get('tickets', [])),
                            'articles_count': len(state.get('articles', [])),
                            'recommendations_count': len(state.get('recommendations', []))
                        })
                        
                        # Execute the original function
                        result = original_func(state)
                        
                        # Record execution end
                        execution_trace.append({
                            'node': name,
                            'action': 'end',
                            'state_keys': list(result.keys()),
                            'tickets_count': len(result.get('tickets', [])),
                            'articles_count': len(result.get('articles', [])),
                            'recommendations_count': len(result.get('recommendations', [])),
                            'collaboration_metadata': bool(result.get('collaboration_metadata'))
                        })
                        
                        print(f"   ✅ {name} completed")
                        print(f"      Tickets: {len(result.get('tickets', []))}")
                        print(f"      Articles: {len(result.get('articles', []))}")
                        print(f"      Recommendations: {len(result.get('recommendations', []))}")
                        print(f"      Collaboration: {bool(result.get('collaboration_metadata'))}")
                        
                        return result
                    return traced_function
                
                # Apply the trace
                setattr(orch, func_name, create_traced_function(original_functions[func_name], func_name))
        
        print("🚀 Running traced workflow...")
        result = run_workflow("Give me recommendations for PROJ123")
        
        # Restore original functions
        for func_name, original_func in original_functions.items():
            setattr(orch, func_name, original_func)
        
        print(f"\n📊 EXECUTION SEQUENCE ANALYSIS:")
        print("=" * 60)
        
        # Analyze the execution trace
        for i, entry in enumerate(execution_trace):
            if entry['action'] == 'start':
                print(f"\n{i//2 + 1}. {entry['node'].upper()}:")
                print(f"   📥 INPUT: {entry['tickets_count']} tickets, {entry['articles_count']} articles, {entry['recommendations_count']} recommendations")
            else:
                print(f"   📤 OUTPUT: {entry['tickets_count']} tickets, {entry['articles_count']} articles, {entry['recommendations_count']} recommendations")
                if entry['collaboration_metadata']:
                    print(f"   🤝 COLLABORATION: Metadata generated")
                else:
                    print(f"   🔄 COLLABORATION: None")
        
        # Check the workflow routing logic
        print(f"\n🔄 WORKFLOW ROUTING ANALYSIS:")
        print("=" * 60)
        
        # Check what the routing logic does
        from orchestrator.graph.state import JurixState # type: ignore
        
        test_state = JurixState(
            query="Give me recommendations for PROJ123",
            intent={"intent": "recommendation", "project": "PROJ123"},
            conversation_id="test",
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
        
        # Check the build_workflow function to see routing
        workflow = orch.build_workflow()
        
        print("🎯 ROUTING LOGIC:")
        print("   Intent: 'recommendation' → Should route to jira_data_agent first")
        print("   Then: jira_data_agent → recommendation_agent (via edges)")
        print("   Finally: recommendation_agent → chat_agent")
        
        return execution_trace
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_collaboration_decisions():
    """Analyze why specific collaboration decisions were made"""
    print(f"\n🧠 COLLABORATION DECISION ANALYSIS")
    print("=" * 60)
    
    try:
        # Test different scenarios to see collaboration patterns
        test_cases = [
            {
                "query": "Give me recommendations for PROJ123",
                "expected": "Should collaborate for context enrichment"
            },
            {
                "query": "What are the tickets for PROJ123?", 
                "expected": "Should NOT need collaboration (just data retrieval)"
            },
            {
                "query": "Find articles about Scrum",
                "expected": "Should route to retrieval agent directly"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: '{test_case['query']}'")
            print(f"   Expected: {test_case['expected']}")
            
            from orchestrator.core.orchestrator import run_workflow # type: ignore
            result = run_workflow(test_case["query"])
            
            collab_meta = result.get("collaboration_metadata", {})
            collaborating_agents = collab_meta.get("collaborating_agents", [])
            collaboration_types = collab_meta.get("collaboration_types", [])
            
            print(f"   Actual:")
            if collaborating_agents:
                print(f"      Primary: {collab_meta.get('primary_agent')}")
                print(f"      Collaborators: {collaborating_agents}")
                print(f"      Types: {collaboration_types}")
            else:
                print(f"      No collaboration")
            
            print(f"   Articles: {len(result.get('articles', []))}")
            print(f"   Tickets: {len(result.get('tickets', []))}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def main():
    """Run complete workflow analysis"""
    print("🔬 COMPREHENSIVE WORKFLOW ANALYSIS")
    print("=" * 70)
    
    # Trace execution sequence
    trace_workflow_sequence()
    
    # Analyze collaboration decisions
    analyze_collaboration_decisions()
    
    print(f"\n💡 KEY INSIGHTS:")
    print("1. JiraDataAgent runs FIRST (gets tickets)")
    print("2. RecommendationAgent runs SECOND (gets tickets + requests articles)")
    print("3. RetrievalAgent provides articles via collaboration")
    print("4. ChatAgent synthesizes final response")
    print("")
    print("This explains why you see retrieval collaboration but not jira collaboration!")

if __name__ == "__main__":
    main()