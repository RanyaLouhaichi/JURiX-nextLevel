# main.py (updated to use the single orchestrator)
import sys
import os
from orchestrator.core.orchestrator import orchestrator # type: ignore
import uuid

def interactive_workflow():
    """Interactive command-line interface"""
    print("\n🤖 JURIX AI System")
    print("=" * 60)
    print("Type your questions or commands:")
    print("  'dashboard <project>' - Generate productivity dashboard")
    print("  'article <ticket>' - Generate article from ticket")
    print("  'api' - Start API server")
    print("  'quit' - Exit")
    print("=" * 60)
    
    conversation_id = str(uuid.uuid4())
    
    while True:
        try:
            query = input("\n🎯 Your question > ").strip()
            
            if query.lower() == "quit":
                print("👋 Goodbye!")
                break
                
            elif query.lower() == "api":
                print("Starting API server...")
                from api import app
                app.run(debug=True, host='0.0.0.0', port=5000)
                
            elif query.lower().startswith("dashboard"):
                parts = query.split()
                project_id = parts[1] if len(parts) > 1 else "PROJ123"
                
                print(f"⏳ Generating dashboard for {project_id}...")
                state = orchestrator.run_productivity_workflow(
                    project_id,
                    {
                        "start": "2025-05-01T00:00:00Z",
                        "end": "2025-05-17T23:59:59Z"
                    }
                )
                
                print(f"\n📊 Dashboard generated!")
                print(f"Dashboard ID: {state.get('dashboard_id', 'N/A')}")
                print(f"Metrics: {bool(state.get('metrics'))}")
                print(f"Report: {state.get('report', '')[:200]}...")
                
            elif query.lower().startswith("article"):
                parts = query.split()
                ticket_id = parts[1] if len(parts) > 1 else "TICKET-001"
                
                print(f"⏳ Generating article for {ticket_id}...")
                state = orchestrator.run_jira_workflow(ticket_id)
                
                print(f"\n📄 Article generation status: {state.get('workflow_status')}")
                if state.get('article'):
                    print(f"Title: {state['article'].get('title', 'N/A')}")
                    print(f"Stage: {state.get('workflow_stage', 'N/A')}")
                
            else:
                # Regular chat workflow
                final_state = orchestrator.run_workflow(query, conversation_id)
                
                print(f"\n🤖 Response:")
                print(final_state.get("response", "No response generated"))
                
                # Show additional info if available
                if final_state.get("articles"):
                    print(f"\n📚 Used {len(final_state['articles'])} articles")
                if final_state.get("recommendations"):
                    print(f"\n💡 Generated {len(final_state['recommendations'])} recommendations")
                if final_state.get("collaboration_metadata"):
                    print(f"\n🤝 Collaboration occurred with: {final_state['collaboration_metadata'].get('collaborating_agents', [])}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def run_single_query(query: str):
    """Run a single query"""
    try:
        conversation_id = str(uuid.uuid4())
        final_state = orchestrator.run_workflow(query, conversation_id)
        
        print(f"Query: {query}")
        print("━" * 60)
        print(f"Response: {final_state.get('response', 'No response generated')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "api":
            from api import app
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            # Run single query
            query = " ".join(sys.argv[1:])
            run_single_query(query)
    else:
        # Run interactive mode
        interactive_workflow()