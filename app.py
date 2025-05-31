import sys
import os
from typing import Dict, Any
from flask import Flask, request, jsonify
from langgraph.graph import StateGraph, END # type: ignore
from orchestrator.graph.state import JurixState # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from agents.jira_data_agent import JiraDataAgent
from agents.recommendation_agent import RecommendationAgent
from agents.productivity_dashboard_agent import ProductivityDashboardAgent
from orchestrator.core.productivity_workflow import build_productivity_workflow, run_productivity_workflow  # type: ignore # Assuming workflow is in a separate file

app = Flask(__name__)

# Initialize agents and shared memory
shared_memory = JurixSharedMemory()
jira_data_agent = JiraDataAgent()
recommendation_agent = RecommendationAgent(shared_memory)
productivity_dashboard_agent = ProductivityDashboardAgent()

# Compile the workflow
workflow = build_productivity_workflow()

@app.route('/dashboard', methods=['POST'])
def get_dashboard():
    try:
        data = request.get_json()
        project_id = data.get("project_id", "PROJ123")
        time_range = data.get("time_range", {
            "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-15T23:59:59Z"
        })
        
        state = run_productivity_workflow(project_id, time_range)
        
        if state["workflow_status"] == "failure":
            return jsonify({"error": state.get("error", "Workflow failed"), "status": "failure"}), 500
        
        return jsonify(state), 200
    except Exception as e:
        return jsonify({"error": str(e), "status": "failure"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)