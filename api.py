from flask import Flask, request, jsonify
from orchestrator.core.orchestrator import orchestrator # type: ignore
import logging
from datetime import datetime
from aria_api import register_aria_routes # type: ignore
from flask_socketio import SocketIO # type: ignore

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO)

@app.route("/ask-orchestrator", methods=["POST"])
def ask_orchestrator():
    """General orchestration endpoint"""
    try:
        data = request.json
        query = data.get("query", "")
        conversation_id = data.get("conversation_id")
        
        result = orchestrator.run_workflow(query, conversation_id)
        
        return jsonify({
            "query": query,
            "response": result.get("response", ""),
            "articles": result.get("articles", []),
            "recommendations": result.get("recommendations", []),
            "predictions": result.get("predictions", {}),
            "collaboration_metadata": result.get("collaboration_metadata", {}),
            "workflow_status": result.get("workflow_status", "completed")
        })
    
    except Exception as e:
        logging.error(f"Error in ask-orchestrator: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard", methods=["POST"])
def dashboard():
    """Productivity dashboard endpoint"""
    try:
        data = request.get_json()
        project_id = data.get("project_id", "PROJ123")
        time_range = data.get("time_range", {
            "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-17T23:59:59Z"
        })
        
        state = orchestrator.run_productivity_workflow(project_id, time_range)
        
        response = {
            "project_id": state["project_id"],
            "time_range": state["time_range"],
            "tickets": state["tickets"],
            "metrics": state["metrics"],
            "visualization_data": state["visualization_data"],
            "recommendations": state["recommendations"],
            "report": state["report"],
            "predictions": state.get("predictions", {}),
            "workflow_status": state["workflow_status"],
            "dashboard_id": state.get("dashboard_id"),
            "collaboration_metadata": state.get("collaboration_metadata", {})
        }
        
        return jsonify(response), 200 if state["workflow_status"] == "success" else 500
    
    except Exception as e:
        logging.error(f"Error in dashboard endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predictions", methods=["POST"])
def predictions():
    """Predictive analysis endpoint"""
    try:
        data = request.get_json()
        project_id = data.get("project_id", "PROJ123")
        analysis_type = data.get("analysis_type", "comprehensive")
        
        state = orchestrator.run_predictive_workflow(project_id, analysis_type)
        
        response = {
            "project_id": project_id,
            "analysis_type": analysis_type,
            "predictions": state.get("predictions", {}),
            "recommendations": state.get("recommendations", []),
            "workflow_status": state.get("workflow_status", "failure"),
            "collaboration_metadata": state.get("collaboration_metadata", {})
        }
        
        return jsonify(response), 200 if state["workflow_status"] == "success" else 500
    
    except Exception as e:
        logging.error(f"Error in predictions endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/jira-workflow", methods=["POST"])
def trigger_jira_workflow():
    """Trigger jira article generation workflow"""
    data = request.json
    ticket_id = data.get("ticket_id", "TICKET-001")
    project_id = data.get("project_id", "PROJ123")
    
    if not ticket_id:
        return jsonify({"error": "ticket_id is required"}), 400
    
    final_state = orchestrator.run_jira_workflow(ticket_id, project_id=project_id)
    
    orchestrator.shared_memory.store(f"workflow_state_{ticket_id}", final_state)
    
    return jsonify({
        "ticket_id": ticket_id,
        "workflow_history": final_state.get("workflow_history", []),
        "current_state": {
            "article": final_state.get("article", {}),
            "redundant": final_state.get("redundant", False),
            "refinement_suggestion": final_state.get("refinement_suggestion"),
            "workflow_status": final_state.get("workflow_status", "unknown"),
            "workflow_stage": final_state.get("workflow_stage", "unknown"),
            "recommendation_id": final_state.get("recommendation_id"),
            "recommendations": final_state.get("recommendations", []),
            "autonomous_refinement_done": final_state.get("autonomous_refinement_done", False),
            "has_refined": final_state.get("has_refined", False)
        }
    })

@app.route("/jira-workflow/approve", methods=["POST"])
def approve_article():
    """Approve generated article"""
    data = request.json
    ticket_id = data.get("ticket_id")
    if not ticket_id:
        return jsonify({"error": "ticket_id is required"}), 400

    current_state = orchestrator.shared_memory.get(f"workflow_state_{ticket_id}")
    if not current_state:
        return jsonify({"error": f"No workflow state found for ticket {ticket_id}"}), 404

    current_state["approved"] = True
    current_state["workflow_stage"] = "complete"
    current_state["workflow_history"].append({
        "step": "approval_submitted",
        "article": current_state["article"],
        "redundant": current_state.get("redundant", False),
        "refinement_suggestion": current_state.get("refinement_suggestion"),
        "approved": True,
        "workflow_status": "success",
        "workflow_stage": "complete",
        "recommendation_id": current_state.get("recommendation_id"),
        "recommendations": current_state.get("recommendations", []),
        "autonomous_refinement_done": current_state.get("autonomous_refinement_done", False)
    })

    orchestrator.shared_memory.store(f"workflow_state_{ticket_id}", current_state)

    return jsonify({
        "ticket_id": ticket_id,
        "workflow_history": current_state.get("workflow_history", []),
        "current_state": {
            "article": current_state.get("article", {}),
            "redundant": current_state.get("redundant", False),
            "refinement_suggestion": current_state.get("refinement_suggestion"),
            "workflow_status": "success",
            "workflow_stage": "complete",
            "recommendation_id": current_state.get("recommendation_id"),
            "recommendations": current_state.get("recommendations", []),
            "autonomous_refinement_done": current_state.get("autonomous_refinement_done", False),
            "has_refined": current_state.get("has_refined", False),
            "approved": True
        }
    })

@app.route("/recommendations/<ticket_id>", methods=["GET"])
def get_recommendations(ticket_id):
    """Get recommendations for a specific ticket"""
    recommendation_id = request.args.get("recommendation_id")
    
    if recommendation_id:
        recommendations_data = orchestrator.shared_memory.get(recommendation_id)
        if recommendations_data:
            return jsonify(recommendations_data)
    
    current_state = orchestrator.shared_memory.get(f"workflow_state_{ticket_id}")
    if not current_state:
        return jsonify({"error": f"No workflow state found for ticket {ticket_id}"}), 404
    
    return jsonify({
        "ticket_id": ticket_id,
        "recommendation_id": current_state.get("recommendation_id"),
        "recommendations": current_state.get("recommendations", [])
    })

register_aria_routes(app, socketio)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)