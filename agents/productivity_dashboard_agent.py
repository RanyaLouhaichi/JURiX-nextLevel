from typing import Dict, Any, List
import json
import logging
from datetime import datetime
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.core.model_manager import ModelManager # type: ignore
import sqlite3
import threading
import time

class ProductivityDashboardAgent(BaseAgent):
    OBJECTIVE = "Analyze Jira ticket data to generate productivity metrics and visualization data"

    def __init__(self):
        super().__init__(name="productivity_dashboard_agent")
        self.model_manager = ModelManager()
        self.db_path = "shared_data.db"  
        self.last_throughput = 0  
        
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT
        ]
        
        self.mental_state.obligations.extend([
            "analyze_ticket_data",
            "generate_metrics", 
            "create_visualization_data",
            "generate_report",
            "check_for_updates"  
        ])
        
        self.log("Initialized ProductivityDashboardAgent")
        
        # Start the monitoring thread
        self.monitoring_thread = threading.Thread(target=self._check_for_updates_loop, daemon=True)
        self.monitoring_thread.start()

    def _check_for_updates(self):
        """Check for updates in shared memory and regenerate dashboard if significant changes are detected."""
        try:
            project_id = self.mental_state.beliefs.get("project_id", "PROJ123")  # Default project
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT has_changes, last_updated FROM updates WHERE project_id = ?", (project_id,))
            result = cursor.fetchone()
            
            if result and result[0]:  # has_changes is True
                self.log(f"Detected changes for project {project_id} at {result[1]}")
                # Fetch the updated tickets
                cursor.execute("SELECT tickets FROM tickets WHERE project_id = ?", (project_id,))
                tickets_result = cursor.fetchone()
                if tickets_result:
                    tickets = json.loads(tickets_result[0])
                    self.mental_state.beliefs["tickets"] = tickets
                    
                    # Recalculate metrics
                    metrics = self._analyze_ticket_data(tickets)
                    if metrics["throughput"] != self.last_throughput:
                        self.log(f"Throughput changed from {self.last_throughput} to {metrics['throughput']}")
                        self.last_throughput = metrics["throughput"]
                        
                        # Regenerate dashboard
                        visualization_data = self._create_visualization_data(metrics)
                        report = self._generate_report(metrics, self.mental_state.beliefs.get("recommendations", []))
                        
                        # Store updated dashboard in beliefs
                        self.mental_state.beliefs.update({
                            "metrics": metrics,
                            "visualization_data": visualization_data,
                            "report": report
                        })
                        self.log("Updated dashboard due to significant change in throughput")
            
            conn.close()
            
        except Exception as e:
            self.log(f"[ERROR] Failed to check for updates: {str(e)}")

    def _check_for_updates_loop(self):
        """Periodically check for updates in shared memory."""
        while True:
            self._check_for_updates()
            self.log("Sleeping for 300 seconds (5 minutes) before next update check")
            time.sleep(10)

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        tickets = input_data.get("tickets", [])
        recommendations = input_data.get("recommendations", [])
        project_id = input_data.get("project_id", "PROJ123")  
        
        self.log(f"Perceiving {len(tickets)} tickets")
        
        self.mental_state.beliefs.update({
            "tickets": tickets,
            "recommendations": recommendations,
            "project_id": project_id
        })

    def _analyze_ticket_data(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.log(f"[DEBUG] Entering _analyze_ticket_data with {len(tickets)} tickets: {tickets}")
        if not tickets:
            self.log("No tickets to analyze")
            return {
                "cycle_time": 0,
                "throughput": 0,
                "workload": {},
                "bottlenecks": {}
            }
        
        try:
            assignee_counts = {}
            status_counts = {}
            done_tickets = []
            
            for idx, ticket in enumerate(tickets):
                self.log(f"[DEBUG] Processing ticket {idx}: {ticket}")
                fields = ticket.get("fields", {})
                current_status = fields.get("status", {}).get("name", "Unknown")
                assignee = fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned"
                
                self.log(f"[DEBUG] Ticket status: {current_status}, Assignee: {assignee}")
                status_counts[current_status] = status_counts.get(current_status, 0) + 1
                if assignee != "Unassigned":
                    assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
                
                if current_status == "Done":
                    done_tickets.append(ticket)
                    self.log(f"[DEBUG] Added to done_tickets: {ticket['key']}")
            
            throughput = len(done_tickets)
            bottlenecks = {status: count for status, count in status_counts.items() if status != "Done"}
            
            self.log(f"[DEBUG] Final metrics - throughput: {throughput}, workload: {assignee_counts}, bottlenecks: {bottlenecks}")
            metrics = {
                "cycle_time": 0,
                "throughput": throughput,
                "workload": assignee_counts,
                "bottlenecks": bottlenecks
            }
            
            return metrics
            
        except Exception as e:
            self.log(f"[ERROR] Failed to analyze ticket data: {str(e)}")
            import traceback
            self.log(f"[ERROR] Traceback: {traceback.format_exc()}")
            return {
                "cycle_time": 0,
                "throughput": 0,
                "workload": {},
                "bottlenecks": {}
            }

    def _create_visualization_data(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        try:
            charts = []
            tables = []
            
            if metrics.get("workload"):
                workload_data = {
                    "type": "bar",
                    "title": "Workload by Assignee",
                    "data": {
                        "labels": list(metrics["workload"].keys()),
                        "datasets": [{
                            "label": "Number of Tickets",
                            "data": list(metrics["workload"].values())
                        }]
                    }
                }
                charts.append(workload_data)
            
            if metrics.get("bottlenecks"):
                bottleneck_data = {
                    "type": "bar",
                    "title": "Tickets by Status",
                    "data": {
                        "labels": list(metrics["bottlenecks"].keys()),
                        "datasets": [{
                            "label": "Number of Tickets",
                            "data": list(metrics["bottlenecks"].values())
                        }]
                    }
                }
                charts.append(bottleneck_data)
            
            performance_table = {
                "title": "Performance Metrics",
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Average Cycle Time (days)", str(metrics.get("cycle_time", 0))],
                    ["Weekly Throughput", str(metrics.get("throughput", 0))]
                ]
            }
            tables.append(performance_table)
            
            visualization_data = {
                "charts": charts,
                "tables": tables
            }
            
            return visualization_data
            
        except Exception as e:
            self.log(f"[ERROR] Failed to create visualization data: {str(e)}")
            return {"charts": [], "tables": []}

    def _generate_report(self, metrics: Dict[str, Any], recommendations: List[str]) -> str:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            insights = []
            
            cycle_time = metrics.get("cycle_time", 0)
            if cycle_time > 5:
                insights.append(f"Average cycle time is high at {cycle_time} days")
            elif cycle_time > 0:
                insights.append(f"Average cycle time is {cycle_time} days")
            
            throughput = metrics.get("throughput", 0)
            if throughput < 3:
                insights.append(f"Weekly throughput is low at {throughput} tickets")
            else:
                insights.append(f"Weekly throughput is {throughput} tickets")
            
            workload = metrics.get("workload", {})
            if workload:
                max_workload = max(workload.items(), key=lambda x: x[1]) if workload else (None, 0)
                if max_workload[0] and max_workload[1] > 3:
                    insights.append(f"{max_workload[0]} has a high workload with {max_workload[1]} tickets")
            
            bottlenecks = metrics.get("bottlenecks", {})
            if bottlenecks:
                max_bottleneck = max(bottlenecks.items(), key=lambda x: x[1]) if bottlenecks else (None, 0)
                if max_bottleneck[0] and max_bottleneck[1] > 3:
                    insights.append(f"Potential bottleneck in '{max_bottleneck[0]}' with {max_bottleneck[1]} tickets")
            
            insights_text = "\n- ".join([""] + insights) if insights else ""
            recommendations_text = "\n- ".join([""] + recommendations[:3]) if recommendations else ""
            
            report = f"""Productivity Dashboard Report ({today}):

Insights:{insights_text}

Recommendations:{recommendations_text}"""
            
            return report
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate report: {str(e)}")
            return f"Productivity Dashboard Report ({datetime.now().strftime('%Y-%m-%d')})\n\nError generating report: {str(e)}"

    def _get_recommendations(self, tickets: List[Dict[str, Any]]) -> List[str]:
        recommendations = self.mental_state.beliefs.get("recommendations", [])
        
        if not recommendations:
            self.log("No recommendations provided, using default recommendations")
            recommendations = [
                "Consider redistributing work to balance team workload",
                "Review tickets in the 'In Progress' state to identify blockers",
                "Schedule sprint retrospective to discuss productivity metrics"
            ]
            
        return recommendations

    def _act(self) -> Dict[str, Any]:
        try:
            tickets = self.mental_state.beliefs.get("tickets", [])
            recommendations = self.mental_state.beliefs.get("recommendations", [])
            
            metrics = self._analyze_ticket_data(tickets)
            self.last_throughput = metrics["throughput"]  
            visualization_data = self._create_visualization_data(metrics)
            report = self._generate_report(metrics, recommendations)
            
            return {
                "metrics": metrics,
                "visualization_data": visualization_data,
                "report": report,
                "recommendations": recommendations,
                "workflow_status": "success"
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to generate productivity dashboard: {str(e)}")
            return {
                "metrics": {},
                "visualization_data": {"charts": [], "tables": []},
                "report": f"Error generating productivity dashboard: {str(e)}",
                "recommendations": [],
                "workflow_status": "failure"
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        status = action_result.get("workflow_status", "failure")
        
        self.mental_state.beliefs["last_dashboard"] = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "metrics_count": len(action_result.get("metrics", {})),
            "recommendation_count": len(action_result.get("recommendations", []))
        }
        
        self.log(f"Dashboard generation completed with status: {status}")