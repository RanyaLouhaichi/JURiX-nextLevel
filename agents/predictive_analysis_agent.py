from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import logging
from agents.base_agent import BaseAgent, AgentCapability
from orchestrator.core.model_manager import ModelManager # type: ignore

class PredictiveAnalysisAgent(BaseAgent):
    OBJECTIVE = "Predict project outcomes, identify risks early, and provide actionable forecasts through collaborative intelligence"

    def __init__(self, redis_client=None):
        super().__init__(name="predictive_analysis_agent", redis_client=redis_client)
        self.model_manager = ModelManager()
        
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,
            AgentCapability.COORDINATE_AGENTS,
            AgentCapability.PROVIDE_RECOMMENDATIONS
        ]
        
        self.mental_state.obligations.extend([
            "predict_sprint_completion",
            "forecast_velocity_trends",
            "identify_future_risks",
            "predict_ticket_completion",
            "generate_early_warnings",
            "collaborate_for_historical_data",
            "share_predictions_with_agents",
            "trigger_preventive_actions"
        ])
        
        self.prediction_thresholds = {
            "high_risk": 0.3,
            "medium_risk": 0.6,
            "low_risk": 0.8,
            "burnout_threshold": 1.5,
            "bottleneck_threshold": 0.7
        }
        
        self.log("PredictiveAnalysisAgent initialized with collaborative forecasting capabilities")

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        tickets = input_data.get("tickets", [])
        metrics = input_data.get("metrics", {})
        historical_data = input_data.get("historical_data", {})
        user_query = input_data.get("user_query", "")
        analysis_type = input_data.get("analysis_type", "comprehensive")
        
        self.log(f"[PERCEPTION] Processing predictive request with {len(tickets)} tickets, analysis type: {analysis_type}")
        
        self.mental_state.add_belief("tickets", tickets, 0.9, "input")
        self.mental_state.add_belief("metrics", metrics, 0.9, "input")
        self.mental_state.add_belief("historical_data", historical_data, 0.8, "input")
        self.mental_state.add_belief("user_query", user_query, 0.9, "input")
        self.mental_state.add_belief("analysis_type", analysis_type, 0.9, "input")
        
        if input_data.get("collaboration_purpose"):
            self.mental_state.add_belief("collaboration_context", input_data.get("collaboration_purpose"), 0.9, "collaboration")
            self.log(f"[COLLABORATION] Operating in collaborative mode: {input_data.get('collaboration_purpose')}")
        
        self._assess_prediction_needs(input_data)

    def _assess_prediction_needs(self, input_data: Dict[str, Any]) -> None:
        tickets = input_data.get("tickets", [])
        historical_data = input_data.get("historical_data", {})
        
        self.log(f"[COLLABORATION ASSESSMENT] Evaluating prediction needs")
        
        if not historical_data or len(historical_data.get("velocity_history", [])) < 3:
            self.log("[COLLABORATION NEED] Insufficient historical data for accurate predictions")
            self.mental_state.request_collaboration(
                agent_type="jira_data_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "need_historical_velocity_data",
                    "minimum_data_points": 5,
                    "time_range": "last_3_months"
                }
            )
        
        if len(tickets) < 5:
            self.log("[COLLABORATION NEED] Limited ticket data for pattern analysis")
            self.mental_state.request_collaboration(
                agent_type="jira_data_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "need_more_ticket_data",
                    "current_count": len(tickets),
                    "minimum_needed": 10
                }
            )

    def _calculate_sprint_completion_probability(self, tickets: List[Dict[str, Any]], 
                                               metrics: Dict[str, Any],
                                               historical_data: Dict[str, Any]) -> Dict[str, Any]:
        self.log("[PREDICTION] Calculating sprint completion probability")
        
        todo_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "To Do"])
        in_progress = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "In Progress"])
        done_tickets = len([t for t in tickets if t.get("fields", {}).get("status", {}).get("name") == "Done"])
        total_tickets = len(tickets)
        
        current_velocity = metrics.get("throughput", 0)
        avg_cycle_time = metrics.get("cycle_time", 5)
        
        remaining_work = todo_tickets + in_progress
        if remaining_work == 0:
            return {
                "probability": 1.0,
                "confidence": 0.95,
                "reasoning": "All tickets are already completed",
                "risk_level": "none"
            }
        
        velocity_history = historical_data.get("velocity_history", [current_velocity])
        if not velocity_history:
            velocity_history = [current_velocity]
        
        avg_velocity = np.mean(velocity_history) if velocity_history else current_velocity
        velocity_std = np.std(velocity_history) if len(velocity_history) > 1 else avg_velocity * 0.2
        
        days_remaining = 10
        expected_completion = avg_velocity * (days_remaining / 7)
        
        if velocity_std > 0:
            z_score = (remaining_work - expected_completion) / velocity_std
            probability = 1 - stats.norm.cdf(z_score)
        else:
            probability = 1.0 if expected_completion >= remaining_work else 0.3
        
        probability = max(0.05, min(0.95, probability))
        
        if probability < self.prediction_thresholds["high_risk"]:
            risk_level = "high"
        elif probability < self.prediction_thresholds["medium_risk"]:
            risk_level = "medium"
        elif probability < self.prediction_thresholds["low_risk"]:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        velocity_trend = self._calculate_velocity_trend(velocity_history)
        
        factors = []
        if velocity_trend < -0.2:
            factors.append("declining velocity trend")
            probability *= 0.85
        elif velocity_trend > 0.2:
            factors.append("improving velocity trend")
            probability *= 1.1
        
        bottleneck_ratio = in_progress / max(total_tickets, 1)
        if bottleneck_ratio > self.prediction_thresholds["bottleneck_threshold"]:
            factors.append("significant work-in-progress bottleneck")
            probability *= 0.9
        
        team_load = self._assess_team_load(tickets)
        if team_load["burnout_risk"]:
            factors.append("team burnout risk detected")
            probability *= 0.8
        
        probability = max(0.05, min(0.95, probability))
        
        return {
            "probability": round(probability, 2),
            "confidence": 0.85 if len(velocity_history) >= 3 else 0.6,
            "reasoning": self._generate_probability_reasoning(probability, remaining_work, avg_velocity, factors),
            "risk_level": risk_level,
            "remaining_work": remaining_work,
            "expected_velocity": round(avg_velocity, 1),
            "contributing_factors": factors,
            "recommended_actions": self._generate_sprint_recommendations(probability, factors, team_load)
        }

    def _calculate_velocity_trend(self, velocity_history: List[float]) -> float:
        if len(velocity_history) < 2:
            return 0.0
        
        x = np.arange(len(velocity_history))
        y = np.array(velocity_history)
        
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            avg_velocity = np.mean(y)
            trend = slope / max(avg_velocity, 1)
            return trend
        return 0.0

    def _assess_team_load(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        assignee_loads = {}
        
        for ticket in tickets:
            assignee_info = ticket.get("fields", {}).get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
                
                if status in ["To Do", "In Progress"]:
                    assignee_loads[assignee] = assignee_loads.get(assignee, 0) + 1
        
        if not assignee_loads:
            return {"burnout_risk": False, "overloaded_members": []}
        
        avg_load = np.mean(list(assignee_loads.values()))
        max_load = max(assignee_loads.values())
        
        overloaded = [name for name, load in assignee_loads.items() 
                     if load > avg_load * self.prediction_thresholds["burnout_threshold"]]
        
        return {
            "burnout_risk": len(overloaded) > 0,
            "overloaded_members": overloaded,
            "load_distribution": assignee_loads,
            "max_load": max_load,
            "avg_load": avg_load
        }

    def _forecast_velocity_trends(self, historical_data: Dict[str, Any], 
                                 current_velocity: float) -> Dict[str, Any]:
        self.log("[PREDICTION] Forecasting velocity trends")
        
        velocity_history = historical_data.get("velocity_history", [current_velocity])
        if not velocity_history:
            velocity_history = [current_velocity]
        
        if len(velocity_history) < 3:
            return {
                "forecast": [current_velocity] * 4,
                "trend": "insufficient_data",
                "confidence": 0.4,
                "insights": "Need more historical data for accurate velocity forecasting"
            }
        
        x = np.arange(len(velocity_history))
        y = np.array(velocity_history)
        
        z = np.polyfit(x, y, min(2, len(x) - 1))
        p = np.poly1d(z)
        
        future_points = 4
        future_x = np.arange(len(velocity_history), len(velocity_history) + future_points)
        forecast = p(future_x)
        
        forecast = np.maximum(forecast, 0)
        
        trend_slope = (forecast[-1] - velocity_history[-1]) / max(velocity_history[-1], 1)
        
        if trend_slope > 0.1:
            trend = "improving"
        elif trend_slope < -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        volatility = np.std(velocity_history) / max(np.mean(velocity_history), 1)
        confidence = max(0.4, min(0.9, 0.9 - volatility))
        
        insights = self._generate_velocity_insights(velocity_history, forecast, trend, volatility)
        
        return {
            "forecast": forecast.tolist(),
            "trend": trend,
            "confidence": round(confidence, 2),
            "insights": insights,
            "volatility": round(volatility, 2),
            "next_week_estimate": round(forecast[0], 1)
        }

    def _identify_future_risks(self, tickets: List[Dict[str, Any]], 
                              metrics: Dict[str, Any],
                              predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.log("[PREDICTION] Identifying future risks")
        
        risks = []
        
        sprint_probability = predictions.get("sprint_completion", {}).get("probability", 1.0)
        if sprint_probability < 0.7:
            risks.append({
                "type": "sprint_failure",
                "severity": "high" if sprint_probability < 0.4 else "medium",
                "probability": round(1 - sprint_probability, 2),
                "description": f"Sprint completion at risk with only {sprint_probability:.0%} success probability",
                "mitigation": "Consider reducing sprint scope or increasing team capacity",
                "timeline": "immediate"
            })
        
        bottlenecks = metrics.get("bottlenecks", {})
        for status, count in bottlenecks.items():
            if count > 5:
                risks.append({
                    "type": "process_bottleneck",
                    "severity": "high" if count > 8 else "medium",
                    "probability": 0.8,
                    "description": f"Critical bottleneck forming in '{status}' with {count} tickets stuck",
                    "mitigation": f"Investigate blockers in {status} status and allocate resources to clear backlog",
                    "timeline": "next_2_days"
                })
        
        team_load = self._assess_team_load(tickets)
        if team_load["burnout_risk"]:
            for member in team_load["overloaded_members"]:
                load = team_load["load_distribution"][member]
                risks.append({
                    "type": "team_burnout",
                    "severity": "high",
                    "probability": 0.7,
                    "description": f"{member} is overloaded with {load} tickets (avg: {team_load['avg_load']:.1f})",
                    "mitigation": f"Redistribute {load - int(team_load['avg_load'])} tickets from {member} to other team members",
                    "timeline": "immediate"
                })
        
        velocity_forecast = predictions.get("velocity_forecast", {})
        if velocity_forecast.get("trend") == "declining":
            risks.append({
                "type": "velocity_decline",
                "severity": "medium",
                "probability": 0.6,
                "description": "Team velocity showing declining trend",
                "mitigation": "Review recent impediments and consider process improvements",
                "timeline": "next_week"
            })
        
        cycle_time = metrics.get("cycle_time", 0)
        if cycle_time > 7:
            risks.append({
                "type": "extended_cycle_time",
                "severity": "medium",
                "probability": 0.7,
                "description": f"Average cycle time is {cycle_time:.1f} days, indicating slow progress",
                "mitigation": "Break down large tickets and implement faster feedback loops",
                "timeline": "ongoing"
            })
        
        return sorted(risks, key=lambda r: (r["severity"] == "high", r["probability"]), reverse=True)

    def _predict_ticket_completion(self, tickets: List[Dict[str, Any]], 
                                  metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.log("[PREDICTION] Predicting ticket completion times")
        
        predictions = []
        avg_cycle_time = metrics.get("cycle_time", 5)
        
        for ticket in tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            if status == "Done":
                continue
            
            ticket_key = ticket.get("key", "Unknown")
            ticket_summary = ticket.get("fields", {}).get("summary", "No summary")
            assignee_info = ticket.get("fields", {}).get("assignee")
            assignee = assignee_info.get("displayName", "Unassigned") if assignee_info else "Unassigned"
            
            if status == "To Do":
                estimated_days = avg_cycle_time * 1.2
            elif status == "In Progress":
                changelog = ticket.get("changelog", {}).get("histories", [])
                days_in_progress = self._calculate_days_in_status(changelog, "In Progress")
                remaining_ratio = max(0.2, 1 - (days_in_progress / max(avg_cycle_time, 1)))
                estimated_days = avg_cycle_time * remaining_ratio
            else:
                estimated_days = avg_cycle_time
            
            complexity_factor = self._estimate_ticket_complexity(ticket)
            estimated_days *= complexity_factor
            
            completion_date = datetime.now() + timedelta(days=estimated_days)
            
            confidence = 0.7
            if status == "In Progress":
                confidence = 0.8
            if assignee == "Unassigned":
                confidence *= 0.7
                estimated_days *= 1.5
            
            predictions.append({
                "ticket_key": ticket_key,
                "summary": ticket_summary[:50] + "..." if len(ticket_summary) > 50 else ticket_summary,
                "status": status,
                "assignee": assignee,
                "estimated_completion": completion_date.strftime("%Y-%m-%d"),
                "days_remaining": round(estimated_days, 1),
                "confidence": round(confidence, 2),
                "factors": self._get_completion_factors(ticket, complexity_factor)
            })
        
        return sorted(predictions, key=lambda p: p["days_remaining"])

    def _calculate_days_in_status(self, changelog: List[Dict[str, Any]], status: str) -> float:
        for i, history in enumerate(reversed(changelog)):
            for item in history.get("items", []):
                if item.get("field") == "status" and item.get("toString") == status:
                    start_time = datetime.fromisoformat(history.get("created", "").replace("Z", "+00:00"))
                    return (datetime.now(start_time.tzinfo) - start_time).days
        return 0

    def _estimate_ticket_complexity(self, ticket: Dict[str, Any]) -> float:
        complexity = 1.0
        
        summary = ticket.get("fields", {}).get("summary", "")
        description = ticket.get("fields", {}).get("description", "")
        
        complex_keywords = ["refactor", "architecture", "migration", "integration", "optimization", "investigation"]
        for keyword in complex_keywords:
            if keyword in summary.lower() or keyword in (description or "").lower():
                complexity *= 1.2
        
        changelog = ticket.get("changelog", {}).get("histories", [])
        if len(changelog) > 10:
            complexity *= 1.3
        
        story_points = ticket.get("fields", {}).get("customfield_10010", 0)
        if story_points > 5:
            complexity *= 1.2
        elif story_points > 8:
            complexity *= 1.5
        
        return min(complexity, 2.0)

    def _get_completion_factors(self, ticket: Dict[str, Any], complexity: float) -> List[str]:
        factors = []
        
        if complexity > 1.3:
            factors.append("high complexity")
        
        assignee_info = ticket.get("fields", {}).get("assignee")
        if not assignee_info:
            factors.append("unassigned")
        
        changelog = ticket.get("changelog", {}).get("histories", [])
        if len(changelog) > 10:
            factors.append("multiple state changes")
        
        status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
        if status == "In Progress":
            days_in_progress = self._calculate_days_in_status(changelog, "In Progress")
            if days_in_progress > 5:
                factors.append("extended time in progress")
        
        return factors

    def _generate_early_warnings(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.log("[PREDICTION] Generating early warnings")
        
        warnings = []
        
        sprint_completion = predictions.get("sprint_completion", {})
        if sprint_completion.get("probability", 1.0) < 0.5:
            warnings.append({
                "type": "sprint_failure_imminent",
                "urgency": "critical",
                "message": f"Sprint completion probability is only {sprint_completion['probability']:.0%}",
                "recommended_action": "Immediate scope reduction or resource reallocation required",
                "trigger_collaboration": ["recommendation_agent", "chat_agent"]
            })
        
        risks = predictions.get("risks", [])
        high_risks = [r for r in risks if r.get("severity") == "high"]
        
        if len(high_risks) >= 2:
            warnings.append({
                "type": "multiple_high_risks",
                "urgency": "high",
                "message": f"{len(high_risks)} high-severity risks detected that could derail the sprint",
                "recommended_action": "Schedule emergency team meeting to address risks",
                "trigger_collaboration": ["recommendation_agent"]
            })
        
        velocity_forecast = predictions.get("velocity_forecast", {})
        if velocity_forecast.get("trend") == "declining" and velocity_forecast.get("next_week_estimate", 0) < 3:
            warnings.append({
                "type": "velocity_collapse",
                "urgency": "high",
                "message": "Velocity trending toward critical levels",
                "recommended_action": "Investigate impediments and consider process intervention",
                "trigger_collaboration": ["productivity_dashboard_agent", "recommendation_agent"]
            })
        
        ticket_predictions = predictions.get("ticket_predictions", [])
        late_tickets = [t for t in ticket_predictions if t.get("days_remaining", 0) > 10]
        
        if len(late_tickets) > 3:
            warnings.append({
                "type": "multiple_delayed_tickets",
                "urgency": "medium",
                "message": f"{len(late_tickets)} tickets predicted to complete late",
                "recommended_action": "Review and reprioritize backlog",
                "trigger_collaboration": ["recommendation_agent"]
            })
        
        return sorted(warnings, key=lambda w: {"critical": 0, "high": 1, "medium": 2}.get(w["urgency"], 3))

    def _generate_probability_reasoning(self, probability: float, remaining_work: int, 
                                      avg_velocity: float, factors: List[str]) -> str:
        base_reasoning = f"With {remaining_work} tickets remaining and average velocity of {avg_velocity:.1f} tickets/week, "
        
        if probability >= 0.8:
            confidence_text = "the team is well-positioned to complete the sprint"
        elif probability >= 0.6:
            confidence_text = "sprint completion is achievable but requires sustained effort"
        elif probability >= 0.4:
            confidence_text = "sprint completion is at risk without intervention"
        else:
            confidence_text = "sprint completion is unlikely without significant changes"
        
        base_reasoning += confidence_text
        
        if factors:
            base_reasoning += f". Key factors: {', '.join(factors)}"
        
        return base_reasoning

    def _generate_sprint_recommendations(self, probability: float, factors: List[str], 
                                       team_load: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if probability < 0.7:
            recommendations.append(f"Consider removing {int((1-probability) * 5)} lower-priority tickets from sprint scope")
        
        if "declining velocity trend" in factors:
            recommendations.append("Investigate recent impediments affecting team velocity")
        
        if "significant work-in-progress bottleneck" in factors:
            recommendations.append("Implement WIP limits and focus on completing in-progress items")
        
        if team_load["burnout_risk"]:
            for member in team_load["overloaded_members"]:
                load = team_load["load_distribution"][member]
                target_member = min(team_load["load_distribution"], key=team_load["load_distribution"].get)
                tickets_to_move = int(load - team_load["avg_load"])
                recommendations.append(
                    f"Reassign {tickets_to_move} tickets from {member} to {target_member} to balance workload"
                )
        
        if probability < 0.5 and not recommendations:
            recommendations.append("Schedule daily check-ins to monitor progress closely")
            recommendations.append("Consider bringing in additional resources or extending timeline")
        
        return recommendations[:3]

    def _generate_velocity_insights(self, history: List[float], forecast: List[float], 
                                   trend: str, volatility: float) -> str:
        avg_historical = np.mean(history)
        avg_forecast = np.mean(forecast)
        
        if trend == "improving":
            insight = f"Velocity is improving from {avg_historical:.1f} to projected {avg_forecast:.1f} tickets/week. "
            insight += "Team efficiency gains are evident."
        elif trend == "declining":
            insight = f"Velocity is declining from {avg_historical:.1f} to projected {avg_forecast:.1f} tickets/week. "
            insight += "Investigation of impediments recommended."
        else:
            insight = f"Velocity remains stable around {avg_historical:.1f} tickets/week. "
            insight += "Consistent team performance observed."
        
        if volatility > 0.3:
            insight += " High volatility suggests unpredictable factors affecting delivery."
        elif volatility < 0.1:
            insight += " Low volatility indicates predictable delivery patterns."
        
        return insight

    def _generate_natural_language_predictions(self, predictions: Dict[str, Any]) -> str:
        prompt_template = f"""You are an AI that explains predictive analytics in natural, conversational language.

Based on these predictions:
- Sprint completion probability: {predictions.get('sprint_completion', {}).get('probability', 0):.0%}
- Velocity trend: {predictions.get('velocity_forecast', {}).get('trend', 'unknown')}
- High-risk items: {len([r for r in predictions.get('risks', []) if r.get('severity') == 'high'])}
- Critical warnings: {len([w for w in predictions.get('warnings', []) if w.get('urgency') == 'critical'])}

Provide a brief, natural language summary of what the team should know. Be specific and actionable.
Focus on the most important insights and what actions would help.
Keep it conversational and under 100 words."""

        try:
            # Use dynamic model selection
            response = self.model_manager.generate_response(
                prompt=prompt_template,
                context={
                    "agent_name": self.name,
                    "task_type": "predictive_summary",
                    "sprint_probability": predictions.get('sprint_completion', {}).get('probability', 0),
                    "risk_count": len(predictions.get('risks', [])),
                    "has_warnings": len(predictions.get('warnings', [])) > 0
                }
            )
            self.log(f"✅ {self.name} received response from model")
            return response.strip()
        except Exception as e:
            self.log(f"[ERROR] Failed to generate natural language predictions: {e}")
            probability = predictions.get('sprint_completion', {}).get('probability', 0)
            return f"Based on current patterns, there's a {probability:.0%} chance of completing the sprint. Focus on clearing bottlenecks and balancing team workload for best results."

    def _act(self) -> Dict[str, Any]:
        try:
            tickets = self.mental_state.get_belief("tickets") or []
            metrics = self.mental_state.get_belief("metrics") or {}
            historical_data = self.mental_state.get_belief("historical_data") or {}
            analysis_type = self.mental_state.get_belief("analysis_type") or "comprehensive"
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            
            self.log(f"[ACTION] Generating {analysis_type} predictive analysis for {len(tickets)} tickets")
            
            if hasattr(self.mental_state, 'add_experience'):
                experience_desc = f"Performing {'collaborative ' if collaboration_context else ''}predictive analysis"
                self.mental_state.add_experience(
                    experience_description=experience_desc,
                    outcome="generating_predictions",
                    confidence=0.8,
                    metadata={
                        "ticket_count": len(tickets),
                        "analysis_type": analysis_type,
                        "collaborative": bool(collaboration_context)
                    }
                )
            
            predictions = {}
            
            if analysis_type in ["comprehensive", "sprint_completion"]:
                predictions["sprint_completion"] = self._calculate_sprint_completion_probability(
                    tickets, metrics, historical_data
                )
            
            if analysis_type in ["comprehensive", "velocity_forecast"]:
                predictions["velocity_forecast"] = self._forecast_velocity_trends(
                    historical_data, metrics.get("throughput", 0)
                )
            
            if analysis_type in ["comprehensive", "risk_assessment"]:
                predictions["risks"] = self._identify_future_risks(tickets, metrics, predictions)
            
            if analysis_type in ["comprehensive", "ticket_predictions"]:
                predictions["ticket_predictions"] = self._predict_ticket_completion(tickets, metrics)
            
            predictions["warnings"] = self._generate_early_warnings(predictions)
            
            predictions["natural_language_summary"] = self._generate_natural_language_predictions(predictions)
            
            collaboration_triggered = False
            for warning in predictions["warnings"]:
                if warning.get("trigger_collaboration"):
                    collaboration_triggered = True
                    for agent in warning["trigger_collaboration"]:
                        self.mental_state.request_collaboration(
                            agent_type=agent,
                            reasoning_type="strategic_reasoning",
                            context={
                                "reason": "predictive_warning_triggered",
                                "warning_type": warning["type"],
                                "urgency": warning["urgency"],
                                "predictions": predictions
                            }
                        )
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Generated comprehensive predictions with {len(predictions['warnings'])} warnings",
                    outcome="predictions_completed",
                    confidence=0.9,
                    metadata={
                        "sprint_probability": predictions.get("sprint_completion", {}).get("probability", 0),
                        "risks_identified": len(predictions.get("risks", [])),
                        "warnings_generated": len(predictions["warnings"]),
                        "collaboration_triggered": collaboration_triggered
                    }
                )
            
            decision = {
                "action": "generate_predictive_analysis",
                "analysis_type": analysis_type,
                "ticket_count": len(tickets),
                "predictions_generated": list(predictions.keys()),
                "collaboration_triggered": collaboration_triggered,
                "confidence": 0.85 if len(historical_data.get("velocity_history", [])) >= 3 else 0.6,
                "reasoning": f"Generated {analysis_type} predictions with {len(predictions.get('warnings', []))} warnings"
            }
            self.mental_state.add_decision(decision)
            
            return {
                "predictions": predictions,
                "workflow_status": "success",
                "collaboration_metadata": {
                    "is_collaborative": bool(collaboration_context),
                    "collaboration_context": collaboration_context,
                    "collaboration_triggered": collaboration_triggered,
                    "warnings_requiring_action": len([w for w in predictions["warnings"] if w.get("trigger_collaboration")])
                }
            }
            
        except Exception as e:
            self.log(f"[ERROR] Failed to generate predictions: {str(e)}")
            
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed predictive analysis",
                    outcome=f"Error: {str(e)}",
                    confidence=0.2,
                    metadata={"error_type": type(e).__name__}
                )
            
            return {
                "predictions": {},
                "workflow_status": "failure",
                "error": str(e),
                "collaboration_metadata": {
                    "is_collaborative": False,
                    "error_occurred": True
                }
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        super()._rethink(action_result)
        
        predictions = action_result.get("predictions", {})
        collaboration_metadata = action_result.get("collaboration_metadata", {})
        
        reflection = {
            "operation": "predictive_analysis",
            "success": action_result.get("workflow_status") == "success",
            "predictions_generated": list(predictions.keys()),
            "warnings_count": len(predictions.get("warnings", [])),
            "risks_identified": len(predictions.get("risks", [])),
            "collaboration_triggered": collaboration_metadata.get("collaboration_triggered", False),
            "sprint_probability": predictions.get("sprint_completion", {}).get("probability", 0),
            "performance_notes": f"Generated predictions with {len(predictions.get('warnings', []))} warnings"
        }
        
        self.mental_state.add_reflection(reflection)
        
        if collaboration_metadata.get("collaboration_triggered"):
            self.mental_state.add_experience(
                experience_description=f"Triggered collaborative response to predictions",
                outcome="collaboration_initiated",
                confidence=0.8,
                metadata={
                    "warning_count": len(predictions.get("warnings", [])),
                    "risk_count": len(predictions.get("risks", []))
                }
            )
        
        self.log(f"[REFLECTION] Predictive analysis completed: {reflection}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)