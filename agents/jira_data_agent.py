from typing import Dict, Any, List
import json
import os
import logging
import redis
from datetime import datetime
from agents.base_agent import BaseAgent, AgentCapability
import time
import threading

class JiraDataAgent(BaseAgent):
    OBJECTIVE = "Retrieve and filter Jira ticket data based on project ID and time range with Redis caching"

    def __init__(self, mock_data_path: str = None, redis_client: redis.Redis = None):
        # Initialize Redis client
        if redis_client is None:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        super().__init__(name="jira_data_agent", redis_client=redis_client)
        
        self.mock_data_path = mock_data_path or os.path.join("data", "mock_jira_data.json")
        self.last_modified = None 
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            self.log("✅ Connected to Redis successfully!")
        except redis.ConnectionError:
            self.log("❌ Redis connection failed")
            raise
        
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA
        ]
        
        self.mental_state.obligations.extend([
            "load_jira_data",
            "filter_tickets",
            "monitor_file",
            "cache_efficiently"
        ])
        
        self.log(f"Initialized JiraDataAgent with mock data path: {self.mock_data_path}")
        
        # Start the monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_file_loop, daemon=True)
        self.monitoring_thread.start()

    def _load_jira_data(self) -> List[Dict[str, Any]]:
        """Load Jira data with Redis caching"""
        try:
            # Check if data is cached and fresh
            cache_key = f"jira_raw_data:{os.path.basename(self.mock_data_path)}"
            
            # Get file modification time
            current_modified = os.path.getmtime(self.mock_data_path)
            cached_modified = self.redis_client.get(f"{cache_key}:modified")
            
            # Use cache if file hasn't changed
            if cached_modified and float(cached_modified) == current_modified:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.log("Using cached Jira data")
                    return json.loads(cached_data)
            
            # Load fresh data
            self.log(f"Loading fresh Jira data from {self.mock_data_path}")
            with open(self.mock_data_path, 'r') as file:
                data = json.load(file)
                issues = data.get("issues", [])
                
                # Cache the data
                self.redis_client.set(cache_key, json.dumps(issues))
                self.redis_client.set(f"{cache_key}:modified", str(current_modified))
                self.redis_client.expire(cache_key, 3600)  # Cache for 1 hour
                self.redis_client.expire(f"{cache_key}:modified", 3600)
                
                self.last_modified = current_modified
                self.log(f"Successfully loaded and cached {len(issues)} tickets")
                
                # Update belief about data freshness
                self.mental_state.add_belief("data_freshness", "fresh", 0.9, "file_load")
                
                return issues
                
        except Exception as e:
            self.log(f"[ERROR] Failed to load Jira data: {str(e)}")
            # Update belief about data availability
            self.mental_state.add_belief("data_availability", "failed", 0.8, "error")
            return []

    def _filter_tickets(self, tickets: List[Dict[str, Any]], project_id: str, time_range: Dict[str, str]) -> List[Dict[str, Any]]:
        """Filter tickets with caching and performance tracking"""
        if not tickets:
            self.log("No tickets to filter")
            return []
        
        try:
            # Create cache key for filtered results
            cache_key = f"filtered_tickets:{project_id}:{time_range['start']}:{time_range['end']}"
            
            # Check cache first
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                self.log("Using cached filtered tickets")
                filtered_tickets = json.loads(cached_result)
                self.mental_state.add_belief("cache_hit", True, 0.9, "filter_operation")
                return filtered_tickets
            
            # Perform filtering
            start_time = datetime.fromisoformat(time_range["start"].replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(time_range["end"].replace("Z", "+00:00"))
            
            self.log(f"Filtering tickets for project {project_id} between {start_time} and {end_time}")
            
            filtered_tickets = []
            for ticket in tickets:
                ticket_project = ticket.get("fields", {}).get("project", {}).get("key", "")
                updated_str = ticket.get("fields", {}).get("updated", "")
                if not updated_str:
                    continue
                    
                updated_time = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                
                if ticket_project == project_id and start_time <= updated_time <= end_time:
                    filtered_tickets.append(ticket)
            
            # Cache the filtered results
            self.redis_client.set(cache_key, json.dumps(filtered_tickets))
            self.redis_client.expire(cache_key, 1800)  # Cache for 30 minutes
            
            self.log(f"Filtered to {len(filtered_tickets)} tickets and cached result")
            
            # Update beliefs about filtering performance
            self.mental_state.add_belief("last_filter_count", len(filtered_tickets), 0.9, "filter_operation")
            self.mental_state.add_belief("cache_hit", False, 0.9, "filter_operation")
            
            return filtered_tickets
            
        except Exception as e:
            self.log(f"[ERROR] Failed to filter tickets: {str(e)}")
            self.mental_state.add_belief("filter_error", str(e), 0.8, "error")
            return []

    def _monitor_file(self, project_id: str, time_range: Dict[str, str]):
        """Monitor file changes and update Redis with real-time notifications"""
        try:
            current_modified = os.path.getmtime(self.mock_data_path)
            
            # Check if file was modified
            if self.last_modified is None or current_modified > self.last_modified:
                self.log(f"File {self.mock_data_path} has been modified at {current_modified}")
                self.last_modified = current_modified
                
                # Load and filter tickets
                all_tickets = self._load_jira_data()
                filtered_tickets = self._filter_tickets(all_tickets, project_id, time_range)
                
                # Store in Redis with metadata
                tickets_key = f"tickets:{project_id}"
                metadata_key = f"tickets_meta:{project_id}"
                
                # Store tickets
                self.redis_client.set(tickets_key, json.dumps(filtered_tickets, default=str))
                self.redis_client.expire(tickets_key, 7200)  # 2 hours
                
                # Store metadata
                metadata = {
                    "last_updated": datetime.now().isoformat(),
                    "ticket_count": len(filtered_tickets),
                    "has_changes": True,
                    "project_id": project_id,
                    "time_range": time_range,
                    "file_modified": current_modified
                }
                self.redis_client.set(metadata_key, json.dumps(metadata))
                self.redis_client.expire(metadata_key, 7200)
                
                # Publish update notification
                update_event = {
                    "event_type": "tickets_updated",
                    "project_id": project_id,
                    "ticket_count": len(filtered_tickets),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id
                }
                self.redis_client.publish("jira_updates", json.dumps(update_event))
                
                self.log(f"Updated Redis with {len(filtered_tickets)} tickets for project {project_id}")
                
                # Update mental state beliefs
                self.mental_state.add_belief("last_update_success", True, 0.9, "file_monitor")
                self.mental_state.add_belief("tickets_cached", len(filtered_tickets), 0.9, "file_monitor")
            
        except Exception as e:
            self.log(f"[ERROR] Failed to monitor file: {str(e)}")
            self.mental_state.add_belief("monitor_error", str(e), 0.8, "error")

    def _monitor_file_loop(self):
        """Run a loop to periodically monitor the file for changes"""
        project_id = "PROJ123"  
        time_range = {
            "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-15T23:59:59Z"
        }
        
        while True:
            try:
                self._monitor_file(project_id, time_range)
                self.log("Sleeping for 10 seconds before next file check")
                time.sleep(10)
            except Exception as e:
                self.log(f"[ERROR] Monitor loop error: {str(e)}")
                time.sleep(30)  # Wait longer if there's an error

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        super()._perceive(input_data)
        
        project_id = input_data.get("project_id", "")
        time_range = input_data.get("time_range", {})
        
        self.log(f"Perceiving request for project {project_id} with time range {time_range}")
        
        # Store perception beliefs
        self.mental_state.add_belief("current_project", project_id, 0.9, "input")
        self.mental_state.add_belief("current_time_range", time_range, 0.9, "input")

    def _act(self) -> Dict[str, Any]:
        """Enhanced action with Redis optimization and performance tracking"""
        try:
            project_id = self.mental_state.get_belief("current_project") or ""
            time_range = self.mental_state.get_belief("current_time_range") or {}
            
            # Try to get from Redis cache first
            tickets_key = f"tickets:{project_id}"
            cached_tickets = self.redis_client.get(tickets_key)
            
            if cached_tickets:
                filtered_tickets = json.loads(cached_tickets)
                self.log(f"Retrieved {len(filtered_tickets)} tickets from Redis cache for project {project_id}")
                
                # Update beliefs about cache performance
                self.mental_state.add_belief("cache_hit_main", True, 0.9, "redis_retrieval")
                
            else:
                # Cache miss - load and filter
                self.log("Cache miss - loading fresh data")
                all_tickets = self._load_jira_data()
                filtered_tickets = self._filter_tickets(all_tickets, project_id, time_range)
                
                self.mental_state.add_belief("cache_hit_main", False, 0.9, "redis_retrieval")
            
            # Store performance metrics
            performance_metrics = {
                "retrieval_timestamp": datetime.now().isoformat(),
                "tickets_retrieved": len(filtered_tickets),
                "cache_hit": bool(cached_tickets),
                "project_id": project_id
            }
            
            metrics_key = f"performance:{self.agent_id}:{datetime.now().strftime('%Y%m%d_%H')}"
            self.redis_client.lpush(metrics_key, json.dumps(performance_metrics))
            self.redis_client.expire(metrics_key, 86400)  # Keep for 24 hours
            
            return {
                "tickets": filtered_tickets,
                "workflow_status": "success",
                "metadata": {
                    "project_id": project_id,
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": bool(cached_tickets),
                    "ticket_count": len(filtered_tickets),
                    "agent_id": self.agent_id
                }
            }
            
        except Exception as e:
            self.log(f"[ERROR] Failed to process Jira data: {str(e)}")
            
            # Update error beliefs
            self.mental_state.add_belief("last_error", str(e), 0.9, "error")
            
            return {
                "tickets": [],
                "workflow_status": "failure",
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id
                }
            }

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        """Enhanced reflection with performance analysis"""
        super()._rethink(action_result)
        
        status = action_result.get("workflow_status", "failure")
        ticket_count = len(action_result.get("tickets", []))
        cache_hit = action_result.get("metadata", {}).get("cache_hit", False)
        
        # Analyze performance and update competencies
        if status == "success":
            if cache_hit:
                self.mental_state.competency_model.add_competency("cache_retrieval", 1.0)
            else:
                self.mental_state.competency_model.add_competency("fresh_data_load", 1.0)
        else:
            self.mental_state.competency_model.add_competency("error_handling", 0.3)
        
        # Store reflection
        reflection = {
            "operation": "jira_data_retrieval",
            "success": status == "success",
            "ticket_count": ticket_count,
            "cache_performance": cache_hit,
            "performance_notes": f"Retrieved {ticket_count} tickets with {'cache hit' if cache_hit else 'fresh load'}"
        }
        self.mental_state.add_reflection(reflection)
        
        self.log(f"Retrieval completed with status: {status}, found {ticket_count} tickets (cache: {cache_hit})")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from Redis"""
        try:
            metrics_key = f"performance:{self.agent_id}:{datetime.now().strftime('%Y%m%d_%H')}"
            raw_metrics = self.redis_client.lrange(metrics_key, 0, -1)
            
            metrics = []
            for raw_metric in raw_metrics:
                metrics.append(json.loads(raw_metric))
            
            return {
                "total_retrievals": len(metrics),
                "cache_hit_rate": sum(1 for m in metrics if m.get("cache_hit")) / len(metrics) if metrics else 0,
                "avg_tickets_per_retrieval": sum(m.get("tickets_retrieved", 0) for m in metrics) / len(metrics) if metrics else 0,
                "recent_metrics": metrics[-10:]  # Last 10 operations
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to get performance metrics: {str(e)}")
            return {}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(input_data)