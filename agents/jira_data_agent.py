# agents/jira_data_agent.py
# Enhanced JiraDataAgent that intelligently collaborates to provide richer data context
# This agent now recognizes when raw data retrieval should trigger additional analysis
# and can coordinate with analytics agents to provide comprehensive insights

from typing import Dict, Any, List
import json
import os
import logging
import redis
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent, AgentCapability
import time
import threading

class JiraDataAgent(BaseAgent):
    OBJECTIVE = "Retrieve and intelligently analyze Jira ticket data, coordinating with other agents to provide comprehensive project insights"

    def __init__(self, mock_data_path: str = None, redis_client: redis.Redis = None):
        # Initialize Redis client for collaborative capabilities
        if redis_client is None:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        super().__init__(name="jira_data_agent", redis_client=redis_client)
        
        self.mock_data_path = mock_data_path or os.path.join("data", "mock_jira_data.json")
        self.last_modified = None 
        
        # Test Redis connection for collaborative features
        try:
            self.redis_client.ping()
            self.log("✅ Connected to Redis successfully for collaborative data operations!")
        except redis.ConnectionError:
            self.log("❌ Redis connection failed - limited collaborative capabilities")
            raise
        
        # Enhanced capabilities including collaboration and analysis
        self.mental_state.capabilities = [
            AgentCapability.RETRIEVE_DATA,
            AgentCapability.RANK_CONTENT,     # Enhanced: can rank data quality
            AgentCapability.COORDINATE_AGENTS  # New: can coordinate with analytics agents
        ]
        
        # Enhanced obligations for intelligent data operations
        self.mental_state.obligations.extend([
            "load_jira_data",
            "filter_tickets",
            "monitor_file",
            "cache_efficiently",
            "assess_data_completeness",    # New: evaluate if data is sufficient
            "trigger_analysis_collaboration", # New: request analytical help when appropriate
            "provide_data_insights"        # New: offer intelligent data summaries
        ])
        
        self.log(f"Enhanced JiraDataAgent initialized with collaborative intelligence at: {self.mock_data_path}")
        
        # Start the monitoring thread for real-time collaborative updates
        self.monitoring_thread = threading.Thread(target=self._monitor_file_loop, daemon=True)
        self.monitoring_thread.start()

    def _load_jira_data(self) -> List[Dict[str, Any]]:
        """Enhanced data loading with quality assessment and collaborative triggers"""
        try:
            # Check if data is cached and fresh
            cache_key = f"jira_raw_data:{os.path.basename(self.mock_data_path)}"
            
            # Get file modification time for freshness validation
            current_modified = os.path.getmtime(self.mock_data_path)
            cached_modified = self.redis_client.get(f"{cache_key}:modified")
            
            # Use cache if file hasn't changed and assess cache quality
            if cached_modified and float(cached_modified) == current_modified:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.log("Using cached Jira data - assessing quality for collaborative needs")
                    data = json.loads(cached_data)
                    
                    # NEW: Assess if cached data triggers collaboration needs
                    self._assess_data_collaboration_needs(data, "cached_data")
                    return data
            
            # Load fresh data with enhanced intelligence
            self.log(f"Loading fresh Jira data from {self.mock_data_path}")
            with open(self.mock_data_path, 'r') as file:
                data = json.load(file)
                issues = data.get("issues", [])
                
                # Cache the data with collaborative metadata
                cache_metadata = {
                    "loaded_at": datetime.now().isoformat(),
                    "issue_count": len(issues),
                    "data_quality_assessed": True
                }
                
                self.redis_client.set(cache_key, json.dumps(issues))
                self.redis_client.set(f"{cache_key}:modified", str(current_modified))
                self.redis_client.set(f"{cache_key}:metadata", json.dumps(cache_metadata))
                self.redis_client.expire(cache_key, 3600)  # Cache for 1 hour
                self.redis_client.expire(f"{cache_key}:modified", 3600)
                self.redis_client.expire(f"{cache_key}:metadata", 3600)
                
                self.last_modified = current_modified
                self.log(f"Successfully loaded and cached {len(issues)} tickets with quality assessment")
                
                # Store data freshness belief with high confidence
                self.mental_state.add_belief("data_freshness", "fresh", 0.9, "file_load")
                
                # NEW: Assess if fresh data suggests collaboration opportunities
                self._assess_data_collaboration_needs(issues, "fresh_data")
                
                return issues
                
        except Exception as e:
            self.log(f"[ERROR] Failed to load Jira data: {str(e)}")
            # Update belief about data availability
            self.mental_state.add_belief("data_availability", "failed", 0.8, "error")
            return []

    def _assess_data_collaboration_needs(self, issues: List[Dict[str, Any]], data_source: str):
        """
        Intelligent assessment of whether this data retrieval should trigger collaborative analysis
        This transforms data retrieval from passive to intelligent and proactive
        """
        if not issues:
            return
            
        # Analyze data characteristics to determine collaboration value
        data_characteristics = self._analyze_data_characteristics(issues)
        
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing {len(issues)} tickets from {data_source}")
        
        # If we have rich productivity data, suggest dashboard collaboration
        if data_characteristics["suggests_productivity_analysis"]:
            self.log("[COLLABORATION OPPORTUNITY] Data suggests productivity analysis would be valuable")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "rich_productivity_data_available",
                    "ticket_count": len(issues),
                    "status_diversity": data_characteristics["status_diversity"],
                    "has_cycle_time_data": data_characteristics["has_cycle_time_data"],
                    "data_source": data_source
                }
            )
        
        # If we have data that could enhance recommendations, suggest collaboration
        if data_characteristics["suggests_recommendation_enhancement"]:
            self.log("[COLLABORATION OPPORTUNITY] Data could enhance recommendation quality")
            self.mental_state.request_collaboration(
                agent_type="recommendation_agent",
                reasoning_type="context_enrichment",
                context={
                    "reason": "data_can_enhance_recommendations",
                    "resolution_patterns": data_characteristics["resolution_patterns"],
                    "team_activity": data_characteristics["team_activity"]
                }
            )

    def _analyze_data_characteristics(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data to understand what collaborative opportunities it presents
        This intelligence helps the agent understand the value and potential of its data
        """
        characteristics = {
            "status_diversity": 0,
            "has_cycle_time_data": False,
            "resolution_patterns": [],
            "team_activity": {},
            "suggests_productivity_analysis": False,
            "suggests_recommendation_enhancement": False
        }
        
        if not issues:
            return characteristics
        
        # Analyze status distribution
        statuses = set()
        assignees = {}
        resolved_tickets = []
        
        for issue in issues:
            fields = issue.get("fields", {})
            
            # Status analysis
            status = fields.get("status", {}).get("name", "Unknown")
            statuses.add(status)
            
            # Team activity analysis
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unknown")
                assignees[assignee] = assignees.get(assignee, 0) + 1
            
            # Resolution pattern analysis
            if status == "Done" and fields.get("resolutiondate"):
                resolved_tickets.append(issue)
            
            # Cycle time data availability
            changelog = issue.get("changelog", {}).get("histories", [])
            if changelog:
                characteristics["has_cycle_time_data"] = True
        
        # Calculate insights
        characteristics["status_diversity"] = len(statuses)
        characteristics["team_activity"] = assignees
        characteristics["resolution_patterns"] = [
            {"resolved_count": len(resolved_tickets), "total_tickets": len(issues)}
        ]
        
        # Determine collaboration suggestions
        # Suggest productivity analysis if we have diverse statuses and cycle time data
        characteristics["suggests_productivity_analysis"] = (
            characteristics["status_diversity"] >= 3 and
            characteristics["has_cycle_time_data"] and
            len(issues) >= 5
        )
        
        # Suggest recommendation enhancement if we have team activity and resolution patterns
        characteristics["suggests_recommendation_enhancement"] = (
            len(assignees) >= 2 and
            len(resolved_tickets) >= 3
        )
        
        return characteristics

    def _filter_tickets(self, tickets: List[Dict[str, Any]], project_id: str, time_range: Dict[str, str]) -> List[Dict[str, Any]]:
        """Enhanced filtering with intelligent caching and collaboration triggers"""
        if not tickets:
            self.log("No tickets to filter")
            return []
        
        try:
            # Create cache key for filtered results
            cache_key = f"filtered_tickets:{project_id}:{time_range['start']}:{time_range['end']}"
            
            # Check cache first with collaboration awareness
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                self.log("Using cached filtered tickets - checking for collaboration opportunities")
                filtered_tickets = json.loads(cached_result)
                self.mental_state.add_belief("cache_hit", True, 0.9, "filter_operation")
                
                # Even with cached data, assess collaboration needs
                self._assess_filtered_data_collaboration(filtered_tickets, project_id, "cached")
                return filtered_tickets
            
            # Perform intelligent filtering
            start_time = datetime.fromisoformat(time_range["start"].replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(time_range["end"].replace("Z", "+00:00"))
            
            self.log(f"Intelligently filtering tickets for project {project_id} between {start_time} and {end_time}")
            
            filtered_tickets = []
            for ticket in tickets:
                ticket_project = ticket.get("fields", {}).get("project", {}).get("key", "")
                updated_str = ticket.get("fields", {}).get("updated", "")
                if not updated_str:
                    continue
                    
                updated_time = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                
                if ticket_project == project_id and start_time <= updated_time <= end_time:
                    filtered_tickets.append(ticket)
            
            # Cache the filtered results with metadata
            filter_metadata = {
                "filtered_at": datetime.now().isoformat(),
                "source_count": len(tickets),
                "filtered_count": len(filtered_tickets),
                "project_id": project_id
            }
            
            self.redis_client.set(cache_key, json.dumps(filtered_tickets))
            self.redis_client.set(f"{cache_key}:metadata", json.dumps(filter_metadata))
            self.redis_client.expire(cache_key, 1800)  # Cache for 30 minutes
            self.redis_client.expire(f"{cache_key}:metadata", 1800)
            
            self.log(f"Filtered to {len(filtered_tickets)} tickets and cached result with metadata")
            
            # Update beliefs about filtering performance
            self.mental_state.add_belief("last_filter_count", len(filtered_tickets), 0.9, "filter_operation")
            self.mental_state.add_belief("cache_hit", False, 0.9, "filter_operation")
            
            # Assess collaboration opportunities with fresh filtered data
            self._assess_filtered_data_collaboration(filtered_tickets, project_id, "fresh")
            
            return filtered_tickets
            
        except Exception as e:
            self.log(f"[ERROR] Failed to filter tickets: {str(e)}")
            self.mental_state.add_belief("filter_error", str(e), 0.8, "error")
            return []

    def _assess_filtered_data_collaboration(self, filtered_tickets: List[Dict[str, Any]], 
                                          project_id: str, data_source: str):
        """
        Assess collaboration opportunities specific to filtered project data
        This allows the agent to be proactive about offering enhanced analysis
        """
        if not filtered_tickets:
            return
        
        ticket_count = len(filtered_tickets)
        self.log(f"[COLLABORATION ASSESSMENT] Analyzing {ticket_count} filtered tickets for {project_id}")
        
        # If we have substantial project data, suggest comprehensive analysis
        if ticket_count >= 10:
            self.log("[COLLABORATION OPPORTUNITY] Substantial project data - suggesting comprehensive analysis")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="comprehensive_analysis",
                context={
                    "reason": "substantial_project_data",
                    "project_id": project_id,
                    "ticket_count": ticket_count,
                    "analysis_type": "comprehensive",
                    "data_source": data_source
                }
            )
        
        # Analyze ticket patterns for recommendation opportunities
        patterns = self._identify_ticket_patterns(filtered_tickets)
        if patterns["has_interesting_patterns"]:
            self.log("[COLLABORATION OPPORTUNITY] Interesting patterns detected - could enhance recommendations")
            self.mental_state.request_collaboration(
                agent_type="recommendation_agent",
                reasoning_type="pattern_analysis",
                context={
                    "reason": "interesting_patterns_detected",
                    "project_id": project_id,
                    "patterns": patterns,
                    "pattern_confidence": patterns["confidence"]
                }
            )

    def _identify_ticket_patterns(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify interesting patterns in ticket data that could benefit from collaboration
        This intelligence helps the agent understand when its data has collaborative value
        """
        patterns = {
            "has_interesting_patterns": False,
            "confidence": 0.0,
            "pattern_types": []
        }
        
        if len(tickets) < 3:
            return patterns
        
        # Analyze for bottleneck patterns
        status_counts = {}
        for ticket in tickets:
            status = ticket.get("fields", {}).get("status", {}).get("name", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Detect bottleneck pattern
        in_progress = status_counts.get("In Progress", 0)
        done = status_counts.get("Done", 0)
        
        if in_progress > done and in_progress > 3:
            patterns["pattern_types"].append("bottleneck_detected")
            patterns["confidence"] += 0.3
        
        # Analyze for velocity patterns
        resolved_recently = 0
        for ticket in tickets:
            resolution_date = ticket.get("fields", {}).get("resolutiondate")
            if resolution_date:
                try:
                    resolved_date = datetime.fromisoformat(resolution_date.replace("Z", "+00:00"))
                    days_ago = (datetime.now(resolved_date.tzinfo) - resolved_date).days
                    if days_ago <= 7:  # Resolved in last week
                        resolved_recently += 1
                except:
                    pass
        
        if resolved_recently >= 3:
            patterns["pattern_types"].append("high_velocity")
            patterns["confidence"] += 0.4
        
        # Analyze for team distribution patterns
        assignees = set()
        for ticket in tickets:
            assignee_info = ticket.get("fields", {}).get("assignee")
            if assignee_info:
                assignees.add(assignee_info.get("displayName", "Unknown"))
        
        if len(assignees) >= 3:
            patterns["pattern_types"].append("distributed_team")
            patterns["confidence"] += 0.2
        
        patterns["has_interesting_patterns"] = patterns["confidence"] >= 0.3
        return patterns

    def _monitor_file(self, project_id: str, time_range: Dict[str, str]):
        """Enhanced file monitoring with collaborative intelligence and real-time notifications"""
        try:
            current_modified = os.path.getmtime(self.mock_data_path)
            
            # Check if file was modified
            if self.last_modified is None or current_modified > self.last_modified:
                self.log(f"File {self.mock_data_path} has been modified at {current_modified}")
                self.last_modified = current_modified
                
                # Load and filter tickets with collaborative assessment
                all_tickets = self._load_jira_data()
                filtered_tickets = self._filter_tickets(all_tickets, project_id, time_range)
                
                # Store in Redis with enhanced metadata
                tickets_key = f"tickets:{project_id}"
                metadata_key = f"tickets_meta:{project_id}"
                
                # Store tickets with collaborative context
                self.redis_client.set(tickets_key, json.dumps(filtered_tickets, default=str))
                self.redis_client.expire(tickets_key, 7200)  # 2 hours
                
                # Store enhanced metadata
                metadata = {
                    "last_updated": datetime.now().isoformat(),
                    "ticket_count": len(filtered_tickets),
                    "has_changes": True,
                    "project_id": project_id,
                    "time_range": time_range,
                    "file_modified": current_modified,
                    "collaborative_opportunities_assessed": True,
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests)
                }
                self.redis_client.set(metadata_key, json.dumps(metadata))
                self.redis_client.expire(metadata_key, 7200)
                
                # Publish enhanced update notification
                update_event = {
                    "event_type": "tickets_updated",
                    "project_id": project_id,
                    "ticket_count": len(filtered_tickets),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id,
                    "collaborative_assessment_done": True,
                    "collaboration_opportunities": len([req for req in self.mental_state.collaborative_requests if req.get("timestamp", "") > datetime.now().replace(hour=0, minute=0, second=0).isoformat()])
                }
                self.redis_client.publish("jira_updates", json.dumps(update_event))
                
                self.log(f"Updated Redis with {len(filtered_tickets)} tickets for project {project_id} (collaborative intelligence applied)")
                
                # Update mental state beliefs with collaborative context
                self.mental_state.add_belief("last_update_success", True, 0.9, "file_monitor")
                self.mental_state.add_belief("tickets_cached", len(filtered_tickets), 0.9, "file_monitor")
                self.mental_state.add_belief("collaborative_assessment_complete", True, 0.9, "intelligence")
            
        except Exception as e:
            self.log(f"[ERROR] Failed to monitor file: {str(e)}")
            self.mental_state.add_belief("monitor_error", str(e), 0.8, "error")

    def _monitor_file_loop(self):
        """Enhanced monitoring loop with collaborative intelligence"""
        project_id = "PROJ123"  
        time_range = {
            "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-15T23:59:59Z"
        }
        
        while True:
            try:
                self._monitor_file(project_id, time_range)
                self.log("Intelligent monitoring cycle complete - sleeping for 10 seconds")
                time.sleep(10)
            except Exception as e:
                self.log(f"[ERROR] Monitor loop error: {str(e)}")
                time.sleep(30)  # Wait longer if there's an error

    def _perceive(self, input_data: Dict[str, Any]) -> None:
        """Enhanced perception with collaborative context awareness"""
        super()._perceive(input_data)
        
        project_id = input_data.get("project_id", "")
        time_range = input_data.get("time_range", {})
        analysis_depth = input_data.get("analysis_depth", "basic")
        
        self.log(f"[PERCEPTION] Processing request for project {project_id} with {analysis_depth} analysis depth")
        
        # Store enhanced perception beliefs
        self.mental_state.add_belief("current_project", project_id, 0.9, "input")
        self.mental_state.add_belief("current_time_range", time_range, 0.9, "input")
        self.mental_state.add_belief("requested_analysis_depth", analysis_depth, 0.9, "input")
        
        # NEW: Handle collaborative context
        if input_data.get("collaboration_purpose"):
            self.mental_state.add_belief("collaboration_context", input_data.get("collaboration_purpose"), 0.9, "collaboration")
            self.log(f"[COLLABORATION] Operating in collaborative mode: {input_data.get('collaboration_purpose')}")
        
        # NEW: Assess collaboration needs based on request characteristics
        self._assess_request_collaboration_needs(input_data)

    def _assess_request_collaboration_needs(self, input_data: Dict[str, Any]) -> None:
        """
        Assess if this specific data request suggests collaboration opportunities
        This proactive intelligence helps anticipate what other agents might need
        """
        analysis_depth = input_data.get("analysis_depth", "basic")
        project_id = input_data.get("project_id")
        collaboration_purpose = input_data.get("collaboration_purpose")
        
        self.log(f"[COLLABORATION ASSESSMENT] Evaluating request for {project_id} with {analysis_depth} depth")
        
        # If enhanced analysis is requested, suggest productivity analysis collaboration
        if analysis_depth == "enhanced":
            self.log("[COLLABORATION OPPORTUNITY] Enhanced analysis requested - suggesting productivity collaboration")
            self.mental_state.request_collaboration(
                agent_type="productivity_dashboard_agent",
                reasoning_type="data_analysis",
                context={
                    "reason": "enhanced_analysis_requested", 
                    "project": project_id,
                    "analysis_type": "comprehensive"
                }
            )
        
        # If this is already a collaborative request, optimize for the requesting agent's needs
        if collaboration_purpose:
            if "recommendation" in collaboration_purpose.lower():
                self.log("[COLLABORATION OPTIMIZATION] Optimizing data retrieval for recommendation context")
                self.mental_state.add_belief("optimize_for_recommendations", True, 0.9, "collaboration")
            elif "analysis" in collaboration_purpose.lower():
                self.log("[COLLABORATION OPTIMIZATION] Optimizing data retrieval for analysis context")
                self.mental_state.add_belief("optimize_for_analysis", True, 0.9, "collaboration")

    def _act(self) -> Dict[str, Any]:
        """Enhanced action method with intelligent collaboration and comprehensive data provision"""
        try:
            project_id = self.mental_state.get_belief("current_project") or ""
            time_range = self.mental_state.get_belief("current_time_range") or {}
            analysis_depth = self.mental_state.get_belief("requested_analysis_depth") or "basic"
            collaboration_context = self.mental_state.get_belief("collaboration_context")
            
            # Try to get from Redis cache first with collaborative intelligence
            tickets_key = f"tickets:{project_id}"
            cached_tickets = self.redis_client.get(tickets_key)
            
            if cached_tickets:
                filtered_tickets = json.loads(cached_tickets)
                self.log(f"Retrieved {len(filtered_tickets)} tickets from Redis cache for project {project_id}")
                
                # Update beliefs about cache performance
                self.mental_state.add_belief("cache_hit_main", True, 0.9, "redis_retrieval")
                
                # Even with cached data, apply collaborative intelligence
                if analysis_depth == "enhanced" or collaboration_context:
                    self._provide_enhanced_data_context(filtered_tickets, project_id)
                
            else:
                # Cache miss - load and filter with intelligence
                self.log("Cache miss - loading fresh data with collaborative assessment")
                all_tickets = self._load_jira_data()
                filtered_tickets = self._filter_tickets(all_tickets, project_id, time_range)
                
                self.mental_state.add_belief("cache_hit_main", False, 0.9, "redis_retrieval")
            
            # Generate enhanced metadata with collaborative insights
            enhanced_metadata = self._generate_enhanced_metadata(filtered_tickets, project_id, collaboration_context)
            
            # Store performance metrics with collaborative context
            performance_metrics = {
                "retrieval_timestamp": datetime.now().isoformat(),
                "tickets_retrieved": len(filtered_tickets),
                "cache_hit": bool(cached_tickets),
                "project_id": project_id,
                "analysis_depth": analysis_depth,
                "collaborative": bool(collaboration_context),
                "collaboration_opportunities_identified": len(self.mental_state.collaborative_requests)
            }
            
            metrics_key = f"performance:{self.agent_id}:{datetime.now().strftime('%Y%m%d_%H')}"
            self.redis_client.lpush(metrics_key, json.dumps(performance_metrics))
            self.redis_client.expire(metrics_key, 86400)  # Keep for 24 hours
            
            # Store successful data retrieval as experience
            if hasattr(self.mental_state, 'add_experience'):
                experience_description = f"Retrieved {len(filtered_tickets)} tickets for {project_id}"
                if collaboration_context:
                    experience_description += f" (collaborative context: {collaboration_context})"
                
                self.mental_state.add_experience(
                    experience_description=experience_description,
                    outcome=f"successful_retrieval_with_{'cache_hit' if cached_tickets else 'fresh_load'}",
                    confidence=0.9,
                    metadata={
                        "project_id": project_id,
                        "ticket_count": len(filtered_tickets),
                        "cache_hit": bool(cached_tickets),
                        "collaborative": bool(collaboration_context),
                        "analysis_depth": analysis_depth
                    }
                )
            
            return {
                "tickets": filtered_tickets,
                "workflow_status": "success",
                "metadata": enhanced_metadata,
                "collaboration_insights": {
                    "collaboration_requests_made": len(self.mental_state.collaborative_requests),
                    "data_quality_assessed": True,
                    "collaborative_opportunities_identified": len([req for req in self.mental_state.collaborative_requests if req.get("timestamp", "") > datetime.now().replace(hour=0, minute=0, second=0).isoformat()]),
                    "analysis_depth_provided": analysis_depth
                }
            }
            
        except Exception as e:
            self.log(f"[ERROR] Failed to process Jira data: {str(e)}")
            
            # Store error as experience
            if hasattr(self.mental_state, 'add_experience'):
                self.mental_state.add_experience(
                    experience_description=f"Failed to retrieve data for {project_id}",
                    outcome=f"Error: {str(e)}",
                    confidence=0.3,
                    metadata={"error_type": type(e).__name__}
                )
            
            # Update error beliefs
            self.mental_state.add_belief("last_error", str(e), 0.9, "error")
            
            return {
                "tickets": [],
                "workflow_status": "failure",
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id
                },
                "collaboration_insights": {
                    "error_occurred": True,
                    "collaboration_requests_made": 0
                }
            }

    def _provide_enhanced_data_context(self, tickets: List[Dict[str, Any]], project_id: str):
        """
        Provide enhanced data context when operating in collaborative or enhanced analysis mode
        This demonstrates how the agent becomes more intelligent about data presentation
        """
        if not tickets:
            return
        
        # Generate intelligent data summary
        data_summary = {
            "total_tickets": len(tickets),
            "status_distribution": {},
            "team_distribution": {},
            "recent_activity": 0,
            "completion_rate": 0
        }
        
        # Analyze ticket characteristics
        completed_tickets = 0
        recent_updates = 0
        one_week_ago = datetime.now() - timedelta(days=7)
        
        for ticket in tickets:
            fields = ticket.get("fields", {})
            
            # Status analysis
            status = fields.get("status", {}).get("name", "Unknown")
            data_summary["status_distribution"][status] = data_summary["status_distribution"].get(status, 0) + 1
            
            if status == "Done":
                completed_tickets += 1
            
            # Team analysis
            assignee_info = fields.get("assignee")
            if assignee_info:
                assignee = assignee_info.get("displayName", "Unassigned")
                data_summary["team_distribution"][assignee] = data_summary["team_distribution"].get(assignee, 0) + 1
            
            # Recent activity analysis
            updated_str = fields.get("updated", "")
            if updated_str:
                try:
                    updated_time = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    if updated_time.replace(tzinfo=None) > one_week_ago:
                        recent_updates += 1
                except:
                    pass
        
        data_summary["recent_activity"] = recent_updates
        data_summary["completion_rate"] = completed_tickets / len(tickets) if tickets else 0
        
        # Store enhanced context as belief
        self.mental_state.add_belief("enhanced_data_context", data_summary, 0.9, "intelligence")
        self.log(f"[INTELLIGENCE] Generated enhanced data context: {data_summary}")

    def _generate_enhanced_metadata(self, tickets: List[Dict[str, Any]], project_id: str, 
                                  collaboration_context: str) -> Dict[str, Any]:
        """Generate comprehensive metadata that provides value to collaborating agents"""
        metadata = {
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": self.mental_state.get_belief("cache_hit_main"),
            "ticket_count": len(tickets),
            "agent_id": self.agent_id,
            "data_intelligence_applied": True
        }
        
        # Add collaborative context
        if collaboration_context:
            metadata["collaboration_context"] = collaboration_context
            metadata["optimized_for_collaboration"] = True
        
        # Add enhanced data context if available
        enhanced_context = self.mental_state.get_belief("enhanced_data_context")
        if enhanced_context:
            metadata["data_analysis"] = enhanced_context
        
        # Add collaboration opportunities identified
        collaboration_requests = self.mental_state.collaborative_requests
        if collaboration_requests:
            metadata["collaboration_opportunities"] = [
                {
                    "agent_type": req.get("agent_type"),
                    "reason": req.get("context", {}).get("reason"),
                    "timestamp": req.get("timestamp")
                }
                for req in collaboration_requests[-5:]  # Last 5 requests
            ]
        
        return metadata

    def _rethink(self, action_result: Dict[str, Any]) -> None:
        """Enhanced reflection with collaborative intelligence and performance analysis"""
        super()._rethink(action_result)
        
        status = action_result.get("workflow_status", "failure")
        ticket_count = len(action_result.get("tickets", []))
        cache_hit = action_result.get("metadata", {}).get("cache_hit", False)
        collaboration_insights = action_result.get("collaboration_insights", {})
        
        # Analyze collaborative performance
        collaborative_interaction = collaboration_insights.get("collaborative_opportunities_identified", 0) > 0
        
        # Update competencies with collaborative context
        if status == "success":
            if cache_hit:
                self.mental_state.competency_model.add_competency("collaborative_cache_retrieval", 1.0)
            else:
                self.mental_state.competency_model.add_competency("collaborative_fresh_data_load", 1.0)
            
            if collaborative_interaction:
                self.mental_state.competency_model.add_competency("collaborative_intelligence", 1.0)
        else:
            self.mental_state.competency_model.add_competency("error_handling", 0.3)
        
        # Store enhanced reflection with collaborative analysis
        reflection = {
            "operation": "intelligent_jira_data_retrieval",
            "success": status == "success",
            "ticket_count": ticket_count,
            "cache_performance": cache_hit,
            "collaborative_interaction": collaborative_interaction,
            "collaboration_opportunities_identified": collaboration_insights.get("collaborative_opportunities_identified", 0),
            "data_intelligence_applied": True,
            "performance_notes": f"Retrieved {ticket_count} tickets with {'cache hit' if cache_hit else 'fresh load'} and {'collaborative intelligence' if collaborative_interaction else 'standard processing'}"
        }
        self.mental_state.add_reflection(reflection)
        
        # Learn from collaborative outcomes
        if collaborative_interaction:
            self.mental_state.add_experience(
                experience_description=f"Applied collaborative intelligence during data retrieval",
                outcome=f"identified_{collaboration_insights.get('collaborative_opportunities_identified', 0)}_collaboration_opportunities",
                confidence=0.8,
                metadata={
                    "collaboration_success": status == "success",
                    "data_quality": "high" if ticket_count >= 5 else "limited"
                }
            )
        
        self.log(f"[REFLECTION] Intelligent retrieval completed: {reflection}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Enhanced performance metrics including collaborative intelligence indicators"""
        try:
            metrics_key = f"performance:{self.agent_id}:{datetime.now().strftime('%Y%m%d_%H')}"
            raw_metrics = self.redis_client.lrange(metrics_key, 0, -1)
            
            metrics = []
            collaborative_interactions = 0
            
            for raw_metric in raw_metrics:
                metric_data = json.loads(raw_metric)
                metrics.append(metric_data)
                
                if metric_data.get("collaborative"):
                    collaborative_interactions += 1
            
            if not metrics:
                return {}
            
            return {
                "total_retrievals": len(metrics),
                "cache_hit_rate": sum(1 for m in metrics if m.get("cache_hit")) / len(metrics),
                "avg_tickets_per_retrieval": sum(m.get("tickets_retrieved", 0) for m in metrics) / len(metrics),
                "collaborative_interaction_rate": collaborative_interactions / len(metrics),
                "intelligence_features_active": True,
                "recent_metrics": metrics[-10:]  # Last 10 operations
            }
        except Exception as e:
            self.log(f"[ERROR] Failed to get performance metrics: {str(e)}")
            return {}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for intelligent, collaborative data retrieval"""
        return self.process(input_data)