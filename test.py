# In Python console:
import asyncio
from orchestrator.core.collaborative_framework import CollaborativeFramework # type: ignore
from orchestrator.memory.shared_memory import JurixSharedMemory # type: ignore
from agents.jira_data_agent import JiraDataAgent

shared_memory = JurixSharedMemory()
jira_agent = JiraDataAgent(redis_client=shared_memory.redis_client)
framework = CollaborativeFramework(shared_memory.redis_client, {"jira_data_agent": jira_agent})

async def test():
    result = await framework.coordinate_agents("jira_data_agent", {"project_id": "PROJ123"})
    print("Direct result keys:", list(result.keys()))
    return result

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
direct_result = loop.run_until_complete(test())
loop.close()