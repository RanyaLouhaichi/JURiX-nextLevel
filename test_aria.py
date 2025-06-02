# test_aria.py
import requests
import json

# Test ARIA introduction
print("🤖 Testing ARIA Introduction...")
response = requests.get("http://localhost:5000/aria/introduce")
print(json.dumps(response.json(), indent=2))

# Test ARIA chat
print("\n💬 Testing ARIA Chat...")
chat_response = requests.post(
    "http://localhost:5000/aria/chat",
    json={
        "message": "Show me the project dashboard",
        "workspace": "jira",
        "project_id": "PROJ123"
    }
)
print(json.dumps(chat_response.json(), indent=2))

# Test ticket resolution
print("\n✅ Testing Ticket Resolution...")
ticket_response = requests.post(
    "http://localhost:5000/aria/webhook/ticket-resolved",
    json={
        "ticket_id": "TICKET-001",
        "project_id": "PROJ123"
    }
)
print(json.dumps(ticket_response.json(), indent=2))