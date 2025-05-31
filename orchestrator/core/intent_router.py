from typing import Dict, List
import ollama  # type: ignore
import json

def classify_intent(query: str, history: List[Dict[str, str]]) -> Dict:
    prompt = (
        "You are an intent classifier for a multi-agent system. Classify the user's query into one of these intents: "
        "'generic_question', 'article_retrieval', 'recommendation', 'follow_up'. Use the conversation history for context. "
        "Return a JSON object with 'intent' and optional 'keywords' (list), 'problem' (string), or 'project' (string) fields.\n\n"
        f"Query: {query}\n"
        f"History: {history}\n\n"
        "Example outputs:\n"
        "- Generic question: {'intent': 'generic_question'}\n"
        "- Article retrieval: {'intent': 'article_retrieval', 'keywords': ['Kubernetes', 'deployment']}\n"
        "- Recommendation: {'intent': 'recommendation', 'problem': 'app crashes with 500 error'}\n"
        "- Recommendation with project: {'intent': 'recommendation', 'project': 'PROJ123'}\n"
        "- Follow-up: {'intent': 'follow_up'}\n\n"
        "Ensure the response is valid JSON."
    )
    
    try:
        response = ollama.generate(model="mistral", prompt=prompt)
        result = json.loads(response["response"])
        #print(f"Intent classification response: {result}")
        if "intent" not in result:
            raise ValueError("No intent in response")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Intent classification error: {e}")
        result = {"intent": "generic_question"}
    
    return result