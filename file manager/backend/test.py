import requests
print(requests.get("http://ollama:11434/api/tags", timeout=5).text)
