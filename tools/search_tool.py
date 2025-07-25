import os
import json
import requests
from langchain.tools import tool

serper_api_key = os.getenv("SERPER_API_KEY")

@tool
def search_the_internet(argument: str) -> str:
    """
    Searches the internet for a given topic using Serper API and retrieves relevant results.
    """
    serper_api_key = os.getenv("SERPER_API_KEY")
    search_url = "https://google.serper.dev/search"
    top_result_to_return = 4
    payload = json.dumps({"q": argument})
    headers = {
        'X-API-KEY': serper_api_key,
        'content-type': 'application/json'
    }
    response = requests.request("POST", search_url, headers=headers, data=payload)
    data = response.json()

    if 'organic' not in data:
        return """Apologies, I couldn't locate any results for that query. 
                  The problem might be with your Serper API key."""
    else:
        results = data['organic']
        string = []
        for result in results[:top_result_to_return]:
            try:
                string.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}", "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(string)