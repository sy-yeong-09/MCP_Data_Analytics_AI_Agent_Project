"""Configure Tavily hosted MCP with either keyless access or an API key."""

import os

from fastmcp.client.transports import StreamableHttpTransport


def web_transport():
    if os.getenv("ENABLE_WEB_SEARCH", "true").lower() not in {"1", "true", "yes"}:
        return None
    url = os.getenv("WEB_SEARCH_MCP_URL") or "https://mcp.tavily.com/mcp/"
    api_key = os.getenv("TAVILY_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {
        "X-Tavily-Access-Mode": "keyless",
    }
    return StreamableHttpTransport(url, headers=headers)
