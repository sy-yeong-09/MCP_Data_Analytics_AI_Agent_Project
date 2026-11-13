"""MCP tool discovery and connection to the local aggregator."""

import asyncio
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config.settings import MCP_URL


REQUIRED_PANDAS_TOOLS = {
    "get_dataset_info",
    "get_column_summary",
    "aggregate_data",
    "groupby_data",
    "filter_dataset",
}

REQUIRED_RAG_TOOLS = {
    "rag_load_dataset",
    "rag_search_dataset",
    "rag_get_dataset_info",
}


@asynccontextmanager
async def aggregator_tools():
    """Open and close the MCP session within the request task."""

    print(f"[MCP] Connecting to {MCP_URL}", flush=True)

    client = MultiServerMCPClient({
        "aggregator": {
            "transport": "streamable_http",
            "url": MCP_URL,
            "timeout": 10,
        }
    })

    async with client.session("aggregator") as session:
        async with asyncio.timeout(20):
            tools = await load_mcp_tools(session)

        names = {tool.name for tool in tools}

        print(
            f"[MCP] Discovered {len(tools)} tools: "
            f"{sorted(names)}",
            flush=True
        )

        required = {f"pandas_{name}" for name in REQUIRED_PANDAS_TOOLS}
        required.update(f"rag_{name}" for name in REQUIRED_RAG_TOOLS)
        missing = sorted(required - names)

        if missing:
            raise RuntimeError(
                "Missing required MCP tools: "
                + ", ".join(missing)
            )

        yield tools
