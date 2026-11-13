"""Expose local STDIO tools and the remote web MCP at a single HTTP endpoint."""

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.server.providers.proxy import ProxyProvider, ProxyTool

from config.settings import HOST, PORT, ROOT
from web_search_mcp.web_search_config import web_transport

logger = logging.getLogger(__name__)


def local_client(module):
    return Client(
        StdioTransport(
            sys.executable, ["-m", module], cwd=str(ROOT), keep_alive=False,
            env={
                name: value for name, value in os.environ.items()
                if name.startswith(("HF_", "OMP_", "EMBEDDING_", "RAG_"))
                or name.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
            },
        ),
        timeout=60,
    )


pandas_client = local_client("pandas_mcp.pandas_server")
rag_client = local_client("rag_mcp.rag_server")


@asynccontextmanager
async def lifespan(server):
    async with AsyncExitStack() as stack:
        # Hold both connections open for the server lifetime so the workers
        # retain workbook and vector-index caches across independent calls.
        await stack.enter_async_context(pandas_client)
        await stack.enter_async_context(rag_client)
        transport = web_transport()
        if transport is not None:
            web_client = Client(transport, timeout=20, init_timeout=15)
            try:
                async with asyncio.timeout(25):
                    await stack.enter_async_context(web_client)
                    tools = await web_client.list_tools()
            except Exception:
                logger.warning("Web MCP discovery failed. Check the Tavily configuration and restart.")
            else:
                web = FastMCP("Tavily")
                for tool in tools:
                    if tool.name in {"tavily_search", "tavily_extract"}:
                        web.add_tool(ProxyTool.from_mcp_tool(lambda: web_client, tool))
                server.mount(web, namespace="web")
                logger.info("Web MCP search and extract tools loaded.")
        yield


aggregator = FastMCP("Data Analytics Aggregator", lifespan=lifespan)
aggregator.add_provider(ProxyProvider(lambda: pandas_client), namespace="pandas")
aggregator.add_provider(ProxyProvider(lambda: rag_client), namespace="rag")

if __name__ == "__main__":
    aggregator.run(transport="http", host=HOST, port=PORT, show_banner=False)
