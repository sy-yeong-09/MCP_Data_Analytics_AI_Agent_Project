"""Bounded LangChain tool calling through the MCP aggregator."""

import asyncio
import json
import logging
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import RateLimitError

from agent.mcp_client import aggregator_tools
from config.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 6
DATASET_PREFIXES = ("pandas_", "rag_")


def _response_text(message):
    return "\n".join(
        block["text"] for block in message.content_blocks if block["type"] == "text"
    ).strip()


async def _answer(model, tools, question, dataset_id):
    """Let the model choose tools, with a fresh budget and result cache per question."""
    by_name = {tool.name: tool for tool in tools}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Active dataset ID: {dataset_id}\nQuestion: {question}"),
    ]
    tool_model = model.bind_tools(tools)
    results = {}
    tools_deadline = asyncio.get_running_loop().time() + 70

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            async with asyncio.timeout_at(tools_deadline):
                reply = await tool_model.ainvoke(messages)
        except TimeoutError:
            break
        if not reply.tool_calls:
            return _response_text(reply) or "The model returned an empty answer."

        messages.append(reply)
        executed = False
        for call in reply.tool_calls:
            name = call["name"]
            args = dict(call["args"])
            if name.startswith(DATASET_PREFIXES):
                args["dataset_id"] = dataset_id
            key = (name, json.dumps(args, sort_keys=True))

            time_left = tools_deadline - asyncio.get_running_loop().time()
            if key not in results and name in by_name and len(results) < MAX_TOOL_CALLS and time_left > 0:
                executed = True
                start = time.perf_counter()
                try:
                    async with asyncio.timeout(min(time_left, 45 if name.startswith("rag_") else 15)):
                        result = await by_name[name].ainvoke({**call, "args": args})
                    content, status = result.content, result.status
                except TimeoutError:
                    content, status = f"{name} timed out.", "error"
                except Exception:
                    logger.exception("MCP tool failed: %s", name)
                    content, status = f"{name} could not complete the request.", "error"
                results[key] = (content, status)
                logger.info("MCP %s completed in %.2fs (%s)", name, time.perf_counter() - start, status)
            elif key in results:
                logger.info("Reusing MCP result for %s", name)

            content, status = results.get(
                key, (f"{name} is unavailable or the tool budget was reached.", "error")
            )
            messages.append(
                ToolMessage(content=content, status=status, name=name, tool_call_id=call["id"])
            )

        if not executed or len(results) >= MAX_TOOL_CALLS:
            break

    messages.append(HumanMessage(content=(
        "Give your final answer using the tool results already present. "
        "If any requested information is missing, explain what you could verify "
        "or ask one focused clarification."
    )))
    reply = await model.ainvoke(messages)
    return _response_text(reply) or "The model returned an empty answer."


async def ask(question: str, dataset_id: str) -> str:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Configure LLM_API_KEY or OPENROUTER_API_KEY in .env")

    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL") or os.getenv("MODEL") or "openrouter/free",
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
        timeout=25,
        max_retries=0,
    )

    async with aggregator_tools() as tools:
        try:
            async with asyncio.timeout(100):
                return await _answer(model, tools, question, dataset_id)
        except (RateLimitError, TimeoutError) as exc:
            failure = exc
    raise failure


def ask_agent(question: str, dataset_id: str) -> str:
    return asyncio.run(ask(question, dataset_id))
