"""MCP wrapper around the original mini project ExcelRAG implementation."""

from functools import lru_cache
from threading import Lock

from fastmcp import FastMCP

from pandas_mcp.analytics import _tables
from rag_mcp.embeddings import ExcelRAG

mcp = FastMCP("Original Excel RAG")
_index_lock = Lock()


@lru_cache(maxsize=4)
def _load(dataset_id):
    tables = _tables(dataset_id)
    rag = ExcelRAG()
    info = rag.build(
        tables,
        sheet_names=sorted({table["metadata"]["sheet"] for table in tables}),
        workbook_name=dataset_id,
    )
    return rag, info


@mcp.tool
def rag_load_dataset(dataset_id: str) -> dict:
    """Build or reuse the SentenceTransformer and FAISS index for a dataset."""
    with _index_lock:
        return _load(dataset_id)[1]


@mcp.tool
def rag_search_dataset(dataset_id: str, question: str, top_k: int = 6) -> list[dict]:
    """Retrieve relevant workbook chunks, indexing the dataset on first use."""
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    with _index_lock:
        return _load(dataset_id)[0].retrieve(question, top_k=top_k)


@mcp.tool
def rag_get_dataset_info(dataset_id: str) -> dict:
    """Return RAG index statistics for a dataset."""
    with _index_lock:
        return _load(dataset_id)[1]


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
