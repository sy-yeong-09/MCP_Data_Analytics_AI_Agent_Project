# MCP Data Analytics AI Agent

Upload a CSV, XLS or XLSX file and ask questions in the Flask chat interface. The LangChain model selects tools from one Streamable HTTP MCP aggregator. The aggregator connects to Pandas and the reused SentenceTransformer/FAISS RAG implementation over STDIO, and Tavily Search and Extract over Streamable HTTP.

## Run on Windows PowerShell

From the extracted project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `LLM_API_KEY` in `.env` to your OpenRouter key. Choose an OpenRouter model that supports tool calling and is available to your account. The supplied `LLM_MODEL` is configurable. A blank `FLASK_SECRET_KEY` generates a temporary secret at startup. Set a private random value if you want local sessions to survive a Flask restart.

Tavily uses its hosted MCP keyless header by default. Use the template's `WEB_SEARCH_MCP_URL=https://mcp.tavily.com/mcp/` and remove the old `?tavilyApiKey=tvly-keyless` suffix when copying settings from an earlier version. Add `TAVILY_API_KEY` if you have a key. Local tools are available if remote discovery fails. Restart the aggregator after editing its settings. Set `ENABLE_WEB_SEARCH=false` for local-only testing. [Tavily keyless setup](https://docs.tavily.com/documentation/keyless)

In the first terminal:

```powershell
python -m aggregator.aggregator_server
```

In a second terminal, activate the same environment:

```powershell
.\.venv\Scripts\Activate.ps1
python test_mcp_tools.py
python app.py
```

Open [the Flask app](http://127.0.0.1:5000). The MCP endpoint is `http://127.0.0.1:8000/mcp`. The smoke script lists tools and performs a real Pandas calculation for each bundled dataset.

## How requests run

- The agent caches each MCP tool result by name and arguments for the duration of the question. A duplicate-only round ends tool calling. The model can still correct arguments after a validation error and combine Pandas, RAG and Tavily.
- Each question allows up to four model rounds that request tools and six distinct MCP executions. A 70-second budget covers tool selection and execution, leaving time for an answer using completed results. The final model call has no tools bound. The request still depends on the model responding.
- The aggregator keeps both STDIO subprocesses open so the Pandas worker reuses parsed tables and the RAG worker reuses its embedding model and recent indexes. The first RAG request can need a model download. You can pre-download it by running `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"` in the virtual environment.
- The SilverBank task table is detected from the project label and task headers in `DevPlan`. Only rows with a task ID are included. Blank completion and effort fields remain missing. Pandas means skip blank values and an entirely blank sum is returned as missing. GroupBy `size` counts rows, while `count` counts nonblank values in a selected column.
- Flask stores the active upload ID in the browser session and validates uploads before accepting them. Keep both services on loopback for this local project. Uploaded files remain in `uploads/` until removed. The supplied workbook contains internal project material. Review and anonymize it before external distribution.


