from pathlib import Path
import os
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
UPLOADS=ROOT/'uploads'
UPLOADS.mkdir(exist_ok=True)
DATA=ROOT/'data'
DATA.mkdir(exist_ok=True)
HOST=os.getenv('AGGREGATOR_HOST','127.0.0.1')
PORT=int(os.getenv('AGGREGATOR_PORT','8000'))
MCP_URL=f'http://{HOST}:{PORT}/mcp'
