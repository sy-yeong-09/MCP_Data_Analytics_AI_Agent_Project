import logging
import os
import secrets
import uuid
from pathlib import Path
from flask import Flask,request,session,jsonify,render_template
from markdown_it import MarkdownIt
from werkzeug.utils import secure_filename
from config.settings import ROOT,UPLOADS
from pandas_mcp.analytics import get_dataset_info
from agent.ai_agent import ask_agent
app=Flask(__name__)
# Model output can contain untrusted text from web pages and uploaded records.
answer_markdown = MarkdownIt("js-default", {"html": False}).disable("image")
configured_secret = os.getenv('FLASK_SECRET_KEY')
app.secret_key = (
    configured_secret if configured_secret and configured_secret != 'replace-with-long-random-value'
    else secrets.token_hex(32)
)
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
app.config['MAX_CONTENT_LENGTH']=int(os.getenv('MAX_UPLOAD_MB','20'))*1024*1024
@app.get('/')
def index():return render_template('index.html')
@app.post('/api/upload')
def upload():
    f=request.files.get('file')
    if not f or not f.filename:return jsonify(error='Select a file'),400
    name=secure_filename(f.filename)
    if Path(name).suffix.lower() not in ('.csv','.xls','.xlsx'):return jsonify(error='CSV/XLS/XLSX only'),400
    dataset_id=uuid.uuid4().hex
    dest=UPLOADS/f'{dataset_id}_{name}'
    f.save(dest)
    try:info=get_dataset_info(dataset_id)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return jsonify(error=str(exc)),400
    session['dataset_id']=dataset_id
    return jsonify(dataset_id=dataset_id,filename=name,info=info)
@app.post('/api/chat')
def chat():
    dataset_id=session.get('dataset_id')
    if not dataset_id:return jsonify(error='Upload a dataset first'),400
    payload = request.get_json(silent=True)
    question = payload.get('question') if isinstance(payload, dict) else None
    if not isinstance(question, str) or not question.strip():
        return jsonify(error='Enter a question'), 400
    question = question.strip()
    try:
        answer = ask_agent(question, dataset_id)
        return jsonify(answer=answer, answer_html=answer_markdown.render(answer))
    except Exception as exc:
        app.logger.exception('Agent error')
        if getattr(exc, 'status_code', None) == 429:
            return jsonify(error='The model has reached its request limit.'), 429
        return jsonify(error='The agent is temporarily unavailable. Check the app and aggregator logs.'), 503
@app.errorhandler(413)
def too_large(_):return jsonify(error='Upload too large'),413
if __name__=='__main__':app.run(host=os.getenv('FLASK_HOST','127.0.0.1'),port=int(os.getenv('FLASK_PORT','5000')),debug=os.getenv('FLASK_DEBUG','false').lower()=='true')
