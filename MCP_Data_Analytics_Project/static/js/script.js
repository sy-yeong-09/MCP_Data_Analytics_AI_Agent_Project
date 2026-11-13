const upload = document.querySelector('#upload');
const chat = document.querySelector('#chat');
const info = document.querySelector('#info');
const messages = document.querySelector('#messages');
const fileInput = document.querySelector('#file');
const selectedFile = document.querySelector('#selected-file');
const dropzone = document.querySelector('#dropzone');
const datasetDot = document.querySelector('#dataset-dot');
const uploadButton = document.querySelector('#upload-button');
const askButton = document.querySelector('#ask-button');
const questionInput = document.querySelector('#question');

function scrollMessages() { messages.scrollTop = messages.scrollHeight; }
function showMessage(role, content) {
  document.querySelector('#welcome')?.remove();
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : role === 'error' ? 'Error' : 'AI Assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  wrapper.append(label, bubble);
  messages.append(wrapper);
  scrollMessages();
  return { wrapper, bubble };
}
function renderAnswer(bubble, result) {
  if (typeof result.answer_html !== 'string') {
    bubble.textContent = result.answer ?? 'No answer returned.';
    return;
  }
  bubble.classList.add('formatted-answer');
  // Only use the Flask Markdown renderer's output, which escapes raw HTML.
  bubble.innerHTML = result.answer_html;
}

function friendlyError(status, message) {
  if (status === 429 || /429|rate.limit|free.models.per.day/i.test(message)) {
    return 'The AI model has reached its request limit. Please wait for the quota to reset or try another available model.';
  }
  if (status === 503 || /timed? out|timeout/i.test(message)) {
    return 'The AI service is taking too long or is temporarily unavailable. Please try again shortly.';
  }
  return message || 'Something went wrong. Please try again.';
}
async function parseResponse(response) {
  const raw = await response.text();
  try { return JSON.parse(raw); }
  catch { throw new Error(`Unexpected server response (HTTP ${response.status}).`); }
}
function readableLabel(key) {
  return String(key).replace(/[_-]+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, letter => letter.toUpperCase());
}
function renderDatasetStatus(filename, data) {
  info.replaceChildren();
  const heading = document.createElement('div');
  heading.className = 'dataset-filename';
  heading.textContent = filename;
  info.append(heading);
  const addField = (parent, label, value) => {
    const field = document.createElement('div');
    field.className = 'dataset-field';
    const name = document.createElement('span');
    name.className = 'dataset-field-label';
    name.textContent = readableLabel(label);
    const detail = document.createElement('span');
    detail.className = 'dataset-field-value';
    detail.textContent = String(value);
    field.append(name, detail);
    parent.append(field);
  };
  const renderValue = (parent, label, value, depth = 0) => {
    if (value === null || value === undefined) return;
    if (Array.isArray(value)) {
      if (value.length === 0) { addField(parent, label, 'None'); return; }
      if (value.every(item => item === null || typeof item !== 'object')) {
        const group = document.createElement('div');
        group.className = 'dataset-group';
        const title = document.createElement('div');
        title.className = 'dataset-group-title';
        title.textContent = readableLabel(label);
        const chips = document.createElement('div');
        chips.className = 'dataset-chips';
        value.forEach(item => {
          const chip = document.createElement('span');
          chip.className = 'dataset-chip';
          chip.textContent = String(item);
          chips.append(chip);
        });
        group.append(title, chips);
        parent.append(group);
      } else {
        const group = document.createElement('div');
        group.className = 'dataset-group';
        const title = document.createElement('div');
        title.className = 'dataset-group-title';
        title.textContent = `${readableLabel(label)} (${value.length})`;
        group.append(title);
        value.forEach((item, index) => renderValue(group, `${readableLabel(label)} ${index + 1}`, item, depth + 1));
        parent.append(group);
      }
    } else if (typeof value === 'object') {
      const group = document.createElement('div');
      group.className = 'dataset-group';
      if (depth > 0 || label !== 'Dataset information') {
        const title = document.createElement('div');
        title.className = 'dataset-group-title';
        title.textContent = readableLabel(label);
        group.append(title);
      }
      Object.entries(value).forEach(([key, entry]) => renderValue(group, key, entry, depth + 1));
      parent.append(group);
    } else {
      addField(parent, label, value);
    }
  };
  if (data && typeof data === 'object') renderValue(info, 'Dataset information', data);
  else if (data !== undefined) renderValue(info, 'Status', data);
}
function showSelectedFile() {
  selectedFile.textContent = fileInput.files[0]?.name || 'No file selected';
}
fileInput.addEventListener('change', showSelectedFile);
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault(); dropzone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault(); dropzone.classList.remove('dragover');
}));
dropzone.addEventListener('drop', event => {
  if (!event.dataTransfer.files.length) return;
  const transfer = new DataTransfer();
  transfer.items.add(event.dataTransfer.files[0]);
  fileInput.files = transfer.files;
  showSelectedFile();
});

upload.addEventListener('submit', async event => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) { info.textContent = 'Please choose a file first.'; return; }
  uploadButton.disabled = true;
  uploadButton.textContent = 'Uploading...';
  datasetDot.className = 'status-dot';
  info.textContent = 'Uploading and reading your dataset...';
  const data = new FormData();
  data.append('file', file);
  try {
    const response = await fetch('/api/upload', { method: 'POST', body: data });
    const result = await parseResponse(response);
    if (!response.ok) throw new Error(friendlyError(response.status, result.error));
    renderDatasetStatus(result.filename || file.name, result.info || {});
    datasetDot.className = 'status-dot ready';
    messages.replaceChildren();
    showMessage('assistant', `Dataset "${result.filename || file.name}" uploaded successfully. What would you like to know?`);
  } catch (error) {
    datasetDot.className = 'status-dot error';
    info.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = 'Upload dataset';
  }
});

chat.addEventListener('submit', async event => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || askButton.disabled) return;
  questionInput.value = '';
  showMessage('user', question);
  const pending = showMessage('assistant', 'Analyzing your dataset...');
  pending.wrapper.classList.add('typing');
  askButton.disabled = true;
  askButton.textContent = 'Analyzing...';
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const result = await parseResponse(response);
    if (!response.ok) throw new Error(friendlyError(response.status, result.error));
    renderAnswer(pending.bubble, result);
  } catch (error) {
    pending.wrapper.classList.remove('assistant');
    pending.wrapper.classList.add('error');
    pending.wrapper.querySelector('.message-label').textContent = 'Error';
    pending.bubble.textContent = error.message;
  } finally {
    pending.wrapper.classList.remove('typing');
    askButton.disabled = false;
    askButton.innerHTML = 'Send <span aria-hidden="true">➤</span>';
    scrollMessages();
    questionInput.focus();
  }
});
questionInput.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    chat.requestSubmit();
  }
});
document.querySelectorAll('.example').forEach(button => button.addEventListener('click', () => {
  questionInput.value = button.dataset.question;
  questionInput.focus();
}));
