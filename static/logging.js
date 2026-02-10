// Logging and error overlay helpers

const errorOverlay = document.getElementById("error-overlay");
const statusEl = document.getElementById("status");

let logOverlay = null;
let _logLines = [];

function createLogOverlay() {
  if (logOverlay) return;
  logOverlay = document.createElement('div');
  logOverlay.id = 'log-overlay';
  Object.assign(logOverlay.style, {
    position: 'absolute',
    right: '12px',
    top: '12px',
    width: '320px',
    maxHeight: '40vh',
    overflowY: 'auto',
    background: 'rgba(0,0,0,0.45)',
    color: '#9ad1ff',
    fontFamily: 'monospace',
    fontSize: '12px',
    padding: '8px',
    borderRadius: '6px',
    zIndex: 9999,
    pointerEvents: 'none',
    whiteSpace: 'pre-wrap'
  });
  document.body.appendChild(logOverlay);
}

function logToOverlay(msg, level='info') {
  try {
    if (!logOverlay) createLogOverlay();
    const ts = (new Date()).toISOString().replace('T',' ').slice(0,19);
    const line = `${ts} ${level.toUpperCase()}: ${msg}`;
    _logLines.push(line);
    if (_logLines.length > 200) _logLines.shift();
    logOverlay.textContent = _logLines.join('\n');
    // keep latest visible
    logOverlay.scrollTop = logOverlay.scrollHeight;
  } catch (e) {
    // ignore overlay errors
  }
}

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
  logToOverlay(message, 'status');
}

function logError(message, err) {
  const details = err ? `\n${err.stack || err}` : "";
  const text = `${message}${details}`;
  if (errorOverlay) {
    errorOverlay.style.display = "block";
    errorOverlay.textContent = text;
  }
  console.error(text);
  logToOverlay(text, 'error');
}
