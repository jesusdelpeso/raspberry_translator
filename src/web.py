"""
Lightweight FastAPI web interface for the real-time translator.

Provides:
- Live transcript display via Server-Sent Events (SSE)
- Start / Stop controls
- Configuration overview panel
- REST API for status, history, and config

Usage::

    # Via the CLI flag:
    raspberry-translator --web --port 7860

    # Direct uvicorn (advanced):
    uvicorn src.web:build_app --factory --host 0.0.0.0 --port 7860

The translator runs in a background thread once started. New transcript
entries are broadcast to all connected SSE clients as JSON events.
"""

import asyncio
import json
import logging
import queue
import threading
from dataclasses import asdict
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded single-page UI
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Raspberry Translator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117;
         color: #e1e4e8; min-height: 100vh; display: flex; flex-direction: column; }
  header { background: #161b22; border-bottom: 1px solid #30363d;
           padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.25rem; font-weight: 600; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #6e7681; flex-shrink: 0; }
  .status-dot.running { background: #3fb950; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  button { padding: 6px 16px; border-radius: 6px; border: none; cursor: pointer;
           font-size: 0.875rem; font-weight: 600; transition: opacity .15s; }
  button:hover { opacity: .85; }
  #btn-start { background: #238636; color: #fff; }
  #btn-stop  { background: #da3633; color: #fff; }
  #status-text { font-size: 0.8rem; color: #8b949e; }
  main { flex: 1; display: grid; grid-template-columns: 1fr 280px; gap: 0;
         max-width: 1200px; width: 100%; margin: 0 auto; padding: 20px; gap: 16px; }
  @media (max-width: 700px) { main { grid-template-columns: 1fr; } }
  .transcript { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                overflow-y: auto; height: calc(100vh - 130px); padding: 12px;
                display: flex; flex-direction: column; gap: 10px; }
  .entry { background: #1c2128; border: 1px solid #30363d; border-radius: 6px;
           padding: 10px 14px; }
  .entry-meta { font-size: 0.75rem; color: #8b949e; margin-bottom: 6px; }
  .entry-source { font-size: 0.9rem; margin-bottom: 4px; }
  .entry-source span { color: #58a6ff; font-weight: 600; }
  .entry-trans { font-size: 0.9rem; color: #3fb950; }
  .empty-state { color: #8b949e; font-size: 0.875rem; margin: auto;
                 text-align: center; padding: 40px; }
  aside { display: flex; flex-direction: column; gap: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 14px; }
  .card h2 { font-size: 0.875rem; font-weight: 600; color: #8b949e;
             text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }
  .config-row { display: flex; justify-content: space-between; padding: 4px 0;
                border-bottom: 1px solid #21262d; font-size: 0.8rem; }
  .config-row:last-child { border: none; }
  .config-key { color: #8b949e; }
  .config-val { color: #e1e4e8; font-family: monospace; text-align: right;
                max-width: 140px; overflow-wrap: anywhere; }
  #conn-status { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
</style>
</head>
<body>
<header>
  <div class="status-dot" id="status-dot"></div>
  <h1>🎙 Raspberry Translator</h1>
  <span id="status-text">Stopped</span>
  <div class="controls">
    <button id="btn-start" onclick="startTranslator()">▶ Start</button>
    <button id="btn-stop"  onclick="stopTranslator()" disabled>■ Stop</button>
  </div>
</header>
<main>
  <div>
    <div class="transcript" id="transcript">
      <div class="empty-state" id="empty-state">
        Press <strong>Start</strong> to begin live translation…
      </div>
    </div>
    <div id="conn-status"></div>
  </div>
  <aside>
    <div class="card">
      <h2>Configuration</h2>
      <div id="config-rows"></div>
    </div>
    <div class="card">
      <h2>Session</h2>
      <div id="session-rows">
        <div class="config-row">
          <span class="config-key">Entries</span>
          <span class="config-val" id="entry-count">0</span>
        </div>
      </div>
    </div>
  </aside>
</main>
<script>
let evtSource = null;
let entryCount = 0;

async function fetchConfig() {
  try {
    const r = await fetch('/api/config');
    const cfg = await r.json();
    const rows = document.getElementById('config-rows');
    const show = [
      ['Source lang', cfg.source_lang],
      ['Target lang', cfg.target_lang],
      ['STT model',   cfg.stt_model.split('/').pop()],
      ['Sample rate', cfg.sample_rate + ' Hz'],
      ['GPU',         cfg.use_gpu ? 'Yes' : 'No'],
      ['ONNX',        cfg.use_onnx ? 'Yes' : 'No'],
      ['Streaming',   cfg.streaming_enabled ? 'Yes' : 'No'],
      ['Bidirectional', cfg.bidirectional_mode ? 'Yes' : 'No'],
    ];
    rows.innerHTML = show.map(([k,v]) =>
      `<div class="config-row"><span class="config-key">${k}</span>` +
      `<span class="config-val">${v}</span></div>`
    ).join('');
  } catch(e) { console.warn('config fetch failed', e); }
}

async function fetchHistory() {
  try {
    const r = await fetch('/api/history');
    const entries = await r.json();
    entries.forEach(addEntry);
  } catch(e) {}
}

function addEntry(e) {
  const t = document.getElementById('transcript');
  document.getElementById('empty-state')?.remove();
  entryCount++;
  document.getElementById('entry-count').textContent = entryCount;
  const ts = e.timestamp.replace('T', ' ').slice(0, 19);
  const div = document.createElement('div');
  div.className = 'entry';
  div.innerHTML =
    `<div class="entry-meta">${ts} &nbsp;·&nbsp; ${e.source_lang} → ${e.target_lang}</div>` +
    `<div class="entry-source"><span>You:</span> ${esc(e.source_text)}</div>` +
    `<div class="entry-trans">⟶ ${esc(e.translated_text)}</div>`;
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function connectSSE() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  evtSource = new EventSource('/api/transcript');
  evtSource.onmessage = ev => { addEntry(JSON.parse(ev.data)); };
  evtSource.onopen = () => {
    document.getElementById('conn-status').textContent = 'Live connection active';
  };
  evtSource.onerror = () => {
    document.getElementById('conn-status').textContent = 'Connection lost – retrying…';
  };
}

async function startTranslator() {
  const r = await fetch('/api/start', {method:'POST'});
  const d = await r.json();
  if (d.ok) {
    document.getElementById('status-dot').classList.add('running');
    document.getElementById('status-text').textContent = 'Running';
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled  = false;
    connectSSE();
  } else {
    alert('Could not start: ' + (d.error || 'unknown error'));
  }
}

async function stopTranslator() {
  await fetch('/api/stop', {method:'POST'});
  document.getElementById('status-dot').classList.remove('running');
  document.getElementById('status-text').textContent = 'Stopped';
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled  = true;
  if (evtSource) { evtSource.close(); evtSource = null; }
  document.getElementById('conn-status').textContent = '';
}

async function syncStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.running) {
      document.getElementById('status-dot').classList.add('running');
      document.getElementById('status-text').textContent = 'Running';
      document.getElementById('btn-start').disabled = true;
      document.getElementById('btn-stop').disabled  = false;
      connectSSE();
    }
  } catch(e) {}
}

fetchConfig();
syncStatus();
fetchHistory();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# SSE broadcaster
# ---------------------------------------------------------------------------

class _SSEBroadcaster:
    """Fan-out new transcript entries to all active SSE connections."""

    def __init__(self) -> None:
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._clients = [c for c in self._clients if c is not q]

    def publish(self, entry_dict: dict) -> None:
        payload = json.dumps(entry_dict, ensure_ascii=False)
        with self._lock:
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(config: Optional[Config] = None) -> FastAPI:
    """Create and return the FastAPI application.

    Parameters
    ----------
    config:
        Translator configuration.  When *None* a default :class:`Config` is
        used.  This makes it easy to inject a custom config in tests.
    """
    if config is None:
        config = Config()

    app = FastAPI(title="Raspberry Translator", version="1.0.0")

    broadcaster = _SSEBroadcaster()

    # Translator state — managed in a background thread
    _state: dict = {
        "translator": None,
        "thread": None,
        "running": False,
        "history": [],       # list[dict] — entries already seen
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_translator():
        """Target function for the translator background thread."""
        from .translator import RealTimeTranslator

        try:
            translator = RealTimeTranslator(config)
        except Exception as exc:
            logger.error("Failed to create translator: %s", exc)
            _state["running"] = False
            return

        _state["translator"] = translator

        # Monkey-patch ConversationHistory.add to also broadcast via SSE
        original_add = translator.history.add

        def _intercepting_add(entry):
            original_add(entry)
            from dataclasses import asdict as _asdict
            d = _asdict(entry)
            _state["history"].append(d)
            broadcaster.publish(d)

        translator.history.add = _intercepting_add

        try:
            translator.start_listening()
        except Exception as exc:
            logger.error("Translator error: %s", exc)
        finally:
            _state["running"] = False
            _state["translator"] = None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content=_HTML)

    @app.get("/api/status")
    async def api_status():
        return {"running": _state["running"]}

    @app.get("/api/config")
    async def api_config():
        from dataclasses import asdict as _asdict
        return _asdict(config)

    @app.get("/api/history")
    async def api_history():
        return _state["history"]

    @app.post("/api/start")
    async def api_start():
        if _state["running"]:
            return {"ok": False, "error": "Already running"}
        _state["running"] = True
        t = threading.Thread(target=_run_translator, daemon=True)
        _state["thread"] = t
        t.start()
        return {"ok": True}

    @app.post("/api/stop")
    async def api_stop():
        tr = _state.get("translator")
        if tr is not None:
            tr.stop()
        _state["running"] = False
        return {"ok": True}

    @app.get("/api/transcript")
    async def api_transcript():
        """Server-Sent Events stream of new transcript entries."""

        async def _generate() -> AsyncGenerator[str, None]:
            q = broadcaster.subscribe()
            try:
                while True:
                    # Poll the thread-safe queue without blocking the event loop
                    try:
                        payload = q.get_nowait()
                        yield f"data: {payload}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            finally:
                broadcaster.unsubscribe(q)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app
