// WebSocket link to sim/web_bridge.py with exponential-backoff auto-reconnect.

export class BridgeConnection {
  constructor(url, { onOpen, onClose, onMessage } = {}) {
    this.url = url;
    this.handlers = { onOpen, onClose, onMessage };
    this.ws = null;
    this.retryMs = 500;
  }

  get connected() {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  connect() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.retryMs = 500;
      this.handlers.onOpen?.();
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return;
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg && typeof msg === 'object') this.handlers.onMessage?.(msg);
    };
    ws.onerror = () => ws.close();
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.handlers.onClose?.();
      setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(5000, this.retryMs * 2);
    };
  }

  /** Send a JSON message; returns false (and drops it) while disconnected. */
  send(obj) {
    if (!this.connected) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }
}

export function bridgeUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws`;
}
