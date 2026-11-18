// WebSocket link to atharv_nandi/nodes/web_bridge.py with exponential-backoff auto-reconnect.
// When the bridge requires auth, the stored console token is sent as the very first message.

const TOKEN_KEY = 'nandi.token';

export class BridgeConnection {
  constructor(url, { onOpen, onClose, onMessage, getToken } = {}) {
    this.url = url;
    this.handlers = { onOpen, onClose, onMessage };
    this.getToken = getToken ?? (() => null);
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
      const token = this.getToken();
      if (token) ws.send(JSON.stringify({ type: 'auth', token }));  // must be first: 5 s deadline
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

/** Same-origin WebSocket URL (ws: on a plain page, wss: behind TLS). */
export function bridgeUrl() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}/ws`;
}

/** Console token: ?token=... on the page URL (moved into sessionStorage and stripped), else storage. */
export function loadToken() {
  const params = new URLSearchParams(location.search);
  const fromUrl = params.get('token');
  if (fromUrl) {
    saveToken(fromUrl);
    params.delete('token');
    const q = params.toString();
    history.replaceState(null, '', `${location.pathname}${q ? `?${q}` : ''}${location.hash}`);
    return fromUrl;
  }
  try { return sessionStorage.getItem(TOKEN_KEY); } catch { return null; }
}

export function saveToken(token) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch { /* private mode: token lives for this page only */ }
}
