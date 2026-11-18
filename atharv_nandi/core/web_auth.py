"""Pure (ROS-free) access control for the Atharv Nandi operator console (atharv_nandi/nodes/web_bridge.py).

Roles
  operator  drive, services, manual arm, gait, envelope, missions, E-STOP and E-STOP reset
  viewer    receives the state stream only; may press E-STOP (stopping is never blocked)
  dev       no tokens configured and not in field mode: today's behaviour, everyone may drive,
            and the browser shows a 'DEV: no auth' badge

Tokens come from NANDI_OPERATOR_TOKEN / NANDI_VIEWER_TOKEN and are compared with
hmac.compare_digest. A client authenticates with the WebSocket handshake query (?token=...)
or with {"type": "auth", "token": "..."} as its first message, within AUTH_TIMEOUT_S.
In field mode (NANDI_FIELD_MODE=1) tokens are mandatory: AuthConfig.from_env raises
AuthConfigError and the bridge refuses to start.

Every accepted command is appended to an audit log (JSON lines {t, client, role, type, name, values}).
"""
import hmac
import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"
ROLE_DEV = "dev"
ROLES = (ROLE_OPERATOR, ROLE_VIEWER, ROLE_DEV)

AUTH_TIMEOUT_S = 5.0
MIN_TOKEN_CHARS = 8
FIELD_MIN_TOKEN_CHARS = 16  # NANDI_FIELD_MODE=1: tokens are reachable from the field network
MAX_AUTH_ATTEMPTS = 3  # wrong tokens per connection before it is closed
IP_FAIL_LIMIT = 10  # failed authentications per peer address within IP_FAIL_WINDOW_S ...
IP_FAIL_WINDOW_S = 60.0  # ... before new handshakes from it are refused (HTTP 429)
MAX_TOKEN_CHARS = 512
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Message types / service names anyone authenticated may send (E-STOP must never be blocked).
ALWAYS_ALLOWED_TYPES = frozenset({"auth", "ping"})
STOP_SERVICES = frozenset({"estop"})
# Everything else that changes the robot or the shared world is operator-only.
OPERATOR_TYPES = frozenset({"cmd", "service", "gait_params", "manual_arm", "envelope", "mission_start"})


class AuthConfigError(RuntimeError):
    """The bridge must not start with this configuration (field mode without tokens)."""


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _token(env: Mapping[str, str], key: str) -> Optional[str]:
    v = (env.get(key) or "").strip()
    return v or None


@dataclass(frozen=True)
class AuthConfig:
    operator_token: Optional[str] = None
    viewer_token: Optional[str] = None
    field_mode: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AuthConfig":
        env = os.environ if env is None else env
        cfg = cls(_token(env, "NANDI_OPERATOR_TOKEN"), _token(env, "NANDI_VIEWER_TOKEN"),
                  _truthy(env.get("NANDI_FIELD_MODE")))
        cfg.validate()
        return cfg

    def validate(self):
        low = FIELD_MIN_TOKEN_CHARS if self.field_mode else MIN_TOKEN_CHARS
        for name, tok in (("NANDI_OPERATOR_TOKEN", self.operator_token), ("NANDI_VIEWER_TOKEN", self.viewer_token)):
            if tok is not None and not low <= len(tok) <= MAX_TOKEN_CHARS:
                mode = " in field mode" if self.field_mode else ""
                raise AuthConfigError(f"{name} must be {low}..{MAX_TOKEN_CHARS} characters{mode}")
        if self.operator_token and self.viewer_token and hmac.compare_digest(
                self.operator_token.encode(), self.viewer_token.encode()):
            raise AuthConfigError("NANDI_OPERATOR_TOKEN and NANDI_VIEWER_TOKEN must differ")
        if self.field_mode and not self.operator_token:
            raise AuthConfigError("NANDI_FIELD_MODE=1 requires NANDI_OPERATOR_TOKEN (and optionally "
                                  "NANDI_VIEWER_TOKEN); refusing to start an unauthenticated field console")

    @property
    def auth_required(self) -> bool:
        return self.field_mode or bool(self.operator_token or self.viewer_token)

    @property
    def dev_mode(self) -> bool:
        return not self.auth_required

    def role_for(self, token) -> Optional[str]:
        """Constant-time token -> role (None if wrong). Both tokens are always compared."""
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_CHARS:
            return None
        given = token.encode()
        op = hmac.compare_digest(given, self.operator_token.encode()) if self.operator_token else False
        vw = hmac.compare_digest(given, self.viewer_token.encode()) if self.viewer_token else False
        if op:
            return ROLE_OPERATOR
        if vw:
            return ROLE_VIEWER
        return None

    def startup_warning(self, bind_host: str) -> Optional[str]:
        """Human-readable warning for risky-but-allowed setups (dev mode on a non-loopback bind)."""
        if self.dev_mode and bind_host not in LOOPBACK_HOSTS:
            return (f"web console bound to {bind_host} with no NANDI_OPERATOR_TOKEN: anyone who can reach "
                    "the port can drive the robot (DEV: no auth). Publish the port on 127.0.0.1 only, or set tokens.")
        if self.auth_required and not self.operator_token:
            return "only NANDI_VIEWER_TOKEN is set: nobody can drive from the web console"
        return None


def token_from_path(path: Optional[str]) -> Optional[str]:
    """?token=... from a WebSocket handshake request path, or None."""
    if not path or "?" not in path:
        return None
    try:
        values = parse_qs(urlsplit(path).query, max_num_fields=8).get("token")
    except ValueError:
        return None
    return values[0] if values else None


class ClientAuth:
    """Per-connection auth state machine. Pure: callers pass the clock in."""

    def __init__(self, config: AuthConfig, now: float, handshake_token: Optional[str] = None):
        self.config = config
        self.deadline = now + AUTH_TIMEOUT_S
        self.role: Optional[str] = ROLE_DEV if config.dev_mode else None
        self.failures = 0
        if self.role is None and handshake_token is not None:
            self.role = config.role_for(handshake_token)
            if self.role is None:
                self.failures = 1

    @property
    def authenticated(self) -> bool:
        return self.role is not None

    @property
    def locked_out(self) -> bool:
        """True once MAX_AUTH_ATTEMPTS wrong tokens were tried: the caller closes the connection."""
        return not self.authenticated and self.failures >= MAX_AUTH_ATTEMPTS

    def try_token(self, token) -> bool:
        if self.config.dev_mode:
            return True
        if self.locked_out:
            return False  # no more guesses on this connection, not even the right token
        role = self.config.role_for(token)
        if role is not None:
            self.role = role
        else:
            self.failures += 1
        return role is not None

    def expired(self, now: float) -> bool:
        """True once an unauthenticated client has used up its AUTH_TIMEOUT_S."""
        return not self.authenticated and now >= self.deadline


class AuthThrottle:
    """Per-peer-address failed-authentication counter (sliding window). Pure: callers pass the clock.

    Reconnecting resets ClientAuth's per-connection limit, so the bridge also refuses new
    handshakes from an address with IP_FAIL_LIMIT failures in the last IP_FAIL_WINDOW_S.
    """

    MAX_PEERS = 1024  # bound memory under a spray of source addresses

    def __init__(self, limit: int = IP_FAIL_LIMIT, window_s: float = IP_FAIL_WINDOW_S):
        self.limit, self.window_s = limit, window_s
        self._fails: Dict[str, list] = {}

    def _prune(self, peer: str, now: float) -> list:
        times = [t for t in self._fails.get(peer, ()) if now - t < self.window_s]
        if times:
            self._fails[peer] = times
        else:
            self._fails.pop(peer, None)
        return times

    def record_failure(self, peer: Optional[str], now: float):
        if not peer:
            return
        times = self._prune(peer, now)
        if peer not in self._fails and len(self._fails) >= self.MAX_PEERS:
            for p in list(self._fails):
                self._prune(p, now)
            if len(self._fails) >= self.MAX_PEERS:
                self._fails.pop(next(iter(self._fails)))
        times.append(now)
        self._fails[peer] = times[-self.limit:]

    def blocked(self, peer: Optional[str], now: float) -> bool:
        return bool(peer) and len(self._prune(peer, now)) >= self.limit


def permit(role: Optional[str], msg_type: str, name: Optional[str] = None) -> Tuple[bool, str]:
    """Role matrix -> (allowed, reason). `name` is the service name for type 'service'."""
    if msg_type in ALWAYS_ALLOWED_TYPES:
        return True, ""
    if role is None:
        return False, "not authenticated: send {\"type\": \"auth\", \"token\": ...} first"
    if msg_type == "service" and name in STOP_SERVICES:
        return True, ""  # any authenticated role may stop the robot
    if role in (ROLE_OPERATOR, ROLE_DEV):
        return True, ""
    if role == ROLE_VIEWER:
        what = f"service {name}" if msg_type == "service" and name else msg_type
        return False, f"viewer role is read-only: {what} refused (E-STOP is allowed)"
    return False, "unknown role"


# ---------------------------------------------------------------- audit log

AUDITED_TYPES = frozenset(OPERATOR_TYPES)
# Only an identical repeat (same type, name AND values) from one client within this window is
# collapsed, e.g. the 5 Hz held-walk refresh. Any change of values is a new row.
DEDUPE_WINDOW_S = 1.0
MAX_AUDIT_VALUES = 16


def _audit_values(values):
    """Command payload -> small JSON-safe value for the audit row (None if there is none)."""
    if values is None:
        return None
    if isinstance(values, (bool, int, float, str)):
        return round(values, 4) if isinstance(values, float) else values
    if isinstance(values, (list, tuple)):
        return [_audit_values(v) for v in list(values)[:MAX_AUDIT_VALUES]]
    return str(values)[:80]


def audit_record(t: float, client: str, role: Optional[str], msg_type: str, name: Optional[str] = None,
                 values=None) -> Dict:
    row = {"t": round(float(t), 3), "client": str(client), "role": role, "type": msg_type, "name": name}
    if values is not None:
        row["values"] = _audit_values(values)
    return row


class AuditLog:
    """Append-only JSON-lines audit of accepted commands. Never raises into the caller."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._last: Dict[str, Tuple[str, Optional[str], str, float]] = {}
        self._lock = threading.Lock()
        self.errors = 0

    @staticmethod
    def _key(values) -> str:
        return json.dumps(_audit_values(values), separators=(",", ":"))

    def should_log(self, client: str, msg_type: str, name: Optional[str], t: float, values=None) -> bool:
        if msg_type not in AUDITED_TYPES:
            return False
        if msg_type == "service" and name in STOP_SERVICES:
            return True  # every E-STOP press is recorded
        last = self._last.get(client)
        if (last is not None and last[0] == msg_type and last[1] == name and last[2] == self._key(values)
                and t - last[3] < DEDUPE_WINDOW_S):
            return False
        return True

    def record(self, t: float, client: str, role: Optional[str], msg_type: str, name: Optional[str] = None,
               values=None) -> bool:
        # The dedupe check and its update are one critical section: two bridge threads recording the same command
        # for one client both passed should_log before either updated _last (WPCOV review: double audit rows).
        with self._lock:
            if not self.should_log(client, msg_type, name, t, values):
                return False
            self._last[client] = (msg_type, name, self._key(values), t)
            if not self.path:
                return True
            line = json.dumps(audit_record(t, client, role, msg_type, name, values), separators=(",", ":")) + "\n"
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                self.errors += 1
        return True

    def forget(self, client: str):
        with self._lock:
            self._last.pop(client, None)
