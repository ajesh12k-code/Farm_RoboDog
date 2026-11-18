"""controllers.web_auth: tokens, roles, field mode, the role matrix and the audit log (ROS-free)."""
import json

import pytest

from atharv_nandi.core import web_auth as wa

OP = "op-secret-123456"
VW = "viewer-secret-654321"


def cfg(op=OP, vw=VW, field=False):
    return wa.AuthConfig(op, vw, field)


# ---------------------------------------------------------------- config / env

def test_from_env_reads_tokens_and_field_mode():
    c = wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": OP, "NANDI_VIEWER_TOKEN": VW, "NANDI_FIELD_MODE": "1"})
    assert (c.operator_token, c.viewer_token, c.field_mode) == (OP, VW, True)
    assert c.auth_required and not c.dev_mode


def test_no_tokens_is_dev_mode():
    c = wa.AuthConfig.from_env({})
    assert c.dev_mode and not c.auth_required


def test_blank_tokens_count_as_unset():
    c = wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": "  ", "NANDI_VIEWER_TOKEN": ""})
    assert c.dev_mode


@pytest.mark.parametrize("env", [{"NANDI_FIELD_MODE": "1"}, {"NANDI_FIELD_MODE": "true"},
                                 {"NANDI_FIELD_MODE": "1", "NANDI_VIEWER_TOKEN": VW}])
def test_field_mode_without_operator_token_refuses_to_start(env):
    with pytest.raises(wa.AuthConfigError, match="NANDI_FIELD_MODE"):
        wa.AuthConfig.from_env(env)


def test_field_mode_zero_is_not_field_mode():
    assert not wa.AuthConfig.from_env({"NANDI_FIELD_MODE": "0"}).field_mode


def test_short_or_identical_tokens_are_rejected():
    with pytest.raises(wa.AuthConfigError):
        wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": "short"})
    with pytest.raises(wa.AuthConfigError, match="differ"):
        wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": OP, "NANDI_VIEWER_TOKEN": OP})


def test_viewer_token_alone_requires_auth():
    c = wa.AuthConfig.from_env({"NANDI_VIEWER_TOKEN": VW})
    assert c.auth_required
    assert "nobody can drive" in c.startup_warning("127.0.0.1")


def test_dev_mode_on_non_loopback_bind_warns_but_loopback_does_not():
    c = wa.AuthConfig.from_env({})
    assert "DEV: no auth" in c.startup_warning("0.0.0.0")
    assert c.startup_warning("127.0.0.1") is None


# ---------------------------------------------------------------- tokens

def test_role_for_tokens():
    c = cfg()
    assert c.role_for(OP) == wa.ROLE_OPERATOR
    assert c.role_for(VW) == wa.ROLE_VIEWER
    for bad in ("", None, "nope", OP + "x", OP[:-1], 123, "x" * 10_000):
        assert c.role_for(bad) is None


def test_role_for_uses_constant_time_compare(monkeypatch):
    calls = []
    real = wa.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(wa.hmac, "compare_digest", spy)
    assert cfg().role_for(VW) == wa.ROLE_VIEWER
    # both tokens are always compared (no early exit that would leak which one matched)
    assert len(calls) == 2
    assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)


def test_token_from_path():
    assert wa.token_from_path("/ws?token=abc123") == "abc123"
    assert wa.token_from_path("/ws?x=1&token=a%2Bb") == "a+b"
    assert wa.token_from_path("/ws") is None
    assert wa.token_from_path("") is None
    assert wa.token_from_path(None) is None


# ---------------------------------------------------------------- per-connection state

def test_dev_mode_client_is_authenticated_immediately():
    a = wa.ClientAuth(wa.AuthConfig(), now=0.0)
    assert a.authenticated and a.role == wa.ROLE_DEV
    assert not a.expired(100.0)


def test_handshake_token_authenticates():
    assert wa.ClientAuth(cfg(), 0.0, OP).role == wa.ROLE_OPERATOR
    assert wa.ClientAuth(cfg(), 0.0, VW).role == wa.ROLE_VIEWER
    assert wa.ClientAuth(cfg(), 0.0, "bad").role is None


def test_auth_message_within_deadline():
    a = wa.ClientAuth(cfg(), now=10.0)
    assert not a.authenticated and not a.expired(14.9)
    assert not a.try_token("wrong")
    assert a.try_token(VW) and a.role == wa.ROLE_VIEWER
    assert not a.expired(100.0)


def test_unauthenticated_client_expires_after_5_s():
    a = wa.ClientAuth(cfg(), now=10.0)
    assert wa.AUTH_TIMEOUT_S == 5.0
    assert not a.expired(14.99)
    assert a.expired(15.0)


# ---------------------------------------------------------------- role matrix

DRIVE_MESSAGES = [("cmd", None), ("service", "stop"), ("service", "auto_harvest"), ("service", "reset_world"),
                  ("service", "estop_reset"), ("service", "mission_abort"), ("manual_arm", None),
                  ("gait_params", None), ("envelope", None), ("mission_start", None)]


@pytest.mark.parametrize("kind,name", DRIVE_MESSAGES)
def test_viewer_cannot_drive(kind, name):
    ok, reason = wa.permit(wa.ROLE_VIEWER, kind, name)
    assert not ok and "viewer" in reason


@pytest.mark.parametrize("kind,name", DRIVE_MESSAGES)
def test_operator_and_dev_can_drive(kind, name):
    assert wa.permit(wa.ROLE_OPERATOR, kind, name)[0]
    assert wa.permit(wa.ROLE_DEV, kind, name)[0]


@pytest.mark.parametrize("role", [wa.ROLE_OPERATOR, wa.ROLE_VIEWER, wa.ROLE_DEV])
def test_estop_allowed_for_every_authenticated_role(role):
    assert wa.permit(role, "service", "estop") == (True, "")


@pytest.mark.parametrize("kind,name", DRIVE_MESSAGES + [("service", "estop")])
def test_unauthenticated_is_refused(kind, name):
    ok, reason = wa.permit(None, kind, name)
    assert not ok and "auth" in reason


def test_ping_and_auth_always_allowed():
    for role in (None, wa.ROLE_VIEWER, wa.ROLE_OPERATOR):
        assert wa.permit(role, "ping")[0] and wa.permit(role, "auth")[0]


# ---------------------------------------------------------------- audit log

def test_audit_log_appends_jsonl(tmp_path):
    path = tmp_path / "runs" / "console_audit.jsonl"
    log = wa.AuditLog(str(path))
    assert log.record(100.0, "c1@1.2.3.4:5", wa.ROLE_OPERATOR, "service", "auto_harvest")
    assert log.record(100.1, "c2@1.2.3.4:6", wa.ROLE_VIEWER, "service", "estop")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {"t": 100.0, "client": "c1@1.2.3.4:5", "role": "operator", "type": "service", "name": "auto_harvest"},
        {"t": 100.1, "client": "c2@1.2.3.4:6", "role": "viewer", "type": "service", "name": "estop"},
    ]


def test_audit_log_dedupes_held_refreshes_but_never_estop(tmp_path):
    path = tmp_path / "a.jsonl"
    log = wa.AuditLog(str(path))
    for i in range(10):  # 5 Hz held cmd for 2 s -> one line per second
        log.record(i * 0.2, "c1", "operator", "cmd")
    for i in range(3):
        log.record(5.0 + i * 0.1, "c1", "operator", "service", "estop")
    log.record(6.0, "c1", "operator", "ping")  # not a command
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in rows].count("cmd") == 2
    assert [r["name"] for r in rows].count("estop") == 3
    assert all(r["type"] != "ping" for r in rows)


def test_audit_log_survives_unwritable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = wa.AuditLog(str(blocker / "sub" / "audit.jsonl"))
    assert log.record(1.0, "c1", "operator", "cmd")  # accepted, write failure counted, no exception
    assert log.errors == 1


def test_audit_log_keeps_distinct_commands_within_window(tmp_path):
    path = tmp_path / "a.jsonl"
    log = wa.AuditLog(str(path))
    assert log.record(0.0, "c1", "operator", "cmd", None, [1.0, 0.0, 0.0, True])   # W
    assert not log.record(0.2, "c1", "operator", "cmd", None, [1.0, 0.0, 0.0, True])  # held refresh
    assert log.record(0.3, "c1", "operator", "cmd", None, [-1.0, 0.0, 0.0, True])  # S
    assert log.record(0.4, "c1", "operator", "cmd", None, [0.0, 0.0, 0.0, False])  # stop
    assert log.record(0.5, "c1", "operator", "manual_arm", None, [0.1] * 7)
    assert log.record(0.6, "c1", "operator", "manual_arm", None, [0.2] * 7)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["values"] for r in rows if r["type"] == "cmd"] == [
        [1.0, 0.0, 0.0, True], [-1.0, 0.0, 0.0, True], [0.0, 0.0, 0.0, False]]
    assert len([r for r in rows if r["type"] == "manual_arm"]) == 2


def test_field_mode_requires_longer_tokens():
    with pytest.raises(wa.AuthConfigError, match="16"):
        wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": "operatortok", "NANDI_FIELD_MODE": "1"})
    assert wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": "operatortok"}).operator_token == "operatortok"
    assert wa.AuthConfig.from_env({"NANDI_OPERATOR_TOKEN": "x" * 16, "NANDI_FIELD_MODE": "1"}).field_mode


def test_connection_locked_out_after_max_wrong_tokens():
    a = wa.ClientAuth(cfg(), now=0.0)
    for _ in range(wa.MAX_AUTH_ATTEMPTS):
        assert not a.locked_out
        assert not a.try_token("guess-guess")
    assert a.locked_out
    assert not a.try_token(OP) and a.role is None  # the right token no longer helps on this socket


def test_bad_handshake_token_counts_as_a_failure():
    a = wa.ClientAuth(cfg(), 0.0, "bad-token")
    assert a.failures == 1 and not a.authenticated


def test_dev_mode_never_locks_out():
    a = wa.ClientAuth(wa.AuthConfig(), now=0.0)
    for _ in range(10):
        assert a.try_token("anything")
    assert not a.locked_out


def test_auth_throttle_blocks_peer_until_window_passes():
    t = wa.AuthThrottle(limit=3, window_s=60.0)
    for i in range(2):
        t.record_failure("1.2.3.4", float(i))
    assert not t.blocked("1.2.3.4", 2.0)
    t.record_failure("1.2.3.4", 2.0)
    assert t.blocked("1.2.3.4", 3.0)
    assert not t.blocked("5.6.7.8", 3.0)
    assert not t.blocked("1.2.3.4", 62.5)  # oldest failures aged out
    assert not t.blocked(None, 3.0)


def test_auth_throttle_memory_is_bounded():
    t = wa.AuthThrottle(limit=2, window_s=60.0)
    for i in range(wa.AuthThrottle.MAX_PEERS + 50):
        t.record_failure(f"10.0.{i // 256}.{i % 256}", 0.0)
    assert len(t._fails) <= wa.AuthThrottle.MAX_PEERS
