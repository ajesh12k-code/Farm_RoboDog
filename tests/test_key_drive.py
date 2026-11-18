import math

import pytest

from atharv_nandi.core.key_drive import (
    DEBOUNCE_S, GAIT_DEFAULTS, GAIT_PARAMS, PAD_PREFIX, RELEASE, REPUBLISH_S, DriveCommand, KeyDrive,
    action_for_key, clamp_gait_params, combine_keys, drive_axis, normalize_key,
)


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clk():
    return FakeClock()


@pytest.fixture
def kd(clk):
    return KeyDrive(clock=clk)


# ---- pure mapping ------------------------------------------------------------------------------
@pytest.mark.parametrize("keys, expected", [
    ([], RELEASE),
    (["w"], DriveCommand(1.0, 0.0, 0.0, True)),
    (["s"], DriveCommand(-1.0, 0.0, 0.0, True)),
    (["a"], DriveCommand(0.0, 1.0, 0.0, True)),  # left is +y
    (["d"], DriveCommand(0.0, -1.0, 0.0, True)),
    (["q"], DriveCommand(0.0, 0.0, 1.0, True)),  # turn left = CCW = +yaw
    (["e"], DriveCommand(0.0, 0.0, -1.0, True)),
    (["w", "q"], DriveCommand(1.0, 0.0, 1.0, True)),  # arc left
    (["s", "d", "e"], DriveCommand(-1.0, -1.0, -1.0, True)),
    (["w", "s"], RELEASE),  # cancels -> release, never a zero-velocity hold
    (["q", "e"], RELEASE),
    (["w", "s", "a"], DriveCommand(0.0, 1.0, 0.0, True)),
    (["w", PAD_PREFIX + "w"], DriveCommand(1.0, 0.0, 0.0, True)),  # same direction twice stays at 1
    ([PAD_PREFIX + "e"], DriveCommand(0.0, 0.0, -1.0, True)),
    (["space", "h", "shift_l"], RELEASE),  # non-drive keys contribute nothing
])
def test_combine_keys(keys, expected):
    assert combine_keys(keys) == expected


def test_axes_stay_in_unit_range():
    for keys in (["w", "q", "a"], ["w", "pad:w", "w"], ["s", "pad:s"]):
        cmd = combine_keys(keys)
        assert all(-1.0 <= v <= 1.0 for v in (cmd.vx, cmd.vy, cmd.yaw))


def test_key_lookup_helpers():
    assert normalize_key("W") == "w"
    assert normalize_key("Escape") == "escape"
    assert normalize_key(None) == ""
    assert drive_axis("pad:a") == (0.0, 1.0, 0.0)
    assert drive_axis("x") is None
    assert action_for_key("space") == "stop"
    assert action_for_key("H") == "harvest"
    assert action_for_key("r") == "reset"
    assert action_for_key("g") == "gait"
    assert action_for_key("Escape") == "release"
    assert action_for_key("w") is None


# ---- press / release / debounce ----------------------------------------------------------------
def test_idle_publishes_nothing(kd, clk):
    assert kd.poll() is None
    clk.advance(5.0)
    assert kd.poll() is None


def test_press_publishes_hold_then_republishes_at_5hz(kd, clk):
    assert kd.press("w") is True
    assert kd.poll() == DriveCommand(1.0, 0.0, 0.0, True)
    clk.advance(0.05)
    assert kd.poll() is None  # unchanged and < 0.2 s since last send
    clk.advance(REPUBLISH_S - 0.05 + 1e-3)
    assert kd.poll() == DriveCommand(1.0, 0.0, 0.0, True)
    sent_at = [clk()]
    for _ in range(40):  # 2 s at a 50 ms GUI tick
        clk.advance(0.05)
        if kd.poll() is not None:
            sent_at.append(clk())
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert len(sent_at) >= 9  # ~5 Hz (4-5 Hz given the 50 ms tick quantization)
    assert all(REPUBLISH_S - 1e-9 <= g <= 0.25 + 1e-9 for g in gaps)  # well inside the 0.5 s dead-man


def test_x11_autorepeat_is_debounced(kd, clk):
    kd.press("w")
    kd.poll()
    for _ in range(10):  # autorepeat: release + press ~2 ms apart, every ~33 ms
        clk.advance(0.031)
        kd.release("w")
        clk.advance(0.002)
        assert kd.press("w") is False  # not a fresh press
        assert kd.command() == DriveCommand(1.0, 0.0, 0.0, True)
        out = kd.poll()
        assert out is None or out == DriveCommand(1.0, 0.0, 0.0, True)


def test_real_release_takes_effect_after_debounce(kd, clk):
    kd.press("w")
    kd.poll()
    clk.advance(0.1)
    kd.release("w")
    clk.advance(DEBOUNCE_S / 2)
    assert kd.command().hold is True  # still inside the debounce window
    clk.advance(DEBOUNCE_S)
    assert kd.poll() == RELEASE
    assert kd.poll() is None  # release is sent once, not repeated
    clk.advance(1.0)
    assert kd.poll() is None


def test_quick_tap_still_sends_hold_then_release(kd, clk):
    # GUI publishes right after a fresh drive press, so a 20 ms tap is never lost
    assert kd.press("e")
    assert kd.poll() == DriveCommand(0.0, 0.0, -1.0, True)
    clk.advance(0.02)
    kd.release("e")
    clk.advance(0.05)
    assert kd.poll() is None  # 70 ms: release still pending (deadline is 80 ms)
    clk.advance(0.05)
    assert kd.poll() == RELEASE


def test_combined_keys_publish_on_change(kd, clk):
    kd.press("w")
    assert kd.poll() == DriveCommand(1.0, 0.0, 0.0, True)
    clk.advance(0.01)
    kd.press("q")
    assert kd.poll() == DriveCommand(1.0, 0.0, 1.0, True)  # immediate, not after 0.2 s
    clk.advance(0.01)
    kd.release("w")
    clk.advance(DEBOUNCE_S + 0.01)
    assert kd.poll() == DriveCommand(0.0, 0.0, 1.0, True)


def test_uppercase_release_matches_lowercase_press(kd, clk):
    kd.press("w")
    kd.poll()
    kd.release("W")
    clk.advance(DEBOUNCE_S + 0.01)
    assert kd.poll() == RELEASE


def test_action_key_fires_once_while_held(kd, clk):
    assert kd.press("space") is True
    for _ in range(5):
        clk.advance(0.03)
        kd.release("space")
        clk.advance(0.001)
        assert kd.press("space") is False
    kd.release("space")
    clk.advance(DEBOUNCE_S + 0.01)
    assert kd.press("space") is True  # a new, real press
    assert kd.poll() is None  # action keys never move the robot


def test_stop_driving_keeps_action_keys_debounced(kd, clk):
    kd.press("w")
    kd.poll()
    kd.press("r")
    kd.stop_driving()
    assert kd.poll() == RELEASE
    clk.advance(0.03)
    kd.release("r")
    clk.advance(0.002)
    assert kd.press("r") is False  # autorepeat of R does not reset the world again
    assert kd.held_keys() == ["r"]


def test_clear_drops_everything(kd, clk):
    kd.press("w")
    kd.press("h")
    kd.poll()
    kd.clear()
    assert kd.held_keys() == []
    assert kd.poll() == RELEASE
    assert kd.press("h") is True


def test_pad_release_is_immediate_and_independent_of_keyboard(kd, clk):
    kd.press(PAD_PREFIX + "a")
    assert kd.poll() == DriveCommand(0.0, 1.0, 0.0, True)
    kd.release(PAD_PREFIX + "a", immediate=True)
    assert kd.poll() == RELEASE  # mouse release needs no debounce
    kd.press("w")
    kd.press(PAD_PREFIX + "w")
    kd.poll()
    kd.release(PAD_PREFIX + "w", immediate=True)
    assert kd.command() == DriveCommand(1.0, 0.0, 0.0, True)  # keyboard W still held


def test_release_of_unknown_key_is_ignored(kd, clk):
    kd.release("w")
    kd.release("")
    assert kd.press("") is False
    assert kd.poll() is None


# ---- gait params -------------------------------------------------------------------------------
def test_gait_defaults_match_contract():
    assert GAIT_DEFAULTS == [1.0, 1.0, 1.8, 1.0]
    assert [(p.lo, p.hi) for p in GAIT_PARAMS] == [(0.2, 1.5), (0.5, 2.0), (0.8, 3.0), (0.0, 2.0)]


@pytest.mark.parametrize("values, expected", [
    ([1.0, 1.0, 1.8, 1.0], [1.0, 1.0, 1.8, 1.0]),
    ([9, -9, 0.1, 5], [1.5, 0.5, 0.8, 2.0]),
    ([0.5], [0.5, 1.0, 1.8, 1.0]),
    (None, GAIT_DEFAULTS),
    (["x", None, math.nan, "1.2"], [1.0, 1.0, 1.8, 1.2]),
])
def test_clamp_gait_params(values, expected):
    assert clamp_gait_params(values) == pytest.approx(expected)
