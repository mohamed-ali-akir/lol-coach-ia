import json
import ssl

import httpx
import pytest

from lolcoach.client_lol import lcu as lcu_module
from lolcoach.client_lol import riot_ssl_context
from lolcoach.client_lol.lcu import LcuClient, LcuError, Lockfile, parse_lockfile, read_lockfile
from lolcoach.client_lol.replay import (
    Moment,
    ReplayClient,
    ReplayError,
    enable_replay_api,
    find_game_cfg,
    merge_moments,
    thin_frames,
)


def test_riot_certificate_is_bundled():
    context = riot_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    names = [dict(item[0] for item in c["subject"]).get("commonName")
             for c in context.get_ca_certs()]
    assert names == ["LoL Game Engineering Certificate Authority"]  # aucune autre racine
    assert not riot_ssl_context(check_hostname=False).check_hostname


def test_lockfile(tmp_path, monkeypatch):
    assert parse_lockfile("LeagueClient:1234:54321:s3cr3t:https\n") == Lockfile(
        1234, 54321, "s3cr3t", "https")
    for bad in ("", "a:b:c", "LeagueClient:x:54321:pw:https", "LeagueClient:1:99999:pw:https",
                "LeagueClient:1:2:pw:https:extra"):
        with pytest.raises(LcuError):
            parse_lockfile(bad)
    assert read_lockfile([tmp_path]) is None  # client fermé
    (tmp_path / "lockfile").write_text("LeagueClient:1:2999:pw:https", encoding="utf-8")
    monkeypatch.setenv("LOLCOACH_LOL_DIR", str(tmp_path))
    assert lcu_module.install_dirs()[0] == tmp_path
    assert read_lockfile().port == 2999


def lcu_with(routes, seen=None):
    def handler(request):
        if seen is not None:
            seen.append((request.method, request.url.path, request.content))
        key = (request.method, request.url.path)
        value = routes.get(key)
        if callable(value):
            value = value()
        if value is None:
            return httpx.Response(404)
        return httpx.Response(200, json=value)
    http = httpx.Client(base_url="https://127.0.0.1:1", transport=httpx.MockTransport(handler))
    return LcuClient(Lockfile(1, 1, "pw", "https"), http=http)


def test_end_of_game_and_phase():
    client = lcu_with({("GET", "/lol-gameflow/v1/gameflow-phase"): "EndOfGame",
                       ("GET", "/lol-end-of-game/v1/eog-stats-block"): {"gameId": 0},
                       ("GET", "/lol-gameflow/v1/session"): {"gameData": {"gameId": 777}}})
    assert client.gameflow_phase() == "EndOfGame"
    assert client.end_of_game_game_id() == 777  # repli sur la session
    assert lcu_with({}).end_of_game_game_id() is None


def test_replay_download_then_ready():
    states = iter(["download", "downloading", "downloading", "watch"])
    seen = []
    client = lcu_with({("GET", "/lol-replays/v1/metadata/42"): lambda: {"state": next(states)},
                       ("POST", "/lol-replays/v1/rofls/42/download/graceful"): {}}, seen)
    assert client.wait_replay_ready(42, sleep=lambda s: None)
    posts = [s for s in seen if s[0] == "POST"]
    assert len(posts) == 1 and json.loads(posts[0][2]) == {
        "componentType": lcu_module.REPLAY_COMPONENT}


def test_replay_expired():
    client = lcu_with({("GET", "/lol-replays/v1/metadata/42"): {"state": "incompatible"}})
    with pytest.raises(LcuError, match="patch"):
        client.wait_replay_ready(42, sleep=lambda s: None)


def test_lcu_unreachable():
    def handler(request):
        raise httpx.ConnectError("refusé")
    http = httpx.Client(base_url="https://127.0.0.1:1", transport=httpx.MockTransport(handler))
    with pytest.raises(LcuError, match="injoignable"):
        LcuClient(Lockfile(1, 1, "pw", "https"), http=http).gameflow_phase()


@pytest.mark.parametrize(("before", "after_has"), [
    ("[General]\nWindowMode=1\n[HUD]\nScale=1\n", "[General]\nWindowMode=1\nEnableReplayApi=1\n"
                                                  "[HUD]"),
    ("[General]\nEnableReplayApi=0\n", "EnableReplayApi=1"),
    ("[HUD]\nScale=1\n", "[General]\nEnableReplayApi=1\n[HUD]"),
    ("[HUD]\nScale=1\n[General]\nX=1\n", "[General]\nX=1\nEnableReplayApi=1"),
])
def test_enable_replay_api(tmp_path, before, after_has):
    cfg = tmp_path / "Config" / "game.cfg"
    cfg.parent.mkdir()
    cfg.write_text(before, encoding="utf-8")
    assert find_game_cfg(tmp_path) == cfg
    assert enable_replay_api(cfg)
    assert after_has in cfg.read_text(encoding="utf-8")
    assert (cfg.parent / "game.cfg.bak").read_text(encoding="utf-8") == before
    assert not enable_replay_api(cfg)  # déjà fait


def test_merge_and_thin():
    moments = [Moment(600_000, "mort", ("A",)), Moment(610_000, "dragon", ("B",)),
               Moment(100_000, "gank")] + [Moment(i * 100_000, "x") for i in range(10, 20)]
    merged = merge_moments(moments, limit=4)
    assert len(merged) == 4 and merged[0].ts_ms == 100_000
    first = next(m for m in merged if m.ts_ms == 600_000)
    assert first.codes == ("A", "B") and "dragon" in first.reason
    frames = [f"f{i}" for i in range(28)]
    thinned = thin_frames(frames, 12)
    assert len(thinned) == 12 and thinned[0] == "f0" and thinned[-1] == "f27"


def test_replay_capture(tmp_path):
    calls = []
    state = {"recording": False}

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/replay/recording":
            if request.method == "POST" and body.get("recording"):
                state["recording"] = True
                for i in range(3):  # le jeu écrit ses images
                    (tmp_path / "cap" / f"{i:04d}.png").write_bytes(b"png")
                return httpx.Response(200, json={"recording": True})
            was, state["recording"] = state["recording"], False
            return httpx.Response(200, json={"recording": was and False})
        if request.url.path == "/replay/playback" and request.method == "GET":
            return httpx.Response(200, json={"length": 1800.0, "time": 0.0})
        if request.url.path == "/replay/game":
            return httpx.Response(200, json={"processID": 4242})
        return httpx.Response(200, json={})

    http = httpx.Client(base_url="https://127.0.0.1:2999", transport=httpx.MockTransport(handler))
    with ReplayClient(http=http, sleep=lambda s: None) as replay:
        length = replay.wait_until_loaded()
        capture = replay.capture(Moment(600_000, "mort"), tmp_path / "cap", "LeeSin", length)
        assert replay.game_process_id() == 4242
    assert len(capture.frames) == 3
    record = next(c for c in calls if c[1] == "/replay/recording" and c[0] == "POST")[2]
    assert record["codec"] == "png" and record["startTime"] == 580.0
    assert record["endTime"] == 608.0 and record["framesPerSecond"] == 1
    render = next(c for c in calls if c[1] == "/replay/render")[2]
    assert render["selectionName"] == "LeeSin" and render["fogOfWar"] is True


def test_replay_not_open():
    def handler(request):
        raise httpx.ConnectError("refusé")
    http = httpx.Client(base_url="https://127.0.0.1:2999", transport=httpx.MockTransport(handler))
    clock = iter(range(0, 1000, 50))
    replay = ReplayClient(http=http, sleep=lambda s: None, clock=lambda: next(clock))
    with pytest.raises(ReplayError, match="EnableReplayApi"):
        replay.wait_until_loaded(timeout_s=100)
