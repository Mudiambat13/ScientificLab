import json
import threading
import urllib.error
import urllib.request

import pytest

from scientificlab.gui.server import MAX_BODY, create_server, main


@pytest.fixture
def server():
    srv = create_server(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]
    yield srv, f"http://127.0.0.1:{port}", port
    srv.shutdown()
    srv.server_close()


def call(url, *, data=None, headers=None, method=None):
    request = urllib.request.Request(
        url, data=data, headers=headers or {}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def post(base, srv, name, payload, **extra):
    headers = {"Content-Type": "application/json", "X-Token": srv.token, **extra}
    return call(
        f"{base}/api/{name}", data=json.dumps(payload).encode(), headers=headers
    )


def test_binds_to_loopback_only(server):
    srv, _, _ = server
    assert srv.server_address[0] == "127.0.0.1"


def test_page_is_served_with_token_and_security_headers(server):
    srv, base, _ = server
    status, body, headers = call(base + "/")
    page = body.decode()
    assert status == 200
    assert srv.token in page
    assert "__SCIENTIFICLAB_TOKEN__" not in page
    assert "__SCIENTIFICLAB_VERSION__" not in page
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"


def test_unknown_path_is_404(server):
    _, base, _ = server
    assert call(base + "/secret")[0] == 404


def test_api_roundtrip(server):
    srv, base, _ = server
    status, body, _ = post(base, srv, "convert", {"value": 1, "from": "km", "to": "m"})
    assert status == 200 and json.loads(body)["value"] == 1000.0


def test_library_errors_are_400_with_message(server):
    srv, base, _ = server
    status, body, _ = post(base, srv, "convert", {"value": 1, "from": "m", "to": "s"})
    assert status == 400
    assert json.loads(body)["type"] == "IncompatibleUnitsError"


def test_missing_or_wrong_token_is_refused(server):
    _srv, base, _ = server
    headers = {"Content-Type": "application/json"}
    assert call(base + "/api/constants", data=b"{}", headers=headers)[0] == 403
    headers["X-Token"] = "wrong"
    assert call(base + "/api/constants", data=b"{}", headers=headers)[0] == 403


def test_foreign_host_header_is_refused(server):
    srv, base, _ = server
    status, _, _ = post(base, srv, "constants", {}, Host="evil.example:80")
    assert status == 403
    assert call(base + "/", headers={"Host": "evil.example"})[0] == 403


def test_foreign_origin_is_refused(server):
    srv, base, port = server
    assert post(base, srv, "constants", {}, Origin="https://evil.example")[0] == 403
    assert post(base, srv, "constants", {}, Origin=f"http://127.0.0.1:{port}")[0] == 200


def test_content_type_and_body_validation(server):
    srv, base, _ = server
    token = {"X-Token": srv.token}
    status, _, _ = call(
        base + "/api/constants",
        data=b"{}",
        headers={**token, "Content-Type": "text/plain"},
    )
    assert status == 415
    status, _, _ = call(
        base + "/api/constants",
        data=b"not json",
        headers={**token, "Content-Type": "application/json"},
    )
    assert status == 400
    big = b" " * (MAX_BODY + 1)
    status, _, _ = call(
        base + "/api/constants",
        data=big,
        headers={**token, "Content-Type": "application/json"},
    )
    assert status == 413
    assert post(base, srv, "nope", {})[0] == 404


def test_unexpected_bug_does_not_leak_internals(server, monkeypatch):
    from scientificlab.gui import server as server_module

    srv, base, _ = server

    def boom(_params):
        raise RuntimeError("secret path /etc/passwd")

    monkeypatch.setitem(server_module.HANDLERS, "boom", boom)
    status, body, _ = post(base, srv, "boom", {})
    assert status == 500
    assert b"secret" not in body and b"passwd" not in body


def test_tokens_differ_between_servers():
    a, b = create_server(0), create_server(0)
    try:
        assert a.token != b.token and len(a.token) >= 40
    finally:
        a.server_close()
        b.server_close()


def test_cli_rejects_bad_port():
    with pytest.raises(SystemExit):
        main(["--port", "70000"])


def test_serve_runs_until_interrupted(monkeypatch, capsys):
    from scientificlab.gui import server as server_module

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module._Server, "serve_forever", interrupt)
    opened = []
    monkeypatch.setattr(server_module.webbrowser, "open", opened.append)
    server_module.serve(0, open_browser=False)
    out = capsys.readouterr().out
    assert "http://127.0.0.1:" in out and "Stopped" in out
    assert opened == []


def test_main_passes_options(monkeypatch):
    from scientificlab.gui import server as server_module

    calls = []
    monkeypatch.setattr(
        server_module,
        "serve",
        lambda port, *, open_browser: calls.append((port, open_browser)),
    )
    main(["--port", "8123", "--no-browser"])
    main([])
    assert calls == [(8123, False), (0, True)]


def test_python_dash_m_entry_point(monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["scientificlab-gui", "--help"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("scientificlab.gui", run_name="__main__")
    assert info.value.code == 0
