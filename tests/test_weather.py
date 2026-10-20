"""Open-Meteo: i 503 momentanei non devono far perdere il meteo del giorno."""
import pytest
import requests

from health_agent import weather


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return self._body


def test_retries_503_then_ok():
    seq = [Resp(503), Resp(503), Resp(200, {"ok": 1})]
    waits = []
    out = weather.fetch_forecast("2026-10-17", _get=lambda *a, **k: seq.pop(0), _sleep=waits.append)
    assert out == {"ok": 1} and waits == [3, 6]


def test_gives_up_after_three_503():
    seq = [Resp(503), Resp(503), Resp(503)]
    with pytest.raises(requests.HTTPError):
        weather.fetch_forecast("2026-10-17", _get=lambda *a, **k: seq.pop(0), _sleep=lambda s: None)


def test_client_error_is_not_retried():
    calls = []

    def get(*a, **k):
        calls.append(1)
        return Resp(400)

    with pytest.raises(requests.HTTPError):
        weather.fetch_forecast("2026-10-17", _get=get, _sleep=lambda s: None)
    assert len(calls) == 1


def test_connection_error_retried():
    seq = [requests.ConnectionError("giù"), Resp(200, {"ok": 2})]

    def get(*a, **k):
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    assert weather.fetch_forecast("2026-10-17", _get=get, _sleep=lambda s: None) == {"ok": 2}
