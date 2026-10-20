"""Meteo del giorno da Open-Meteo (gratuito, nessuna chiave)."""
import os
from datetime import datetime

import requests

URL = "https://api.open-meteo.com/v1/forecast"
WMO = {0: "sereno", 1: "prevalentemente sereno", 2: "parzialmente nuvoloso", 3: "coperto",
       45: "nebbia", 48: "nebbia", 51: "pioviggine", 53: "pioviggine", 55: "pioviggine",
       61: "pioggia debole", 63: "pioggia", 65: "pioggia forte", 71: "neve", 73: "neve", 75: "neve",
       80: "rovesci", 81: "rovesci", 82: "rovesci forti", 95: "temporale", 96: "temporale", 99: "temporale"}
IDEAL_TEMP = 17        # °C percepiti: comodo per corsa/camminata
RAIN_LIKELY = 50       # % probabilità
UV_HIGH = 6


RETRIES = 3            # tentativi totali: i 503 di Open-Meteo sono quasi sempre momentanei
RETRY_WAIT = 3         # secondi prima del secondo tentativo, poi raddoppia


def fetch_forecast(day: str, _get=None, _sleep=None) -> dict:
    import time
    get, sleep = _get or requests.get, _sleep or time.sleep
    params = {
        "latitude": os.getenv("WEATHER_LAT", "41.89"), "longitude": os.getenv("WEATHER_LON", "12.49"),
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,uv_index",
        "daily": "temperature_2m_min,temperature_2m_max,weather_code,precipitation_probability_max,sunrise,sunset",
        "timezone": "Europe/Rome", "start_date": day, "end_date": day,
    }
    for attempt in range(RETRIES):
        try:
            r = get(URL, params=params, timeout=10)
            if r.status_code >= 500 and attempt < RETRIES - 1:
                sleep(RETRY_WAIT * 2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == RETRIES - 1:
                raise
            sleep(RETRY_WAIT * 2 ** attempt)


def _ranges(hours: list[int]) -> list[tuple[int, int]]:
    """[16,17,18,21] -> [(16,19),(21,22)]"""
    out = []
    for h in sorted(hours):
        if out and h == out[-1][1]:
            out[-1] = (out[-1][0], h + 1)
        else:
            out.append((h, h + 1))
    return out


def summarize(fc: dict) -> dict:
    d, h = fc["daily"], fc["hourly"]
    sunrise = datetime.fromisoformat(d["sunrise"][0])
    sunset = datetime.fromisoformat(d["sunset"][0])
    hours = [{"h": datetime.fromisoformat(t).hour, "temp": h["apparent_temperature"][i],
              "rain": h["precipitation_probability"][i] or 0, "uv": h["uv_index"][i] or 0}
             for i, t in enumerate(h["time"])]

    def cost(x):  # più basso = meglio per uscire
        dark = 0 if sunrise.hour <= x["h"] < sunset.hour else 3
        return x["rain"] / 10 + abs(x["temp"] - IDEAL_TEMP) / 2 + max(0, x["uv"] - 5) + dark

    candidates = [(sum(cost(x) for x in hours[s:s + 2]), s) for s in range(7, 21)]
    _, start = min(candidates)
    block = hours[start:start + 2]

    return {
        "tmin": d["temperature_2m_min"][0], "tmax": d["temperature_2m_max"][0],
        "sky": WMO.get(d["weather_code"][0], "variabile"),
        "rain_max": d["precipitation_probability_max"][0] or 0,
        "sunrise": sunrise.strftime("%H:%M"), "sunset": sunset.strftime("%H:%M"),
        "daylight": sunset - sunrise,
        "best": {"start": start, "end": start + 2, "dry": max(x["rain"] for x in block) < RAIN_LIKELY,
                 "temp": round(sum(x["temp"] for x in block) / 2), "rain": max(x["rain"] for x in block)},
        "rain_ranges": _ranges([x["h"] for x in hours if x["rain"] >= RAIN_LIKELY and 7 <= x["h"] <= 23]),
        "uv_ranges": _ranges([x["h"] for x in hours if x["uv"] >= UV_HIGH]),
        "evening_rain": any(x["rain"] >= RAIN_LIKELY for x in hours if 18 <= x["h"] <= 23),
    }
