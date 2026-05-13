from __future__ import annotations

from typing import Any

import requests

from weather_rag.observability import Observer, Timer


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

CITY_ALIASES = {
    "bangaluru": "Bangalore",
    "bengaluruu": "Bangalore",
    "benagluru": "Bangalore",
    "new delhi india": "New Delhi",
}

FAKE_WEATHER = {
    "hyderabad": {"name": "Hyderabad", "country": "India", "temperature": 31.2, "windspeed": 9.4},
    "new delhi": {"name": "New Delhi", "country": "India", "temperature": 29.7, "windspeed": 11.1},
    "bangalore": {"name": "Bangalore", "country": "India", "temperature": 24.6, "windspeed": 7.3},
    "bengaluru": {"name": "Bengaluru", "country": "India", "temperature": 24.6, "windspeed": 7.3},
}


class WeatherTool:
    name = "get_current_weather"
    description = "Fetches current weather for a city through Open-Meteo geocoding and forecast APIs."

    def __init__(self, observer: Observer | None = None, *, fake: bool = False) -> None:
        self.observer = observer
        self.fake = fake

    def __call__(self, city: str) -> str:
        city = clean_city(city)
        if self.fake:
            return self._fake_weather(city)

        geocode = self._geocode(city)
        retry_note = ""
        if geocode.get("error") == "not_found":
            alias = CITY_ALIASES.get(city.lower())
            if alias:
                retry_note = f" Retried with '{alias}' after '{city}' was not found."
                geocode = self._geocode(alias)

        if geocode.get("error"):
            return geocode["message"]

        weather = self._weather(geocode["latitude"], geocode["longitude"])
        if weather.get("error"):
            return weather["message"]

        current = weather["current_weather"]
        return (
            f"Current weather in {geocode['name']}, {geocode.get('country', '')}: "
            f"Temperature {current['temperature']}°C, "
            f"wind speed {current['windspeed']} km/h.{retry_note}"
        ).strip()

    def _fake_weather(self, city: str) -> str:
        alias = CITY_ALIASES.get(city.lower())
        lookup = alias or city
        data = FAKE_WEATHER.get(lookup.lower())
        if not data:
            return f"Error: City '{city}' not found. Check spelling or try a nearby major city."
        retry_note = f" Retried with '{alias}' after '{city}' was not found." if alias else ""
        return (
            f"Current weather in {data['name']}, {data['country']}: "
            f"Temperature {data['temperature']}°C, "
            f"wind speed {data['windspeed']} km/h.{retry_note}"
        ).strip()

    def _geocode(self, city: str) -> dict[str, Any]:
        timer = Timer.start()
        try:
            response = requests.get(
                GEOCODING_URL,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            self._log_api("open_meteo_geocoding", False, timer.ms(), {"city": city}, "timeout")
            return {
                "error": "timeout",
                "message": f"Error: Geocoding service timed out for '{city}'. Try again later.",
            }
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            self._log_api("open_meteo_geocoding", False, timer.ms(), {"city": city}, f"HTTP {status}")
            return {"error": "http", "message": f"Error: Geocoding API returned HTTP {status}."}
        except requests.RequestException as exc:
            self._log_api("open_meteo_geocoding", False, timer.ms(), {"city": city}, str(exc))
            return {"error": "network", "message": f"Error: Could not reach geocoding service. ({exc})"}

        self._log_api("open_meteo_geocoding", True, timer.ms(), {"city": city}, None)
        results = data.get("results") or []
        if not results:
            return {
                "error": "not_found",
                "message": f"Error: City '{city}' not found. Check spelling or try a nearby major city.",
            }
        first = results[0]
        return {
            "latitude": first["latitude"],
            "longitude": first["longitude"],
            "name": first.get("name", city),
            "country": first.get("country", ""),
        }

    def _weather(self, latitude: float, longitude: float) -> dict[str, Any]:
        timer = Timer.start()
        params = {"latitude": latitude, "longitude": longitude, "current_weather": "true"}
        try:
            response = requests.get(WEATHER_URL, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            self._log_api("open_meteo_forecast", False, timer.ms(), params, "timeout")
            return {"error": "timeout", "message": "Error: Weather service timed out. Try again later."}
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            self._log_api("open_meteo_forecast", False, timer.ms(), params, f"HTTP {status}")
            return {"error": "http", "message": f"Error: Weather API returned HTTP {status}."}
        except requests.RequestException as exc:
            self._log_api("open_meteo_forecast", False, timer.ms(), params, str(exc))
            return {"error": "network", "message": f"Error: Could not reach weather service. ({exc})"}

        self._log_api("open_meteo_forecast", True, timer.ms(), params, None)
        if "current_weather" not in data:
            return {"error": "shape", "message": "Error: Unexpected response from weather service."}
        return {"current_weather": data["current_weather"]}

    def _log_api(
        self,
        name: str,
        success: bool,
        latency_ms: float,
        args: dict[str, Any],
        error: str | None,
    ) -> None:
        if self.observer:
            self.observer.log(
                "api_call",
                name=name,
                args=args,
                latency_ms=latency_ms,
                success=success,
                error=error,
            )


def clean_city(city: str) -> str:
    return " ".join(str(city).strip().strip("'\"").split())
