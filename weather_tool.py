import requests
from geopy.geocoders import Nominatim

# ----------------------
# Why a separate file?
# Each tool should live in its own module.
# Your agent.py stays clean — it just calls get_weather().
# If you want to swap weather providers later, you only touch this file.
# This is the "single responsibility principle" in practice.
# ----------------------

def get_coordinates(city: str) -> tuple:
    """
    Converts a city name into latitude and longitude.
    
    Why do we need this?
    Weather APIs don't accept "Dubai" — they need numbers like (25.2048, 55.2708).
    Geocoding is the process of converting human names into map coordinates.
    """
    geolocator = Nominatim(user_agent="agentic_ai_weather")
    # user_agent is required by Nominatim — it identifies your app.
    # Use any string here, just make it descriptive.

    location = geolocator.geocode(city)
    # geocode() sends the city name to OpenStreetMap's database
    # and returns a location object with coordinates

    if location is None:
        raise ValueError(f"Could not find coordinates for city: {city}")

    return location.latitude, location.longitude


def get_weather(city: str) -> dict:
    """
    Gets real weather forecast for a city using Open-Meteo API.
    Returns temperature, weather condition, wind speed, and humidity.
    """

    # Step 1: get coordinates
    try:
        lat, lon = get_coordinates(city)
    except ValueError as e:
        return {"error": str(e)}

    print(f"   📍 Coordinates for {city}: {lat:.4f}, {lon:.4f}")

    # Step 2: call Open-Meteo with coordinates
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",       # temperature at 2 meters height
            "relative_humidity_2m", # humidity percentage
            "wind_speed_10m",       # wind speed at 10 meters
            "weather_code"          # WMO weather code (we'll decode this)
        ],
        "timezone": "auto",
        # "auto" detects the timezone from coordinates automatically
        "forecast_days": 1
    }

    response = requests.get(url, params=params)
    # GET request — we're asking for data, not sending any
    # params are automatically appended to the URL as query string

    if response.status_code != 200:
        return {"error": f"Weather API returned status {response.status_code}"}

    data = response.json()
    current = data["current"]
    # The API returns a nested structure — we drill into "current" for live data

    # Decode the WMO weather code into a human-readable condition
    condition = decode_weather_code(current["weather_code"])

    return {
        "city": city,
        "temperature_c": current["temperature_2m"],
        "condition": condition,
        "humidity_percent": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"]
    }


def decode_weather_code(code: int) -> str:
    """
    WMO Weather Interpretation Codes → plain English.
    Open-Meteo uses standard WMO codes (World Meteorological Organization).
    Without this, the API returns a number like 61 which means nothing to a user.
    """
    codes = {
        0:  "Clear sky",
        1:  "Mainly clear",
        2:  "Partly cloudy",
        3:  "Overcast",
        45: "Foggy",
        48: "Icy fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight showers",
        81: "Moderate showers",
        82: "Violent showers",
        95: "Thunderstorm",
        99: "Thunderstorm with hail"
    }
    return codes.get(code, f"Unknown condition (code {code})")