import os
import logging
from fastapi import FastAPI
import uvicorn
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Server")
mcp_server = Server("weather-assistant")

# Weather API configuration (using Open-Meteo, which is free and doesn't require API key)
WEATHER_API_BASE = "https://api.open-meteo.com/v1"
GEOCODING_API_BASE = "https://geocoding-api.open-meteo.com/v1"

# Create a persistent HTTP client with connection pooling
http_client = None

async def get_http_client():
    """Get or create HTTP client with connection pooling"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    return http_client

async def get_coordinates(city: str) -> dict:
    """Get latitude and longitude for a city."""
    try:
        client = await get_http_client()
        # Clean up the city name
        city_clean = city.strip()
        logger.info(f"Fetching coordinates for city: '{city_clean}'")
        response = await client.get(
            f"{GEOCODING_API_BASE}/search",
            params={"name": city_clean, "count": 5, "language": "en", "format": "json"}
        )
        logger.info(f"Geocoding API response status: {response.status_code}")
        data = response.json()

        if data.get("results"):
            # Log all results found
            logger.info(f"Found {len(data['results'])} results for '{city_clean}'")
            for idx, res in enumerate(data["results"]):
                logger.info(f"  Result {idx + 1}: {res.get('name')}, {res.get('country')} (admin1: {res.get('admin1')})")

            result = data["results"][0]
            location_info = {
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "name": result["name"],
                "country": result.get("country", "")
            }
            logger.info(f"Selected location: {location_info}")
            return location_info

        logger.warning(f"No results found for city: '{city_clean}'")
        logger.info(f"Response data: {data}")
        return None
    except Exception as e:
        logger.error(f"Error fetching coordinates: {e}", exc_info=True)
        raise

async def get_weather(latitude: float, longitude: float) -> dict:
    """Get weather data for given coordinates."""
    try:
        client = await get_http_client()
        logger.info(f"Fetching weather for coordinates: {latitude}, {longitude}")
        response = await client.get(
            f"{WEATHER_API_BASE}/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch"
            }
        )
        logger.info(f"Weather API response status: {response.status_code}")
        data = response.json()
        logger.info(f"Weather data retrieved successfully")
        return data
    except Exception as e:
        logger.error(f"Error fetching weather: {e}", exc_info=True)
        raise

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather information for a city. Provides temperature, humidity, wind speed, and precipitation data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for (e.g., 'London', 'New York')"
                    }
                },
                "required": ["city"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_weather":
        city = arguments.get("city")
        if not city:
            error_msg = "Error: City name is required"
            logger.error(error_msg)
            return [TextContent(type="text", text=error_msg)]

        try:
            logger.info(f"Processing weather request for: '{city}'")

            # Get coordinates for the city
            location = await get_coordinates(city)
            if not location:
                error_msg = f"Error: Could not find coordinates for '{city}'. Please check the city name and try again (e.g., 'Austin' instead of 'Austin TX')."
                logger.warning(error_msg)
                return [TextContent(
                    type="text",
                    text=error_msg
                )]

            # Get weather data
            weather_data = await get_weather(location["latitude"], location["longitude"])

            current = weather_data.get("current", {})

            # Format the response
            result = {
                "location": f"{location['name']}, {location['country']}",
                "coordinates": {
                    "latitude": location["latitude"],
                    "longitude": location["longitude"]
                },
                "current_weather": {
                    "temperature": f"{current.get('temperature_2m')}°F",
                    "feels_like": f"{current.get('apparent_temperature')}°F",
                    "humidity": f"{current.get('relative_humidity_2m')}%",
                    "wind_speed": f"{current.get('wind_speed_10m')} mph",
                    "precipitation": f"{current.get('precipitation')} inch"
                }
            }

            logger.info(f"Successfully retrieved weather for {city}")
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        except Exception as e:
            error_msg = f"Error: Failed to get weather data. {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [TextContent(
                type="text",
                text=error_msg
            )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "MCP Server is running with weather assistance"}

@app.post("/mcp/call")
async def mcp_call(request: dict):
    """Handle MCP tool calls."""
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})
        logger.info(f"MCP call received: {tool_name} with arguments: {arguments}")
        result = await call_tool(tool_name, arguments)
        return {"result": [content.text for content in result]}
    except Exception as e:
        logger.error(f"Error in MCP call: {e}", exc_info=True)
        return {"error": str(e)}

@app.get("/mcp/tools")
async def mcp_tools():
    """List available MCP tools."""
    tools = await list_tools()
    return {"tools": [{"name": t.name, "description": t.description} for t in tools]}

async def shutdown_event():
    """Cleanup on shutdown"""
    global http_client
    if http_client:
        await http_client.aclose()

app.add_event_handler("shutdown", shutdown_event)

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8084"))

    logger.info(f"Starting MCP server on {host}:{port}")
    logger.info("Weather assistance tool enabled")
    logger.info(f"Weather API Base: {WEATHER_API_BASE}")
    logger.info(f"Geocoding API Base: {GEOCODING_API_BASE}")
    uvicorn.run(app, host=host, port=port)
