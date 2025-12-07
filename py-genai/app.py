import os
import json
import requests
import httpx
import datetime
import logging
from flask import Flask, render_template, request, jsonify, make_response, url_for
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_swagger_ui import get_swaggerui_blueprint
from functools import lru_cache

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Configure cache
cache_config = {
    "CACHE_TYPE": "SimpleCache",  # Simple in-memory cache
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutes
}
cache = Cache(app, config=cache_config)

# Configure rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Configure Swagger UI
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Hello-GenAI API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

def configure_logging():
    """Configure application logging"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Configure Flask logger
    app.logger.setLevel(numeric_level)
    
    # Add a formatter to the handler
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    for handler in app.logger.handlers:
        handler.setFormatter(formatter)

def get_llm_endpoint():
    """Returns the complete LLM API endpoint URL"""
    base_url = os.getenv("LLM_BASE_URL", "")
    return f"{base_url}/api/chat"

def get_model_name():
    """Returns the model name to use for API requests"""
    return os.getenv("LLM_MODEL_NAME", "")

def get_mcp_server_url():
    """Returns the MCP server URL"""
    return os.getenv("MCP_SERVER_URL", "http://mcp-server:8084")

def check_weather_intent(message):
    """
    Use the LLM to determine if the message is a weather query and extract the city.
    Returns the city name if it's a weather query, otherwise None.
    """
    llm_endpoint = get_llm_endpoint()
    model_name = get_model_name()

    chat_request = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier for a chatbot. "
                    "Given a user message, respond ONLY with a JSON object in this format: "
                    "{\"intent\": \"weather\" or \"other\", \"city\": \"<city name or null>\"}. "
                    "If the user is asking about the weather, temperature, forecast, or climate in a location, "
                    "set \"intent\" to \"weather\" and extract the city name (or null if not found). "
                    "For all other messages, set \"intent\" to \"other\" and \"city\" to null. "
                    "Do not include any explanation or extra text."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            llm_endpoint,
            headers=headers,
            json=chat_request,
            timeout=30
        )
        if response.status_code != 200:
            app.logger.error(f"LLM intent check failed: {response.status_code} {response.text}")
            return None
        result = response.json()
        content = result.get("message", {}).get("content", "")
        try:
            # Try to extract the JSON from the LLM's response
            intent_data = json.loads(content)
        except Exception as e:
            app.logger.error(f"Failed to parse LLM intent response: {e}. Content: {content}")
            return None
        if intent_data.get("intent") == "weather" and intent_data.get("city"):
            city = intent_data["city"]
            app.logger.info(f"LLM detected weather intent for city: '{city}'")
            return city
        app.logger.info(f"LLM intent result: {intent_data}")
    except Exception as e:
        app.logger.error(f"Error calling LLM for intent detection: {e}", exc_info=True)
    return None

# Cache intent detection for repeated messages (in-memory, up to 128 unique messages)
@lru_cache(maxsize=128)
def cached_check_weather_intent(message):
    """Cached version of check_weather_intent to improve performance for repeated queries"""
    llm_endpoint = get_llm_endpoint()
    model_name = get_model_name()

    chat_request = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier for a chatbot. "
                    "Given a user message, respond ONLY with a JSON object in this format: "
                    "{\"intent\": \"weather\" or \"other\", \"city\": \"<city name or null>\"}. "
                    "If the user is asking about the weather, temperature, forecast, or climate in a location, "
                    "set \"intent\" to \"weather\" and extract the city name (or null if not found). "
                    "For all other messages, set \"intent\" to \"other\" and \"city\" to null. "
                    "Do not include any explanation or extra text."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            llm_endpoint,
            headers=headers,
            json=chat_request,
            timeout=30
        )
        if response.status_code != 200:
            app.logger.error(f"LLM intent check failed: {response.status_code} {response.text}")
            return None
        result = response.json()
        content = result.get("message", {}).get("content", "")
        try:
            # Try to extract the JSON from the LLM's response
            intent_data = json.loads(content)
        except Exception as e:
            app.logger.error(f"Failed to parse LLM intent response: {e}. Content: {content}")
            return None
        if intent_data.get("intent") == "weather" and intent_data.get("city"):
            city = intent_data["city"]
            app.logger.info(f"LLM detected weather intent for city: '{city}' (cached)")
            return city
        app.logger.info(f"LLM intent result: {intent_data} (cached)")
    except Exception as e:
        app.logger.error(f"Error calling LLM for intent detection: {e}", exc_info=True)
    return None

# Cache weather results for each city for 10 minutes (600 seconds)
@cache.memoize(timeout=600)
def cached_call_mcp_weather_tool(city):
    """Cached version of call_mcp_weather_tool to improve performance for repeated weather queries"""
    try:
        mcp_url = get_mcp_server_url()
        app.logger.info(f"Calling MCP server at {mcp_url} for city: {city} (cached)")

        # Use the implemented /mcp/call endpoint
        tool_response = requests.post(
            f"{mcp_url}/mcp/call",
            json={
                "name": "get_weather",
                "arguments": {"city": city}
            },
            timeout=60
        )

        app.logger.info(f"MCP tool call response status: {tool_response.status_code}")

        if tool_response.status_code == 200:
            result = tool_response.json()
            content = result.get("result", [])
            if content and len(content) > 0:
                text_content = content[0]
                app.logger.info(f"MCP server returned: {text_content}")
                return text_content
            app.logger.error("No content in MCP response")
            return None
        else:
            app.logger.error(f"MCP tool call failed: {tool_response.status_code}: {tool_response.text}")
            return None

    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout calling MCP server at {mcp_url}")
        return None
    except requests.exceptions.ConnectionError as e:
        app.logger.error(f"Connection error calling MCP server at {mcp_url}: {e}")
        return None
    except Exception as e:
        app.logger.error(f"Error calling MCP weather tool: {e}", exc_info=True)
        return None

def validate_environment():
    """Validates required environment variables and provides warnings"""
    llm_base_url = os.getenv("LLM_BASE_URL", "")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "")
    mcp_server_url = os.getenv("MCP_SERVER_URL", "")

    if not llm_base_url:
        app.logger.warning("LLM_BASE_URL is not set. API calls will fail.")
    
    if not llm_model_name:
        app.logger.warning("LLM_MODEL_NAME is not set. Using default model.")
    
    if not mcp_server_url:
        app.logger.warning("MCP_SERVER_URL is not set. Weather features will be disabled.")

    return llm_base_url and llm_model_name

@app.route('/')
def index():
    """Serves the chat web interface"""
    return render_template('index.html')

@app.route('/example')
def example():
    """Serves an example of structured formatting"""
    with open('static/examples/structured_response_example.md', 'r') as file:
        example_text = file.read()
    return jsonify({'response': example_text})

@app.route('/health')
def health_check():
    """Health check endpoint for container orchestration"""
    # Check if LLM API is accessible
    llm_status = "ok"
    try:
        # Simple check if the LLM endpoint is configured
        if not get_llm_endpoint():
            llm_status = "not_configured"
    except Exception as e:
        llm_status = "error"
        app.logger.error(f"Health check error: {e}")
    
    return jsonify({
        "status": "healthy",
        "llm_api": llm_status,
        "timestamp": datetime.datetime.now().isoformat()
    })

def validate_chat_request(data):
    """Validates and sanitizes chat API request data"""
    if not isinstance(data, dict):
        return False, "Invalid request format"
    
    message = data.get('message', '')
    if not message or not isinstance(message, str):
        return False, "Message is required and must be a string"
    
    if len(message) > 4000:  # Reasonable limit
        return False, "Message too long (max 4000 characters)"
    
    return True, message

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat_api():
    """Processes chat API requests"""
    data = request.json
    
    # Validate request
    valid, result = validate_chat_request(data)
    if not valid:
        return jsonify({'error': result}), 400
    
    message = result
    
    # Special command for getting model info
    if message == "!modelinfo":
        return jsonify({'model': get_model_name()})
    
    # Use cached intent detection
    city = cached_check_weather_intent(message)
    if city:
        app.logger.info(f"Detected weather query for city: {city}")
        weather_data = cached_call_mcp_weather_tool(city)

        if weather_data:
            try:
                # Check if weather_data is already a string or dict
                if isinstance(weather_data, str):
                    weather_json = json.loads(weather_data)
                else:
                    weather_json = weather_data

                # Format a nice response
                location = weather_json.get('location', city)
                weather = weather_json.get('current_weather', {})

                formatted_response = f"""# Weather for {location}

**Current Conditions:**
- 🌡️ Temperature: {weather.get('temperature', 'N/A')}
- 🤒 Feels Like: {weather.get('feels_like', 'N/A')}
- 💧 Humidity: {weather.get('humidity', 'N/A')}
- 💨 Wind Speed: {weather.get('wind_speed', 'N/A')}
- 🌧️ Precipitation: {weather.get('precipitation', 'N/A')}

---
*Data provided by MCP Weather Service*
"""
                app.logger.info(f"Successfully formatted weather response for {location}")
                return jsonify({'response': formatted_response, 'source': 'mcp'})
            except json.JSONDecodeError as e:
                app.logger.error(f"Failed to parse weather data from MCP server: {e}. Data: {weather_data}")
                return jsonify({'response': f"Sorry, I couldn't parse the weather data for {city}. Please try again.", 'source': 'error'}), 500
        else:
            app.logger.warning(f"No weather data received for city: {city}")
            return jsonify({'response': f"I couldn't find weather data for {city}. Please check the spelling and try again.", 'source': 'mcp'}), 200

    # Call the LLM API for non-weather queries
    try:
        response = call_llm_api(message)
        return jsonify({'response': response, 'source': 'llm'})
    except Exception as e:
        app.logger.error(f"Error calling LLM API: {e}")
        return jsonify({'error': 'Failed to get response from LLM'}), 500

@cache.memoize(timeout=500)
def call_llm_api(user_message):
    """Calls the LLM API and returns the response with caching"""
    chat_request = {
        "model": get_model_name(),
        "stream": False,
        "messages": [
            # {
            #     "role": "system",
            #     "content": "You are a helpful assistant. Please provide structured responses using markdown formatting. Use headers (# for main points), bullet points (- for lists), bold (**text**) for emphasis, and code blocks (```code```) for code examples. Organize your responses with clear sections and concise explanations."
            # },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }
    app.logger.info(f"Sending chat request to LLM API: {chat_request}")
    
    headers = {"Content-Type": "application/json"}
    
    # Send request to LLM API
    response = requests.post(
        get_llm_endpoint(),
        headers=headers,
        json=chat_request,
        timeout=60
    )
    app.logger.info(f"Chat API response: {response.content}")
    # Check if the status code is not 200 OK
    if response.status_code != 200:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")
    
    # Parse the response
    chat_response = response.json()
    app.logger.info(f"Chat API response: {chat_response}")
    
    # Extract the assistant's message
    if chat_response.get('message', {}).get('content', {}):
        return chat_response['message']['content'].strip()
    
    raise Exception("No response choices returned from API")

@app.after_request
def add_security_headers(response):
    """Add security headers to response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com"
    return response

if __name__ == '__main__':
    # Configure logging
    configure_logging()
    
    # Validate environment
    port = int(os.getenv("PORT", 8081))
    env_valid = validate_environment()
    
    if not env_valid:
        app.logger.warning("Environment not fully configured. Some features may not work.")
    
    app.logger.info(f"Server starting on http://localhost:{port}")
    app.logger.info(f"Using LLM endpoint: {get_llm_endpoint()}")
    app.logger.info(f"Using model: {get_model_name()}")
    app.logger.info(f"Using MCP server: {get_mcp_server_url()}")

    app.run(host='0.0.0.0', port=port, debug=os.getenv("DEBUG", "false").lower() == "true")
