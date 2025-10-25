import streamlit as st
import requests
import json
import os

# --- FIX FOR ImportError: cannot import name 'genai' from 'google' ---
# The correct library must be installed. Please run this command in your terminal:
# pip install google-genai
from google import genai
from google.genai import types

# ------------------- API Key Setup -------------------
# The API key must be set in Streamlit secrets.toml as GEMINI_API_KEY
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 Configuration Error: GEMINI_API_KEY not found in Streamlit secrets.")
    st.info("Please add it to your secrets.toml file and ensure you have run: pip install google-genai")
    st.stop()

# --- REVERTED: Swapping Google Places back to Yelp API ---
try:
    YELP_API_KEY = st.secrets["YELP_API_KEY"]
except KeyError:
    st.error("🚨 Configuration Error: YELP_API_KEY not found in Streamlit secrets.")
    st.info("To use the Yelp search, please add your YELP_API_KEY to your secrets.toml file.")
    YELP_API_KEY = None  # Set to None if not found

# Initialize the Gemini Client
try:
    # Use the retrieved secret key
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()


# ------------------- Gemini Conversational Response -------------------

def generate_chat_response(user_input, chat_history):
    """
    Generates a conversational response for non-search queries using a standard
    chat model, leveraging the conversation history for context.
    """
    history_for_gemini = []

    # Format the Streamlit history for the Gemini API
    for role, text, in chat_history:
        # Clean up the AI's displayed text before feeding it back into the model's history
        clean_text = text.replace('🤖', '').strip()

        if role == 'user':
            # FIX: Use types.Part(text=...) instead of Part.from_text(...) to resolve TypeError
            history_for_gemini.append(types.Content(role='user', parts=[types.Part(text=text)]))
        elif role == 'ai_result':
            # FIX: Use types.Part(text=...) instead of Part.from_text(...) to resolve TypeError
            # Only include the AI's previous conversational replies for context
            history_for_gemini.append(types.Content(role='model', parts=[types.Part(text=clean_text)]))

    system_instruction = (
        "You are a friendly and enthusiastic AI Restaurant Finder, now using Yelp for search. "
        "Your primary goal is to help the user find food and location. When the user doesn't specify a city or food, "
        "you must gently prompt them to provide that information to continue the search. Keep your responses short and welcoming, and always "
        "steer the conversation back toward a search query (City and Food). If the user just says 'hi' or 'hello', "
        "greet them warmly and ask for their desired food and location."
    )

    try:
        chat_session = gemini_client.chats.create(
            model='gemini-2.5-flash',
            history=history_for_gemini,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

        response = chat_session.send_message(user_input)
        return response.text

    except Exception as e:
        # Log the detailed error but return a generic user-friendly message
        st.error(f"Gemini Chat Error: {e}")
        return "I seem to be having trouble connecting to the chat service right now. Please try a food and location search."


# ------------------- Gemini Informational Response -------------------

def generate_info_response(query, city=None):
    """
    Generates a response for informational queries (suggestions, culture, etc.).
    Uses the current query and extracted city for context.
    """

    city_context = f" in {city}" if city else ""

    # Updated prompt to be more specific about the location
    prompt = f"The user asked about popular food{city_context} with the query: '{query}'. Provide a concise, engaging response about the cuisine of {city if city else 'the region mentioned in the query'}."

    system_instruction = (
        "You are an expert culinary guide and local food culture specialist. "
        "Your task is to provide engaging and accurate suggestions about popular foods, cultural dishes, or dining tips "
        "based on the user's query and location context. Use bullet points for easy reading if suggesting multiple dishes. "
        "Do not offer to perform a Yelp search, redirect the topic, or suggest a different cuisine. Your response must be solely about the food and culture of the location specified in the query."
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text

    except Exception as e:
        st.error(f"Gemini Info Generation Error: {e}")
        return "I'm having trouble retrieving culinary information right now. Please try a specific food or location search."


# ------------------- Gemini Structured Extraction & Intent Classification -------------------

def extract_structured_info(user_input):
    """
    Uses the Gemini API to reliably extract the city, food type, and classify the intent
    from the user's natural language input, forcing a JSON output structure.
    """
    extraction_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING,
                description="The geographic location (city and country/state, e.g., 'New York, USA'). Defaults to null if not found."
            ),
            "food": types.Schema(
                type=types.Type.STRING,
                description="The specific type of cuisine or dish (e.g., 'sushi', 'pizza', 'vegan'). Defaults to null if not found."
            ),
            "intent": types.Schema(
                type=types.Type.STRING,
                enum=["SEARCH", "INFO", "CHAT"],
                description="Classify the user's primary goal: SEARCH (finding a specific restaurant/food listing), INFO (asking for suggestions, popular foods, or culture, it has to be a full question if its not then its a search intent), or CHAT (greeting/general talk)"
            ),
        }
    )

    # UPDATED: Enforce translation of colloquial/transliterated language before extraction AND add intent classification
    system_instruction = (
        "You are an expert entity extraction and intent classification system. "
        "Analyze the user's request. "
        "If the input contains transliterated or colloquial language (like 'batata maslou2a' or Franco-Arabic), first translate the food item into formal English (e.g., 'Boiled potato'). "
        "Then, extract the City/Location and the desired Food/Cuisine. "
        "Classify the user's primary intent as 'SEARCH', 'INFO', or 'CHAT'. "
        "Return the result strictly in the provided JSON schema. "
        "IMPORTANT: You MUST extract the location/city if one is mentioned, even if the user is asking generally about the food of that location (e.g., 'what about New York'). "
        "If the food term is highly generic (e.g., 'food', 'restaurant', 'meal') or a greeting, return null for the 'food' field. "
        "If a city is not explicitly mentioned, return null for 'city'."
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=extraction_schema
            ),
        )

        data = json.loads(response.text)

        # 1. Get and strip the strings
        city = data.get('city', '').strip()
        food = data.get('food', '').strip()
        intent = data.get('intent', 'CHAT').upper().strip()  # Default to CHAT if missing

        # 2. VITAL FIX: Explicitly convert the literal string "null", "none", or empty strings to Python None
        null_or_empty_or_none = ["null", "none", ""]
        if city.lower() in null_or_empty_or_none:
            city = None
        if food.lower() in null_or_empty_or_none:
            food = None

        # 3. Final check for common greetings/generic search terms from the original input
        generic_terms = ["hi", "hello", "hey", "food", "restaurant", "meal"]

        # Check if the extracted food is generic (if it wasn't nullified above)
        if food and food.lower() in generic_terms:
            food = None

        # Also check if the entire input was a single greeting and force intent to CHAT
        if user_input.lower().strip() in generic_terms:
            return None, None, 'CHAT'

        return city, food, intent

    except Exception as e:
        st.error(f"Gemini API or Parsing Error: {e}")
        return None, None, 'CHAT'  # Default to CHAT to prevent crashing


# ------------------- Yelp API (Re-implemented) -------------------
def get_restaurants(city, food):
    global YELP_API_KEY

    # Check if the user has provided the required API key for Yelp
    if not YELP_API_KEY:
        return None, "Yelp API key is missing. Please add YELP_API_KEY to secrets.toml to enable search."

    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}

    if not city or not food:
        return None, "Missing required location or search term (API call prevented)."

    params = {"term": food, "location": city, "limit": 5}

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            error_details = response.json().get('error', {}).get('description', 'Unknown API Error.')
            return None, f"Yelp Request Failed ({response.status_code}): {error_details}"

        data = response.json()
        businesses = data.get("businesses", [])

        if not businesses:
            return None, f"No {food.title()} restaurants found in {city.title()}."

        return businesses, None

    except requests.exceptions.RequestException as req_e:
        return None, f"Network Error contacting Yelp: {req_e}"


# ------------------- Streamlit Interface -------------------
st.set_page_config(page_title="🍽️ AI Restaurant Finder", layout="centered")

st.title("🍽️ AI Restaurant Finder")
st.write(
    "Type naturally, e.g. 'Find me **sushi** in **New York**' or 'What are the best places for **vegan food**'.")
st.markdown("---")

# Conversation + memory initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_city" not in st.session_state:
    st.session_state.last_city = None
if "last_food" not in st.session_state:
    st.session_state.last_food = None

# Display conversation
for role, message in st.session_state.messages:
    if role == "user":
        st.markdown(f"**You:** {message}")
    elif role == "ai_result":
        with st.chat_message("assistant"):
            st.markdown(message)

# User input
user_input = st.chat_input("Ask for food and location (e.g., 'Mexican food in Dallas'):")

if user_input:
    # 1. Add user message to state
    st.session_state.messages.append(("user", user_input))

    # 2. Extract structured info and intent using Gemini
    with st.spinner("🤖 Analyzing your request with Gemini..."):
        city_extracted, food_extracted, intent = extract_structured_info(user_input)

    # Clean and normalize extracted values
    city = city_extracted.title() if city_extracted else None
    food = food_extracted.lower() if food_extracted else None

    ai_response = ""

    # --- Intent Routing Logic ---
    if intent == 'CHAT':
        # --- Conversational Flow (for greetings/casual chat) ---
        with st.spinner("🤖 Generating conversational reply..."):
            ai_response_text = generate_chat_response(user_input, st.session_state.messages[:-1])
        ai_response = f"🤖 {ai_response_text}"

    elif intent == 'INFO':
        # --- Informational Flow (suggestions/culture) ---

        # 3. Memory Update for INFO: If a city is mentioned in an INFO query, it updates the geographic context.
        # FIX: Update st.session_state.last_city if a new city is mentioned, even for an INFO query.
        if city:
            st.session_state.last_city = city
            final_city = city
        else:
            # Use the city from memory if no new city was specified
            final_city = st.session_state.last_city

        # VITAL FIX: Final cleanup
        invalid_search_terms = ["null", "none", "n/a", ""]
        if final_city and final_city.lower() in invalid_search_terms:
            final_city = None

        with st.spinner("💡 Consulting the culinary guide..."):
            ai_response_text = generate_info_response(user_input, final_city)
        ai_response = f"🤖 {ai_response_text}"

        # Food memory remains untouched for INFO requests.

    elif intent == 'SEARCH':
        # --- Search Flow (Find a restaurant) ---

        # 3. Apply memory logic and set final search terms

        # Update memory and set final city
        if city:
            st.session_state.last_city = city
        final_city = city if city else st.session_state.last_city

        # Update memory and set final food
        if food:
            st.session_state.last_food = food
        final_food = food if food else st.session_state.last_food

        # VITAL FIX 2: Final cleanup after memory application to prevent "Null" strings from triggering search
        invalid_search_terms = ["null", "none", "n/a", ""]
        if final_city and final_city.lower() in invalid_search_terms:
            final_city = None
        if final_food and final_food.lower() in invalid_search_terms:
            final_food = None

        # Check if we have both pieces of information needed for the Places API call
        ready_to_search = final_food and final_city

        # 4. Handle missing info and search

        if not ready_to_search:
            # We are in search flow but are missing information. Generate prompt.

            if not final_food:
                ai_response = f"🤖 I've noted the location as **{final_city}**. Now, what type of cuisine or dish would you like to search for?"
            elif not final_city:
                # Correctly handles the case where final_food is present but final_city is not
                ai_response = f"🤖 I know you're looking for **{final_food.title()}**, but I need a location. Which city should I search in?"

        else:
            # --- EXECUTE SEARCH ---
            # Perform the Yelp search using the original get_restaurants function
            final_food_title = final_food.title()
            with st.spinner(f"🔎 Searching Yelp for {final_food_title} in {final_city}..."):
                restaurants, error = get_restaurants(final_city, final_food)

            if error:
                # Convert technical error to conversational AI response
                if "API key is missing" in error:
                    ai_response = f"🤖 I need the **YELP_API_KEY** in `secrets.toml` to perform the search. Please add your Yelp API key to continue."
                elif "Missing required location" in error:
                    ai_response = f"🤖 I need a specific city and food type to search. Please provide both!"
                elif "Yelp Request Failed (400)" in error:
                    ai_response = f"🤖 I'm sorry, I couldn't find any results for **{final_food_title}** in **{final_city}**. Yelp might not have data for that specific location or cuisine. Could you try a different city or a more general food type? (e.g. 'Pizza in London, UK')"
                elif "Yelp Request Failed" in error:
                    ai_response = f"🤖 Uh oh! I ran into a service issue while searching. The Yelp service reported: *{error}*. Please try again shortly."
                else:
                    ai_response = f"🤖 Hmm, an unexpected error occurred: *{error}*. Could you rephrase your request?"

            else:
                # Build the success response string
                response_parts = [f"🤖 Here are the top {final_food_title} restaurants in {final_city} (via Yelp):"]
                for r in restaurants:
                    name = r["name"]
                    # Yelp returns address in location->address1
                    address = r["location"]["address1"]
                    rating = r["rating"]
                    maps_query = f"{name}, {final_city}"
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query.replace(' ', '+')}"
                    response_parts.append(f"- [{name}]({maps_url}) | ⭐ **{rating}** | 📍 {address}")

                ai_response = "\n".join(response_parts)

    # 5. Append final AI response
    st.session_state.messages.append(("ai_result", ai_response))

    # 6. FORCE RERUN to display the new message immediately
    st.rerun()
