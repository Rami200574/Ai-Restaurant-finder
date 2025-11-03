import streamlit as st
import requests
import json
import os
from google import genai
from google.genai import types

# ------------------- API Key Setup -------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 Configuration Error: GEMINI_API_KEY not found in Streamlit secrets.")
    st.info("Please add it to your secrets.toml file and ensure you have run: pip install google-genai")
    st.stop()

try:
    YELP_API_KEY = st.secrets["YELP_API_KEY"]
except KeyError:
    st.error("🚨 Configuration Error: YELP_API_KEY not found in Streamlit secrets.")
    st.info("To use the Yelp search, please add your YELP_API_KEY to your secrets.toml file.")
    YELP_API_KEY = None

# Initialize the Gemini Client
try:
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
    for role, text in chat_history:
        # Clean up the AI's displayed text before feeding it back into the model's history
        clean_text = text.replace('🤖', '').strip()

        if role == 'user':
            history_for_gemini.append(types.Content(role='user', parts=[types.Part(text=text)]))
        elif role == 'ai_result':
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
        st.error(f"Gemini Chat Error: {e}")
        return "I seem to be having trouble connecting to the chat service right now. Please try a food and location search."


# ------------------- Gemini Informational Response -------------------

def generate_info_response(query, city=None):
    """
    Generates a response for informational queries (suggestions, culture, etc.).
    Uses the current query and extracted city for context.
    """

    city_context = f" in {city}" if city else ""

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
                description="Classify the user's primary goal: SEARCH (finding restaurants/food listing in a certain address atleast the food or the address should be specified), INFO (asking for popular foods in certain countries not restaurants, it has to be a question), or CHAT (only for greeting/general talk)"
            ),
        })

    system_instruction = (
        "You are an expert entity extraction and intent classification system. "
        "Analyze the user's request. "
        "If the input contains transliterated or colloquial language, first translate the food item into formal English. "
        "Then, extract the City/Location and the desired Food/Cuisine. "
        "Classify the user's primary intent as 'SEARCH', 'INFO', or 'CHAT'. "
        "Return the result strictly in the provided JSON schema. "
        "IMPORTANT: You MUST extract the location/city if one is mentioned. "
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
        intent = data.get('intent', 'CHAT').upper().strip()

        # 2. VITAL FIX: Explicitly convert the literal string "null", "none", or empty strings to Python None
        null_or_empty_or_none = ["null", "none", ""]
        if city.lower() in null_or_empty_or_none:
            city = None
        if food.lower() in null_or_empty_or_none:
            food = None

        # 3. Final check for common greetings/generic search terms from the original input
        generic_terms = ["hi", "hello", "hey", "food", "restaurant", "meal"]
        if food and food.lower() in generic_terms:
            food = None
        if user_input.lower().strip() in generic_terms:
            return None, None, 'CHAT'

        return city, food, intent

    except Exception as e:
        st.error(f"Gemini API or Parsing Error: {e}")
        return None, None, 'CHAT'


# ------------------- Yelp API (Re-implemented) -------------------
def get_restaurants(city, food):
    global YELP_API_KEY

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
# NEW: Track the type of the last successful action (SEARCH or INFO)
if "last_action_type" not in st.session_state:
    st.session_state.last_action_type = None

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
    with st.spinner("🤖"):
        city_extracted, food_extracted, intent = extract_structured_info(user_input)

    # Clean and normalize extracted values
    city = city_extracted.title() if city_extracted else None
    food = food_extracted.lower() if food_extracted else None

    # VITAL: Memory & Final Search Term Determination (regardless of initial intent)

    # 3. Determine Final City: Use extracted city, otherwise use memory
    if city:
        st.session_state.last_city = city
    final_city = city if city else st.session_state.last_city

    # 4. Determine Final Food: Use extracted food, otherwise use memory
    if food:
        st.session_state.last_food = food
    final_food = food if food else st.session_state.last_food

    # Final cleanup after memory application
    invalid_terms = ["null", "none", "n/a", ""]
    if final_city and final_city.lower() in invalid_terms: final_city = None
    if final_food and final_food.lower() in invalid_terms: final_food = None

    # -------------------- ROBUST INTENT OVERRIDE FIX --------------------
    # FIX: Only override to SEARCH if:
    # 1. The query is short/ambiguous (INFO/CHAT intent).
    # 2. A new city is provided.
    # 3. We have a food item in memory.
    # 4. AND the previous successful action was a SEARCH (to confirm we are continuing a search thread).

    user_words = user_input.lower().split()
    is_short_query = 1 <= len(user_words) <= 4

    if (intent == 'INFO' or intent == 'CHAT') and is_short_query and final_city and \
            st.session_state.last_food and not food and \
            st.session_state.last_action_type == 'SEARCH':
        # This catches: "what about newyork" after a search.
        intent = 'SEARCH'
        st.markdown(
            f"*(**System Note:** Intent overridden to **SEARCH** for '{st.session_state.last_food.title()}' in '{final_city}')*",
            help="The short query was interpreted as a location change for the last search.")

    # -------------------- Intent Routing Logic --------------------
    ai_response = ""

    if intent == 'CHAT':
        # --- Conversational Flow (for greetings/casual chat) ---
        with st.spinner("🤖"):
            # Pass the conversation history minus the current user message
            ai_response_text = generate_chat_response(user_input, st.session_state.messages[:-1])
        ai_response = f"🤖 {ai_response_text}"
        st.session_state.last_action_type = 'CHAT'  # Update last action

    elif intent == 'INFO':
        # --- Informational Flow (suggestions/culture) ---
        # This now correctly fires for "what is the best food to eat in london" AND "what about paris" after the London query.
        with st.spinner("🤖"):
            ai_response_text = generate_info_response(user_input, final_city)
        ai_response = f"🤖 {ai_response_text}"
        st.session_state.last_action_type = 'INFO'  # Update last action

    elif intent == 'SEARCH':
        # --- Search Flow (Find a restaurant) ---

        ready_to_search = final_food and final_city

        if not ready_to_search:
            # We are in search flow but are missing information. Generate prompt.

            if not final_food:
                ai_response = f"🤖 I've noted the location as **{final_city}**. Now, what type of cuisine or dish would you like to search for?"
            elif not final_city:
                ai_response = f"🤖 Nice! you're looking for **{final_food.title()}**, but I need a location. Which city should I search in?"

            st.session_state.last_action_type = 'CHAT'  # Treat prompting as a chat response

        else:
            # --- EXECUTE SEARCH ---
            final_food_title = final_food.title()
            with st.spinner(f"🤖 searching for '{final_food}' in '{final_city}'"):
                restaurants, error = get_restaurants(final_city, final_food)

            if error:
                # Convert technical error to conversational AI response
                if "API key is missing" in error:
                    ai_response = f"🤖 I need the **YELP_API_KEY** in `secrets.toml` to perform the search. Please add your Yelp API key to continue."
                elif "Yelp Request Failed (400)" in error:
                    ai_response = f"🤖 I'm sorry, I couldn't find any results for **{final_food_title}** in **{final_city}**. Yelp might not have data for that specific location or cuisine. Could you try a different city or a more general food type? (e.g. 'Pizza in London, UK')"
                elif "Yelp Request Failed" in error:
                    ai_response = f"🤖 Uh oh! I ran into a service issue while searching. The Yelp service reported: *{error}*. Please try again shortly."
                else:
                    ai_response = f"🤖 Hmm, an unexpected error occurred: *{error}*. Could you rephrase your request?"

                st.session_state.last_action_type = 'CHAT'  # Treat error as a chat response

            else:
                # Build the success response string
                response_parts = [f"🤖 Here are the top {final_food_title} restaurants in {final_city}:"]
                for r in restaurants:
                    name = r["name"]
                    address = r["location"]["address1"]
                    rating = r["rating"]
                    maps_query = f"{name}, {final_city}"
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query.replace(' ', '+')}"
                    response_parts.append(f"- [{name}]({maps_url}) | ⭐ **{rating}** | 📍 {address}")

                ai_response = "\n".join(response_parts)
                st.session_state.last_action_type = 'SEARCH'  # Update last action

    # 5. Append final AI response
    st.session_state.messages.append(("ai_result", ai_response))

    # 6. FORCE RERUN to display the new message immediately
    st.rerun()
