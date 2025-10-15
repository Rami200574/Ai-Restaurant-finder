import streamlit as st
import requests
import spacy
#from rapidfuzz import process

# ------------------- Setup NLP -------------------
nlp = spacy.load("en_core_web_sm")

# ------------------- NLP Extraction -------------------
def extract_city_food(user_input):
    doc = nlp(user_input)
    city = None
    food = None

    # Extract city (GPE)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            city = ent.text
            break

    # Extract potential food (nouns, proper nouns, unknown words)
    candidates = [
        token.text.lower() for token in doc
        if token.pos_ in ["NOUN", "PROPN", "X"] and (not city or token.text.lower() != city.lower())
    ]

    if candidates:
        # Just take the first noun/proper noun as food
        food = candidates[0]

    return city, food


def correct_city(city):
    if city is None:
        return None
    return city.strip().title()


def correct_food(food):
    if food is None:
        return None
    return food.strip().lower()

# ------------------- Yelp API -------------------
def get_restaurants(city, food):
    api_key = "FdtNaIRMu-USU1SkdqL500SASXqhYY4nWezxSDgjfTzD4E9ldokL0BcTXqvdAWHcNioh0YjB1kIpr804GlcHe3njgPdY5LT9c_m6SWdOdrTHcyCSSrPlizy_Bx7qaHYx"
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"term": food, "location": city, "limit": 5}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return None, f"Error connecting to Yelp API: {response.status_code}"
    data = response.json()
    businesses = data.get("businesses", [])
    if not businesses:
        return None, f"No {food} restaurants found in {city}."
    return businesses, None

# ------------------- Streamlit Interface -------------------
st.title("🍽️ AI Restaurant Finder")
st.write("Type naturally, e.g. 'Find me sushi in Tokyo' or 'Show me pizza nearby'.")

# Conversation + memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_city" not in st.session_state:
    st.session_state.last_city = None
if "last_food" not in st.session_state:
    st.session_state.last_food = None

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # Extract entities
    city_raw, food_raw = extract_city_food(user_input)
    corrected_city = correct_city(city_raw)
    corrected_food = correct_food(food_raw)

    # --- MEMORY SYSTEM (Safe) ---
    # Only update memory if user actually typed a new city or food
    if city_raw:
        st.session_state.last_city = corrected_city
    if food_raw:
        st.session_state.last_food = corrected_food

    # Only use memory if user typed at least one piece of info
    if city_raw or food_raw:
        if not corrected_city:
            corrected_city = st.session_state.last_city
        if not corrected_food:
            corrected_food = st.session_state.last_food

    # --- Handle missing info ---
    if not corrected_food and not corrected_city:
        st.warning("🤖 Please type at least a city or a food you want.")
    elif not corrected_food:
        st.warning("🤖 What type of food are you looking for?")
    elif not corrected_city:
        st.warning("🤖 Which city?")
    else:
        restaurants, error = get_restaurants(corrected_city, corrected_food)
        if error:
            st.error(f"🤖 {error}")
        else:
            st.success(f"🤖 Here are {corrected_food} restaurants in {corrected_city}:")
            for r in restaurants:
                name = r["name"]
                address = r["location"]["address1"]
                rating = r["rating"]
                maps_url = f"https://www.google.com/maps/search/?api=1&query={name.replace(' ', '+')},+{corrected_city}"
                st.markdown(f"- [{name}]({maps_url}) | ⭐ {rating} | 📍 {address}")


# Display conversation
for role, message in st.session_state.messages:
    if role == "user":
        st.markdown(f"**You:** {message}")
