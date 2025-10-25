🍽️ AI Restaurant Finder (Gemini + Yelp)

This is an interactive, conversational restaurant finder application built using Python and Streamlit. It leverages the Google Gemini API for natural language understanding (NLU), intent classification, and chat responses, and the Yelp API for real-time restaurant data retrieval.

The application intelligently maintains conversational context, allowing users to search for food in one city and then switch locations or ask follow-up questions without repeating the cuisine type.

✨ Features

Conversational Interface: Uses Gemini's chat model (gemini-2.5-flash) for friendly, context-aware responses and prompts.

Structured Entity Extraction: Gemini is used to reliably extract key entities (city and food) and classify the user's intent (SEARCH, INFO, or CHAT) using structured JSON output.

Contextual Memory: Stores the last successfully searched city and food type to enable rapid follow-up searches (e.g., "What about New York?" will search for the previously mentioned food in New York).

Yelp Integration: Performs restaurant searches using the Yelp Fusion API (businesses/search endpoint).

Culinary Information: Handles INFO queries by providing cultural or popular food suggestions for a specific city.

⚙️ Setup and Installation

1. Clone the repository

git clone <your-repo-url>
cd ai-restaurant-finder


2. Install Dependencies

The application relies on streamlit, google-genai (for the Gemini API), and requests (for the Yelp API).

pip install streamlit google-genai requests


3. API Key Configuration

You will need API keys from both Google AI Studio (for Gemini) and Yelp (for restaurant data).

Create a file named .streamlit/secrets.toml in your project root directory (if it doesn't exist already) and add your keys:

# .streamlit/secrets.toml

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
YELP_API_KEY = "YOUR_YELP_API_KEY_HERE"


Google Gemini API: Obtain your key from [Google AI Studio].

Yelp API: Obtain your key (specifically the 'API Key') from the [Yelp Developer Portal].

🚀 How to Run the App

Once the dependencies are installed and the secrets.toml file is configured, you can run the Streamlit application from your terminal:

streamlit run streamlit_app.py


The app will open in your default web browser (usually at http://localhost:8501).

👨‍💻 Usage Examples

Direct Search: "I want sushi in London."

Conversational Switch:

User: "Find me Italian food in Rome."

AI: (Returns results)

User: "What about Milan?" (The app searches for "Italian food in Milan" using the memory context.)

Informational Query: "What are some popular dishes in New Orleans?" (This triggers the INFO intent and generates a cultural response using Gemini.)
