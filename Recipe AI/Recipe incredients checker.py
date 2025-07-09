
import streamlit as st
import google.generativeai as genai
import os

# Load Gemini API key from an environment variable (make sure to replace with your key)
genai.configure(api_key="AIzaSyCLpifWg0cbS62bJ0rAyscNkt5-PbDYOcs")  # You can also use os.getenv('GEMINI_API_KEY') for production

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# Streamlit app title
st.title("🥘 Recipe Ingredient Checker")
st.write("Enter the ingredients you have, and get recipe ideas!")

# Input fields
ingredients = st.text_area("Enter ingredients (comma-separated):")
diet = st.selectbox("Select dietary preference", ["None", "Vegetarian", "Vegan", "Gluten-Free", "Keto"])
cuisine = st.selectbox("Select preferred cuisine", ["Any", "Indian", "Italian", "Mexican", "Chinese", "American"])
meal_type = st.selectbox("Meal type", ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert"])

# Button to generate recipe suggestions
if st.button("Get Recipes"):
    if not ingredients:
        st.warning("Please enter some ingredients!")
    else:
        with st.spinner("Generating recipes..."):
            # Constructing the prompt for the Gemini API
            prompt = f"""
            I have the following ingredients: {ingredients}.
            Please suggest a few {meal_type.lower()} recipes that are {diet.lower()} (if specified) and inspired by {cuisine} cuisine (if applicable).
            Include:
            - Recipe name
            - List of ingredients
            - Preparation steps
            - Estimated preparation time
            """

            try:
                # Sending the prompt to Gemini API and getting the response
                response = model.generate_content(prompt)
                recipe_text = response.text

                # Displaying the recipe suggestions
                st.markdown("### Suggested Recipes:")
                st.markdown(recipe_text)
            except Exception as e:
                # Handling errors gracefully
                st.error(f"Failed to generate recipe: {e}")
