import streamlit as st
import google.generativeai as genai
import os

# Load Gemini API key
genai.configure(api_key="AIzaSyCLpifWg0cbS62bJ0rAyscNkt5-PbDYOcs")

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# Streamlit app title
st.title("🥘 Recipe Ingredient Checker")
st.write("Enter the ingredients you have, and get recipe ideas!")

# Input fields
ingredients = st.text_area("Enter ingredients (comma-separated):")
diet = st.selectbox("Select dietary preference", ["None", "Vegetarian", "Vegan", "Gluten-Free", "Keto"])
cuisine = st.selectbox("Select preferred cuisine", ["Any", "Indian", "Italian", "Mexican", "Chinese", "American"])
meal_type = st.selectbox("Meal type", ["Breakfast", "Lunch", "Dinner"])

# Button to generate recipe suggestions
if st.button("Get Recipes"):
    if not ingredients.strip():
        st.warning("Please enter some ingredients!")
    else:
        with st.spinner("Generating recipes..."):

            # Add basic ingredients based on the type of recipe
            basic_ingredients = ""
            if meal_type in ["Breakfast", "Lunch", "Dinner"]:
                if any(sweet in ingredients.lower() for sweet in ["milk", "banana", "honey", "jaggery", "fruit", "chocolate"]):
                    basic_ingredients = ", sugar, ghee"
                else:
                    basic_ingredients = ", salt, pepper, oil"

            all_ingredients = ingredients + basic_ingredients

            # Constructing the prompt
            prompt = f"""
            I have the following ingredients only: {all_ingredients}.
            Please suggest a few {meal_type.lower()} recipes using strictly these ingredients.
            Ensure the recipes are suitable for a {diet.lower()} diet if specified and follow {cuisine} cuisine style if chosen.

            For each recipe, include:
            - Recipe name
            - List of ingredients used (only from the list given)
            - Preparation steps
            - Estimated preparation time
            Do not include any ingredients that are not mentioned above.
            """

            try:
                # Generate response
                response = model.generate_content(prompt)
                recipe_text = response.text

                # Display recipes
                st.markdown("### Suggested Recipes:")
                st.markdown(recipe_text)
            except Exception as e:
                st.error(f"Failed to generate recipe: {e}")
