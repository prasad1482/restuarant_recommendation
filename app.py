import streamlit as st
import pandas as pd
import pickle

# Load data and model
df = pd.read_csv('data/cleaned_restaurants.csv')
with open('model/cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Set page config
st.set_page_config(page_title="Restaurant Recommender", layout="wide")

# App title
st.title("🍽️ Restaurant Recommendation System")
st.markdown("Get personalized restaurant suggestions based on your preferences!")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
selected_city = st.sidebar.selectbox("City", options=["Any"] + sorted(df['City'].unique().tolist()))
selected_price = st.sidebar.selectbox("Price Range (1 = Low, 4 = High)", options=["Any", 1, 2, 3, 4])
min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=5.0, step=0.1, value=3.5)

# Search input
search_input = st.text_input("Enter a Restaurant or Dish/Cuisine Name").strip()

# Recommendation logic
def get_recommendations(name_or_dish, cosine_sim=cosine_sim):
    # Try finding exact restaurant match
    matches = df[df['Restaurant Name'].str.lower() == name_or_dish.lower()]
    if not matches.empty:
        idx = matches.index[0]
    else:
        # Fallback: try partial match in cuisines
        cuisine_matches = df[df['Cuisines'].str.lower().str.contains(name_or_dish.lower(), na=False)]
        if cuisine_matches.empty:
            return pd.DataFrame()  # No match found
        idx = cuisine_matches.index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]  # Top 30 similar
    restaurant_indices = [i[0] for i in sim_scores]
    results = df.iloc[restaurant_indices]

    # Apply filters
    if selected_city != "Any":
        results = results[results['City'] == selected_city]
    if selected_price != "Any":
        results = results[results['Price range'] == int(selected_price)]
    results = results[results['Aggregate rating'] >= min_rating]

    # Return top 10
    return results[['Restaurant Name', 'City', 'Cuisines', 'Aggregate rating', 'Price range']].head(10)

# Button action
if st.button("🔍 Recommend"):
    if search_input:
        recommendations = get_recommendations(search_input)
        if recommendations.empty:
            st.error("No matching restaurants found. Try different input or adjust filters.")
        else:
            st.success("Here are your recommended restaurants:")
            st.dataframe(recommendations)
    else:
        st.warning("Please enter a restaurant name or dish/cuisine.")
