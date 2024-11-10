import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import folium
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Constants
EARTH_RADIUS = 6371.0  # Radius of the Earth in kilometers
FLATMATES_FILE = "user_data.csv"
COORDINATES_FILE = "localities_coordinates.csv"
MAX_DISTANCE_KM = 7  # Upper limit for geographical distance in kilometers

# Load coordinates from CSV
def load_coordinates(file_path):
    df = pd.read_csv(file_path)
    return {row['Locality'].lower(): (row['Latitude'], row['Longitude']) for _, row in df.iterrows()}

locality_coordinates = load_coordinates(COORDINATES_FILE)

def get_coordinates(locality):
    """Retrieve latitude and longitude for a given locality."""
    return locality_coordinates.get(locality.lower(), (None, None))

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * atan2(sqrt(a), sqrt(1 - a)) * EARTH_RADIUS

# Load and preprocess dataset
df = pd.read_csv(FLATMATES_FILE)

features = [
    'Age', 'Gender', 'Personality', 'Occupation', 'Tidiness Preference',
    'Dietary Preferences', 'Looking for (Gender)', 'Chore Preferences',
    'Personality Type', 'Lifestyle', 'Do you smoke?', 'Do you consume alcohol?',
    'Locality'
]

standardization_mappings = {
    'Tidiness Preference': {
        'Flexible with tidiness levels': 'Flexible',
        "I'm flexible with tidiness levels": 'Flexible',
        'Comfortable with some clutter': 'Some Clutter',
        "I'm comfortable with some clutter": 'Some Clutter',
        'Prefer a very tidy space': 'Very Tidy',
        'I prefer a very tidy space': 'Very Tidy'
    },
    'Do you consume alcohol?': {'yes': 'Yes', 'no': 'No', 'occasionally': 'Occasionally'},
    'Do you smoke?': {'yes': 'Yes', 'no': 'No'},
    'Locality': {'Shivaji Nagar': 'Shivajinagar'}
}

def preprocess_data(df):
    """Standardize data and lower-case string features."""
    for column, mapping in standardization_mappings.items():
        df[column] = df[column].replace(mapping)
    df[features] = df[features].astype(str).apply(lambda x: x.str.lower())
    return df

df = preprocess_data(df)

def combine_features(row):
    """Combine features for TF-IDF vectorization."""
    return ' '.join([f"{feature}:{str(row[feature])}" for feature in features])

df['combined_features'] = df.apply(combine_features, axis=1)

# Fit TF-IDF and KNN models
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
knn_model = NearestNeighbors(metric='cosine', algorithm='brute').fit(tfidf_matrix)

# Streamlit UI setup
st.set_page_config(page_title="Flatmate Finder", layout="centered")
st.title("Flatmate Recommendation System")
st.write("Fill out your preferences to find compatible flatmates.")

# User preference input
with st.expander("Select Your Preferences"):
    user_preferences = {feature: st.selectbox(f"Select {feature}", sorted(df[feature].unique())) for feature in features}

if st.button("Recommend Flatmates"):
    def recommend_flatmates(user_features, gender_pref=None):
        """Return recommended flatmates with compatibility and distance filtering."""
        user_tfidf = tfidf_vectorizer.transform([user_features])
        distances, indices = knn_model.kneighbors(user_tfidf, n_neighbors=5 + 1)
        recommended_indices = [idx for idx in indices.flatten()[1:] if not gender_pref or df.iloc[idx]['Gender'] == gender_pref]
        return recommended_indices, distances.flatten()[1:]

    user_features = ' '.join([f"{feature}:{user_preferences[feature]}" for feature in features])
    user_locality = user_preferences['Locality']
    user_lat, user_lon = get_coordinates(user_locality)

    recommended_indices, distances = recommend_flatmates(user_features, user_preferences.get('Looking for (Gender)'))
    compatibility_scores = [(1 - dist) * 100 for dist in distances]

    # Calculate geographical distances and filter
    geo_distances = [haversine(user_lat, user_lon, *get_coordinates(df.iloc[idx]['Locality'])) if user_lat and user_lon else np.inf for idx in recommended_indices]
    filtered_indices = [idx for idx, geo_dist in zip(recommended_indices, geo_distances) if geo_dist <= MAX_DISTANCE_KM]
    
    final_scores = [
        0.7 * compatibility_scores[i] + 0.3 * (max(0, (100 - geo_dist) / 100) * 100)
        for i, (idx, geo_dist) in enumerate(zip(filtered_indices, geo_distances))
    ]

    # Display recommendations
    st.write("### Recommended Flatmates (within 6-7 km):")
    for i, idx in enumerate(filtered_indices):
        flatmate_info = df.iloc[idx].to_dict()
        st.markdown(f"""
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <h3>Flatmate {i+1}</h3>
                <p><strong>Compatibility:</strong> {round(compatibility_scores[i], 2)}%</p>
                <p><strong>Final Compatibility (with distance):</strong> {round(final_scores[i], 2)}%</p>
                <ul>
                    {"".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in flatmate_info.items() if key != 'combined_features'])}
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Map with MarkerCluster
    st.write("### Flatmates on the Map:")
    flatmate_map = folium.Map(location=[user_lat, user_lon], zoom_start=16)
    marker_cluster = MarkerCluster().add_to(flatmate_map)
    folium.Marker([user_lat, user_lon], popup="Your Location", icon=folium.Icon(color='blue')).add_to(marker_cluster)

    for i, idx in enumerate(filtered_indices):
        lat, lon = get_coordinates(df.iloc[idx]['Locality'])
        if lat and lon:
            folium.Marker([lat, lon], popup=f"Flatmate {i+1}", icon=folium.Icon(color='green')).add_to(marker_cluster)

    html(flatmate_map._repr_html_(), height=600)
