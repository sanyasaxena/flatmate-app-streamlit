# Flatmate Recommendation System
Flatmate Recommendation System: A Streamlit-based web app that helps users find compatible flatmates based on preferences like age, occupation, tidiness, lifestyle, and locality. The app uses a ML model (KNN with TF-IDF) to match users based on both user preferences and locality.

Features
User Preferences Input:

Users provide preferences related to personality, age, occupation, tidiness, dietary habits, and other lifestyle choices.
The app collects user preferences including locality to make accurate recommendations.
Compatibility Scoring:

The app uses a K-Nearest Neighbors (KNN) model with TF-IDF vectorization to compare user preferences and generate a compatibility score.
The system considers multiple features like age, gender, tidiness preference, occupation, and lifestyle.
Geographical Proximity Filtering:

User locality is matched with corresponding latitude and longitude coordinates.
The app calculates the geographical distance between the user and potential flatmates based on their locality and filters recommendations to those within 6-7 km.
Locality Map Visualization:

The app integrates a map that displays the user’s location and recommended flatmates’ locations using Folium.
A MarkerCluster is used to show multiple flatmate locations on the map, with markers representing both the user's and recommended flatmates' locations.
Recommendation Display:

Recommendations are displayed along with compatibility scores, filtered by geographical proximity.
Each recommendation includes user preferences and a final compatibility score that factors in both user preferences and geographical distance.
How It Works
User Input:

Users input their personal preferences, including age, occupation, tidiness, dietary preferences, lifestyle, and locality.
Data Processing:

The app preprocesses the data, standardizing text fields (like tidiness and alcohol consumption) and lowercasing string values.
The preferences are combined into a single feature set for each user and transformed using TF-IDF vectorization to measure textual similarity.
K-Nearest Neighbors (KNN):

The KNN algorithm is applied to match users with the most compatible flatmates based on the cosine similarity of their features.
Geographical Filtering:

Latitude and longitude for the user's locality are fetched, and the Haversine formula is used to calculate the distance between the user and the recommended flatmates.
Only those within 6-7 km of the user are shown as recommendations.
Map Visualization:

Folium is used to display a map with markers for the user’s location and the recommended flatmates’ locations.
A MarkerCluster groups the flatmate markers, providing an interactive map experience.
Data
User Preferences:

Includes fields such as Age, Gender, Personality, Occupation, Tidiness Preference, Dietary Preferences, Chore Preferences, Lifestyle, Smoking and Alcohol Habits, and Locality.
Locality Coordinates:

Each locality is associated with latitude and longitude coordinates to calculate the distance between users and potential flatmates.
Technology Stack
Streamlit: To build the interactive web app interface.
Pandas: For managing user data and performing data manipulation.
Scikit-learn: For TF-IDF vectorization and K-Nearest Neighbors (KNN) implementation.
Folium: For generating the interactive map with markers to visualize the flatmates' locations.
Numpy: For numerical calculations, especially in distance and similarity computations.
