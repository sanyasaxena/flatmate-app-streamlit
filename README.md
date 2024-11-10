# Flatmate Recommendation System

A web-based application built with **Streamlit** that helps users find compatible flatmates based on their personal preferences and geographical proximity. The app uses machine learning (K-Nearest Neighbors with TF-IDF vectorization) to match users based on personality traits and preferences, and calculates geographical distances to recommend flatmates within a reasonable location.

## Features

- **User Preferences Matching**: Users provide preferences like age, gender, occupation, tidiness, dietary habits, lifestyle, and locality.
- **Compatibility Scoring**: The system calculates compatibility scores by comparing user preferences using **TF-IDF vectorization** and the **K-Nearest Neighbors (KNN)** algorithm.
- **Geographical Proximity**: The app calculates geographical distances between users using **Haversine formula**, ensuring recommendations are limited to flatmates within **6-7 km**.
- **Locality Map**: The app displays user and flatmate locations on an interactive **map** using **Folium**, helping users visualize proximity.
- **Interactive Streamlit UI**: The app offers an easy-to-use interface for users to select their preferences and view the results.

## How It Works

1. **User Input**: 
   - Users select preferences such as age, gender, occupation, tidiness preference, dietary habits, and locality.
2. **Data Processing**:
   - User input is preprocessed and standardized (e.g., cleaning text, mapping responses like "yes" to "Yes").
   - The preferences are combined into a single string for each user, which is then vectorized using **TF-IDF** to quantify textual data.
3. **Compatibility Matching**:
   - A **K-Nearest Neighbors (KNN)** algorithm is applied to the transformed data to identify the most compatible flatmates.
4. **Geographical Filtering**:
   - The geographical distance between the user and potential flatmates is calculated using **latitude** and **longitude** coordinates.
   - Recommendations are filtered to only show flatmates within **6-7 km** of the user.
5. **Flatmate Recommendations**:
   - Flatmates are ranked based on compatibility scores and final scores that incorporate both compatibility and geographical proximity.
6. **Locality Map**:
   - The app plots the user’s and recommended flatmates’ locations on a **map** using **Folium**.
   - The user’s location is marked in **blue**, while recommended flatmates' locations are marked in **green**.

## Data

- **User Preferences**: Fields like age, gender, personality type, occupation, tidiness preference, dietary habits, smoking/alcohol consumption, and locality.
- **Locality Coordinates**: The localities are associated with **latitude** and **longitude** values to compute geographical distances.
- **Standardized Features**: Some features, such as "Do you consume alcohol?" and "Tidiness Preference," are standardized to maintain consistency in user input.

## Technology Stack

- **Streamlit**: For building the web application interface and user interaction.
- **Pandas**: For data manipulation, handling user preferences, and processing CSV files.
- **Scikit-learn**: For machine learning tasks, including **TF-IDF vectorization** and **K-Nearest Neighbors (KNN)** algorithm.
- **Folium**: For generating interactive maps and displaying geographical data.
- **Numpy**: For handling numerical operations, especially in distance calculations using the Haversine formula.

## Workflow

1. **Preprocessing**:
   - User input is standardized and converted into lowercase.
   - Features are combined into a single string for each user to be used in TF-IDF vectorization.
   
2. **Model Training**:
   - The system uses **TF-IDF** to vectorize user preferences and **KNN** to find the nearest neighbors (most compatible flatmates).

3. **Geographical Filtering**:
   - The **Haversine formula** is used to calculate distances between localities and filter out flatmates who are not within a reasonable proximity (6-7 km).

4. **Recommendation Generation**:
   - Recommendations are generated based on compatibility scores (calculated by KNN) and geographical proximity (based on Haversine distance).

5. **Map Visualization**:
   - **Folium** is used to display the user's location and the recommended flatmates on a map, making it easy to visualize proximity.

## Conclusion

The **Flatmate Recommendation System** uses machine learning and geospatial data to help users find flatmates who match their preferences in terms of personality and lifestyle, as well as being geographically close. The system ensures both compatibility and convenience by factoring in user preferences and location, providing personalized recommendations with a visual map interface.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/flatmate-recommendation-system.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run flatmate_app.py
    ```

---
