"""
Cognifyz Technologies - Machine Learning Internship
Task 2: Restaurant Recommendation System
-----------------------------------------------------
Objective:
    Build a content-based restaurant recommendation system that suggests
    restaurants to a user based on their preferences (cuisine, price
    range, and city).

Steps followed (as per task brief):
    1. Preprocess the dataset - handle missing values, encode categorical
       variables.
    2. Determine criteria for recommendations (cuisine preference, price
       range, city).
    3. Implement content-based filtering - restaurants similar to the
       user's preferred criteria are recommended using cosine similarity.
    4. Test the system with sample user preferences and evaluate the
       quality of the recommendations.

Author: Jeeva Karnan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


# ------------------------------------------------------------------
# STEP 1: LOAD & PREPROCESS THE DATASET
# ------------------------------------------------------------------
def load_and_preprocess(path="data/Dataset.csv"):
    df = pd.read_csv(path)

    # --- Handle missing values ---
    # A handful of rows have no listed cuisine; fill with 'Unknown'
    # instead of dropping rows, so we don't lose restaurant records.
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # --- Clean up the Cuisines column into a list per restaurant ---
    df["Cuisines_List"] = (
        df["Cuisines"].str.split(",").apply(lambda lst: [c.strip() for c in lst])
    )

    # --- Encode categorical variables ---
    # Cuisines: multi-hot encoding (a restaurant can serve many cuisines)
    mlb = MultiLabelBinarizer()
    cuisine_encoded = mlb.fit_transform(df["Cuisines_List"])
    cuisine_df = pd.DataFrame(
        cuisine_encoded, columns=mlb.classes_, index=df.index
    )

    # City: one-hot encoding (used as a feature + hard filter option)
    city_encoded = pd.get_dummies(df["City"], prefix="City")

    # --- Numeric features scaled to 0-1 so no single feature dominates ---
    scaler = MinMaxScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[["Average Cost for two", "Price range", "Votes"]]),
        columns=["Cost_scaled", "PriceRange_scaled", "Votes_scaled"],
        index=df.index,
    )

    # --- Final feature matrix used for similarity computation ---
    # Weight cuisines highest (core of "content"), then price range,
    # then cost/votes/city as secondary signals.
    feature_matrix = pd.concat(
        [cuisine_df * 2.0, numeric_scaled, city_encoded * 0.5], axis=1
    )

    return df, feature_matrix, mlb


# ------------------------------------------------------------------
# STEP 2 & 3: BUILD USER PROFILE VECTOR + CONTENT-BASED FILTERING
# ------------------------------------------------------------------
def build_user_vector(user_prefs, feature_matrix, mlb, df, scaler_ref):
    """Builds a feature vector for the user matching feature_matrix columns."""
    vector = pd.Series(0.0, index=feature_matrix.columns)

    # Cuisine preference(s)
    for cuisine in user_prefs.get("cuisines", []):
        if cuisine in mlb.classes_:
            vector[cuisine] = 2.0  # match the weighting used above

    # Price range preference (1 = cheap ... 4 = expensive), scaled 0-1
    if "price_range" in user_prefs:
        vector["PriceRange_scaled"] = (user_prefs["price_range"] - 1) / 3.0

    # Approx cost preference, scaled using same range as dataset
    if "avg_cost_for_two" in user_prefs:
        min_c, max_c = df["Average Cost for two"].min(), df["Average Cost for two"].max()
        vector["Cost_scaled"] = (user_prefs["avg_cost_for_two"] - min_c) / (max_c - min_c)

    # Votes: assume user wants a well-reviewed place -> aim high
    vector["Votes_scaled"] = 0.7

    # City preference (hard-ish signal via weighting)
    city_col = f"City_{user_prefs.get('city', '')}"
    if city_col in vector.index:
        vector[city_col] = 0.5

    return vector.values.reshape(1, -1)


def recommend_restaurants(user_prefs, df, feature_matrix, mlb, top_n=10):
    user_vector = build_user_vector(user_prefs, feature_matrix, mlb, df, None)

    similarities = cosine_similarity(user_vector, feature_matrix.values)[0]

    result = df.copy()
    result["Similarity Score"] = similarities

    # Optional hard filter: only same city if user specified one and
    # matches exist, otherwise fall back to similarity ranking city-wide.
    if user_prefs.get("city"):
        city_matches = result[result["City"].str.lower() == user_prefs["city"].lower()]
        if len(city_matches) >= top_n:
            result = city_matches

    result = result.sort_values("Similarity Score", ascending=False)

    cols = [
        "Restaurant Name",
        "City",
        "Cuisines",
        "Average Cost for two",
        "Price range",
        "Aggregate rating",
        "Votes",
        "Similarity Score",
    ]
    return result[cols].head(top_n).reset_index(drop=True)


# ------------------------------------------------------------------
# STEP 4: TEST THE SYSTEM WITH SAMPLE USER PREFERENCES
# ------------------------------------------------------------------
def evaluate_recommendations(recommendations, user_prefs):
    """Simple quality check: % of results matching the requested cuisine,
    and whether results stay within a reasonable price-range band."""
    if len(recommendations) == 0:
        return {"cuisine_match_rate": 0.0, "avg_rating": 0.0}

    requested = set(c.lower() for c in user_prefs.get("cuisines", []))
    match_count = recommendations["Cuisines"].apply(
        lambda c: any(r in c.lower() for r in requested)
    ).sum()

    return {
        "cuisine_match_rate": round(match_count / len(recommendations) * 100, 1),
        "avg_rating": round(recommendations["Aggregate rating"].mean(), 2),
        "avg_price_range": round(recommendations["Price range"].mean(), 2),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2: RESTAURANT RECOMMENDATION SYSTEM (Content-Based Filtering)")
    print("=" * 70)

    df, feature_matrix, mlb = load_and_preprocess("data/Dataset.csv")
    print(f"\nDataset loaded: {df.shape[0]} restaurants, {df.shape[1]} original columns")
    print(f"Feature matrix built: {feature_matrix.shape[1]} features "
          f"({len(mlb.classes_)} cuisine categories + price/cost/votes + city)")

    # --- Sample user 1: wants Italian food, mid price range, in a specific city ---
    user1 = {
        "cuisines": ["Italian"],
        "price_range": 3,
        "avg_cost_for_two": 1200,
        "city": "New Delhi",
    }
    print("\n" + "-" * 70)
    print(f"SAMPLE USER 1 PREFERENCES: {user1}")
    print("-" * 70)
    recs1 = recommend_restaurants(user1, df, feature_matrix, mlb, top_n=10)
    print(recs1.to_string(index=False))
    print("\nEvaluation:", evaluate_recommendations(recs1, user1))

    # --- Sample user 2: wants North Indian + Chinese, budget-friendly ---
    user2 = {
        "cuisines": ["North Indian", "Chinese"],
        "price_range": 1,
        "avg_cost_for_two": 400,
        "city": "New Delhi",
    }
    print("\n" + "-" * 70)
    print(f"SAMPLE USER 2 PREFERENCES: {user2}")
    print("-" * 70)
    recs2 = recommend_restaurants(user2, df, feature_matrix, mlb, top_n=10)
    print(recs2.to_string(index=False))
    print("\nEvaluation:", evaluate_recommendations(recs2, user2))

    # --- Sample user 3: no city preference, wants Japanese cuisine, fine dining ---
    user3 = {
        "cuisines": ["Japanese"],
        "price_range": 4,
        "avg_cost_for_two": 3000,
        "city": "",
    }
    print("\n" + "-" * 70)
    print(f"SAMPLE USER 3 PREFERENCES: {user3}")
    print("-" * 70)
    recs3 = recommend_restaurants(user3, df, feature_matrix, mlb, top_n=10)
    print(recs3.to_string(index=False))
    print("\nEvaluation:", evaluate_recommendations(recs3, user3))

    print("\n" + "=" * 70)
    print("Recommendation system test complete.")
    print("=" * 70)
