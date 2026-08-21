"""
Cognifyz Technologies - Machine Learning Internship
Task 2: Restaurant Recommendation System

Content-based restaurant recommendation using cuisine, price, cost,
votes, and city preferences with cosine similarity.

Author: Jeeva Karnan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def load_and_preprocess(path="data/Dataset.csv"):
    df = pd.read_csv(path)
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    df["Cuisines_List"] = df["Cuisines"].str.split(",").apply(
        lambda lst: [c.strip() for c in lst]
    )

    mlb = MultiLabelBinarizer()
    cuisine_encoded = mlb.fit_transform(df["Cuisines_List"])
    cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_, index=df.index)
    city_encoded = pd.get_dummies(df["City"], prefix="City")

    scaler = MinMaxScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[["Average Cost for two", "Price range", "Votes"]]),
        columns=["Cost_scaled", "PriceRange_scaled", "Votes_scaled"],
        index=df.index,
    )

    feature_matrix = pd.concat(
        [cuisine_df * 2.0, numeric_scaled, city_encoded * 0.5], axis=1
    )
    return df, feature_matrix, mlb


def build_user_vector(user_prefs, feature_matrix, mlb, df):
    vector = pd.Series(0.0, index=feature_matrix.columns)

    for cuisine in user_prefs.get("cuisines", []):
        if cuisine in mlb.classes_:
            vector[cuisine] = 2.0

    if "price_range" in user_prefs:
        vector["PriceRange_scaled"] = (user_prefs["price_range"] - 1) / 3.0

    if "avg_cost_for_two" in user_prefs:
        min_c = df["Average Cost for two"].min()
        max_c = df["Average Cost for two"].max()
        vector["Cost_scaled"] = (user_prefs["avg_cost_for_two"] - min_c) / (max_c - min_c)

    vector["Votes_scaled"] = 0.7

    city_col = f"City_{user_prefs.get('city', '')}"
    if city_col in vector.index:
        vector[city_col] = 0.5

    return vector.values.reshape(1, -1)


def recommend_restaurants(user_prefs, df, feature_matrix, mlb, top_n=10):
    user_vector = build_user_vector(user_prefs, feature_matrix, mlb, df)
    similarities = cosine_similarity(user_vector, feature_matrix.values)[0]

    result = df.copy()
    result["Similarity Score"] = similarities

    if user_prefs.get("city"):
        city_matches = result[result["City"].str.lower() == user_prefs["city"].lower()]
        if len(city_matches) >= top_n:
            result = city_matches

    result = result.sort_values("Similarity Score", ascending=False)
    cols = [
        "Restaurant Name", "City", "Cuisines", "Average Cost for two",
        "Price range", "Aggregate rating", "Votes", "Similarity Score"
    ]
    return result[cols].head(top_n).reset_index(drop=True)


def evaluate_recommendations(recommendations, user_prefs):
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
    print("TASK 2: RESTAURANT RECOMMENDATION SYSTEM")
    print("=" * 70)

    df, feature_matrix, mlb = load_and_preprocess("data/Dataset.csv")
    print(f"\nDataset loaded: {df.shape[0]} restaurants")
    print(f"Feature matrix: {feature_matrix.shape[1]} features")

    sample_users = [
        {"cuisines": ["Italian"], "price_range": 3, "avg_cost_for_two": 1200, "city": "New Delhi"},
        {"cuisines": ["North Indian", "Chinese"], "price_range": 1, "avg_cost_for_two": 400, "city": "New Delhi"},
        {"cuisines": ["Japanese"], "price_range": 4, "avg_cost_for_two": 3000, "city": ""},
    ]

    for i, user in enumerate(sample_users, 1):
        print(f"\nSAMPLE USER {i}: {user}")
        recs = recommend_restaurants(user, df, feature_matrix, mlb, top_n=10)
        print(recs.to_string(index=False))
        print("Evaluation:", evaluate_recommendations(recs, user))

    print("\nRecommendation system test complete.")
