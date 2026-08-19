# Restaurant Recommendation System

A content-based restaurant recommendation engine built as part of the **Cognifyz Technologies Machine Learning Internship** (Task 2). Given a user's preferences — cuisine, price range, budget, and city — the system recommends the most similar restaurants from a dataset of ~9,500 restaurants using cosine similarity.

## Objective

Build a recommendation system that suggests restaurants to a user based on their preferences, using a content-based filtering approach (as opposed to collaborative filtering, which needs historical user-item interaction data this dataset doesn't have).

## Dataset

- **Source:** `data/Dataset.csv`
- **Size:** 9,551 restaurants × 21 original columns
- **Key fields used:** Cuisines, City, Average Cost for two, Price range, Votes, Aggregate rating

## Approach

1. **Preprocessing**
   - Missing `Cuisines` values filled with `"Unknown"` instead of dropping rows.
   - `Cuisines` split into individual cuisine tags per restaurant.

2. **Feature Encoding**
   - **Cuisines** → multi-hot encoded (a restaurant can serve multiple cuisines) via `MultiLabelBinarizer`.
   - **City** → one-hot encoded via `pd.get_dummies`.
   - **Average Cost for two, Price range, Votes** → scaled to [0, 1] with `MinMaxScaler` so no single numeric feature dominates the distance calculation.
   - Cuisine features are weighted 2× since cuisine match is the primary driver of a good recommendation; city features are weighted 0.5× as a secondary signal.

3. **User Profile Vector**
   - The user's stated preferences (cuisines, price range, budget, city) are encoded into a vector in the *same feature space* as the restaurants.

4. **Content-Based Filtering**
   - **Cosine similarity** between the user vector and every restaurant vector ranks restaurants by how closely they match the user's stated tastes.
   - If the user specifies a city with enough matching restaurants, results are filtered to that city first, then ranked by similarity.

5. **Evaluation**
   - Cuisine match rate (% of top-N results containing a requested cuisine)
   - Average rating and average price range of the recommended set

## Project Structure

```
restaurant-recommendation-system/
├── data/
│   └── Dataset.csv              # Restaurant dataset
├── recommendation_system.py     # Main script: preprocessing + recommender
├── sample_output.txt            # Captured output from a sample run
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the recommender (executes 3 built-in sample users)
python recommendation_system.py
```

To get recommendations for your own preferences, import the functions directly:

```python
from recommendation_system import load_and_preprocess, recommend_restaurants

df, feature_matrix, mlb = load_and_preprocess("data/Dataset.csv")

user_prefs = {
    "cuisines": ["Italian"],
    "price_range": 3,        # 1 (cheap) - 4 (expensive)
    "avg_cost_for_two": 1200,
    "city": "New Delhi",
}

recommendations = recommend_restaurants(user_prefs, df, feature_matrix, mlb, top_n=10)
print(recommendations)
```

## Sample Results

| User Preference | Cuisine Match Rate | Avg. Rating |
|---|---|---|
| Italian, mid-price, New Delhi | 100% | 3.79 |
| North Indian + Chinese, budget, New Delhi | 100% | 2.97 |
| Japanese, fine dining, any city | 100% | 3.78 |

Full output for all three test cases is saved in `sample_output.txt`.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`MultiLabelBinarizer`, `MinMaxScaler`, `cosine_similarity`)

## Possible Improvements

- Add a hybrid approach combining content-based filtering with collaborative filtering if user rating history becomes available.
- Let users weight criteria themselves (e.g. prioritize cuisine over price).
- Wrap in a small Streamlit/Flask app for an interactive demo.

## Author

**Jeeva Karnan** — B.Tech, Artificial Intelligence and Data Science
Cognifyz Technologies ML Internship — Task 2: Restaurant Recommendation

#cognifyz #cognifyzTech #cognifyzTechnologies
