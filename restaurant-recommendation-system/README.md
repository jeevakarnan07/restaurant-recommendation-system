# Restaurant Recommendation System

A content-based recommendation engine that ranks restaurants against user preferences for cuisine, budget, price range, and city.

## Project Overview

Built for the Cognifyz Technologies Machine Learning Internship (Task 2), this project converts restaurant attributes into a shared feature space and uses cosine similarity to rank the best matches for a user profile.

## How It Works

1. Load and clean 9,551 restaurant records.
2. Split multi-cuisine values into individual cuisine tags.
3. Multi-hot encode cuisines with `MultiLabelBinarizer`.
4. One-hot encode cities.
5. Scale cost, price range, and votes with `MinMaxScaler`.
6. Weight cuisine and city signals to reflect their importance.
7. Build a user-preference vector in the same feature space.
8. Rank restaurants with cosine similarity.
9. Evaluate top-N results using cuisine match rate, average rating, and average price range.

## Example Preferences

```python
user_prefs = {
    "cuisines": ["Italian"],
    "price_range": 3,
    "avg_cost_for_two": 1200,
    "city": "New Delhi",
}
```

## Current Sample Results

| Preference | Cuisine Match | Avg. Rating |
|---|---:|---:|
| Italian · mid-price · New Delhi | 100% | 3.79 |
| North Indian + Chinese · budget · New Delhi | 100% | 2.97 |
| Japanese · fine dining · any city | 100% | 3.78 |

These figures come from the committed sample run and should be regenerated if the model or dataset changes.

## Tech Stack

Python · pandas · NumPy · scikit-learn · cosine similarity

## Project Structure

```text
restaurant-recommendation-system/
├── data/
│   └── Dataset.csv
├── recommendation_system.py
├── sample_output.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
python recommendation_system.py
```

## Recommended Next Improvements

- Add a Streamlit interface for interactive recommendations.
- Persist the fitted preprocessing objects so inference uses exactly the training-time feature space.
- Allow users to control the relative weights of cuisine, budget, rating, and location.
- Add recommendation diversity so the top-N list is not overly similar.
- Add a hybrid recommender when user-item interaction history becomes available.

## Resume Description

**Restaurant Recommendation System | Python, pandas, scikit-learn** — Developed a content-based recommender for 9,551 restaurants using multi-hot cuisine encoding, feature scaling, weighted user profiles, and cosine similarity; evaluated recommendation quality with cuisine-match and rating metrics.

## Author

**Jeeva Karnan** · B.Tech Artificial Intelligence & Data Science
