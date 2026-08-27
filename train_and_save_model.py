"""
train_and_save_model.py
------------------------
Trains the final Gradient Boosting Classifier (using the best hyperparameters
found by GridSearchCV in Section 5.4 of the report -- confirmed as the
best-performing model overall in Section 5.5 Evaluation / 7.1 Selection of
Best Model) on the full processed dataset, and saves the fitted model +
supporting metadata as a pickle file for use in the Streamlit deployment
prototype (app.py).

Run this once before launching the Streamlit app:
    python train_and_save_model.py
"""

import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# ----------------------------------------------------------------------
# 1. Load the processed dataset (output of Section 4.0 Data Preparation)
# ----------------------------------------------------------------------
df = pd.read_csv("processed_gaming_dataset.csv")

drop_cols = ["PlayerID", "GameDifficulty", "EngagementLevel", "EngagementLevel_enc", "AgeGroup"]
X = df.drop(columns=drop_cols)
y = df["EngagementLevel_enc"]  # 0 = Low, 1 = Medium, 2 = High

feature_columns = X.columns.tolist()

# ----------------------------------------------------------------------
# 2. Train the final model using the best hyperparameters from GridSearchCV
#    (Section 5.4.2, Table 5.4.2(a): n_estimators=50, learning_rate=0.05,
#     max_depth=7, min_samples_split=20, min_samples_leaf=10)
# ----------------------------------------------------------------------
final_model = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
final_model.fit(X, y)

print("Model trained on full dataset:", X.shape)
print("Training accuracy on full dataset:", final_model.score(X, y))

# ----------------------------------------------------------------------
# 3. Package the model with everything the deployment app needs:
#    - the fitted model
#    - the exact feature column order used during training
#    - encoding maps to convert raw user input into the same format
# ----------------------------------------------------------------------
difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
engagement_map_inverse = {0: "Low", 1: "Medium", 2: "High"}

genders = ["Male", "Female"]
locations = ["USA", "Europe", "Asia", "Other"]
genres = ["Action", "Strategy", "RPG", "Sports", "Simulation"]

bundle = {
    "model": final_model,
    "model_name": "Gradient Boosting Classifier",
    "feature_columns": feature_columns,
    "difficulty_map": difficulty_map,
    "engagement_map_inverse": engagement_map_inverse,
    "genders": genders,
    "locations": locations,
    "genres": genres,
}

with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved model bundle to gradient_boosting_model.pkl")
print("Feature columns saved ({} total):".format(len(feature_columns)))
print(feature_columns)
