import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# LOAD DATA
data = pd.read_csv("student_data.csv")

# Clean column names (VERY IMPORTANT FIX)
data.columns = data.columns.str.strip()

# Remove missing values
data = data.dropna()

# CHECK DATA
print("Dataset Shape:", data.shape)
print("Columns:", data.columns)
print("Class Distribution:\n", data['final_result'].value_counts())

# FEATURES
X = data[['attendance', 'study_time', 'assignments', 'previous_marks']]
y = data['final_result']

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MODEL
base_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Train model
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# CROSS VALIDATION
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Accuracy:", np.mean(cv_scores))

# PREDICTION
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# FEATURE IMPORTANCE
importance = model.feature_importances_
features = X.columns

print("Feature Importance:")
for f, imp in zip(features, importance):
    print(f"{f}: {round(imp, 3)}")

# SAVE MODEL
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")