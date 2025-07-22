import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
BASE_DIR = Path(__file__).parent.resolve()
csv_file = BASE_DIR / 'diabetes.csv'
dataset = pd.read_csv(csv_file)

# Replace zeroes with NaN for selected columns where zero is invalid
cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
dataset[cols_with_zero_invalid] = dataset[cols_with_zero_invalid].replace(0, np.nan)

# Fill NaNs with column means
for col in cols_with_zero_invalid:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

# Optional: visualize correlation heatmap (comment out if running in non-GUI environment)
plt.figure(figsize=(10, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Selected features for the model
selected_features = ["Glucose", "BMI", "Age", "Insulin", "DiabetesPedigreeFunction"]

X = dataset[selected_features]
y = dataset["Outcome"]

# Note: Do NOT scale here manually when using pipeline, pipeline will do it internally!

# Handle class imbalance with SMOTE on raw X, y
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.20, random_state=42, stratify=y_resampled
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Model: RandomForestClassifier with GridSearchCV hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "classifier__n_estimators": [50, 100, 150],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_split": [2, 5]
}

# Create a pipeline that includes MinMaxScaler and the classifier placeholder
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", rf)
])

grid_rf = GridSearchCV(pipeline, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

print("Best RF Params:", grid_rf.best_params_)

# Evaluate on test data
y_pred = grid_rf.predict(X_test)
print("Random Forest with Pipeline Evaluation")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Optional: Evaluate Logistic Regression similarly if needed
# ...

# Save the entire pipeline (scaler + best model) to one file for deployment
best_pipeline = grid_rf.best_estimator_

with open('Diabetesmodel.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

print("Model pipeline saved to 'Diabetesmodel.pkl'")

