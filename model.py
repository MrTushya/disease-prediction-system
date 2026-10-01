import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# LOAD DATA
train_data = pd.read_csv("Training.csv")
test_data = pd.read_csv("Testing.csv")

train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]

train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()

# SPLIT
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']

X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# ENCODE
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# MODEL COMPARISON
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_acc = 0

print("\nModel Comparison:")

for name, model in models.items():
    model.fit(X_train, y_train_encoded)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_encoded, preds)

    print(f"{name}: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_model = model

# CROSS VALIDATION
scores = cross_val_score(best_model, X_train, y_train_encoded, cv=5)
print("\nCross Validation Accuracy:", scores.mean())

# FINAL EVALUATION
y_pred = best_model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test_encoded, y_pred))

# FEATURE IMPORTANCE
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]

    print("\nTop Important Symptoms:")
    for i in indices:
        print(X_train.columns[i])

# SAVE
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("\nModel saved successfully!")