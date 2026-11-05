
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# 1. Load dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("IRIS FLOWER CLASSIFICATION")
print("-" * 40)

# 2. Understand dataset
print("\nDataset Information:")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Flower classes:", iris.target_names)

print("\nFirst five records:")
print(X.head())

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# 6. Train model
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nMODEL EVALUATION")
print("-" * 40)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Iris Flower Classification - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.show()

# 10. Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\nNEW FLOWER PREDICTION")
print("-" * 40)
print("Predicted species:", iris.target_names[prediction[0]])
print("\nK VALUE COMPARISON")
print("-" * 40)

for k in [3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    score = accuracy_score(y_test, predictions)

    print(f"K = {k}, Accuracy = {score * 100:.2f}%")
