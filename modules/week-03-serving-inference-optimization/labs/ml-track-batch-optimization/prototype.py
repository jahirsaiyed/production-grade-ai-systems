"""
BEFORE: this is how the fraud model started life as a notebook cell.
No structure, no tests, no versioning, no API. Compare this to app/ to see
the same problem solved with a layered, production-ready service.
"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=2000, n_features=6, weights=[0.9, 0.1])
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

new_transaction = [0.1, 0.2, -0.3, 0.4, 0.5, -0.1]
probability = model.predict_proba([new_transaction])[0][1]
print(f"Fraud probability: {probability:.2f}")
if probability > 0.5:
    print("FLAGGED AS FRAUD")
