import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier  # Swapped RandomForest with GradientBoosting
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss

# --- 1. Data Generation and Splitting ---
np.random.seed(42)  # Changed seed slightly and kept original data seed

# Generate data
X, y = make_blobs(
    n_samples=2000, n_features=2, centers=3, random_state=42, cluster_std=5.0
)

# Splitting data (same sizes as original example)
X_train, y_train = X[:600], y[:600]  # Training set for base estimator
X_valid, y_valid = X[600:1000], y[600:1000]  # Validation set for calibrator fitting
X_test, y_test = X[1000:], y[1000:]  # Test set for evaluation
X_full_train, y_full_train = X[:1000], y[:1000]  # Full training set for UNCALIBRATED model

# --- 2. Model Fitting and Calibration ---

# 2.1 Uncalibrated Classifier
# Trained on the combined training and validation sets (first 1000 samples)
uncal_clf = GradientBoostingClassifier(n_estimators=25, random_state=42)
uncal_clf.fit(X_full_train, y_full_train)

# 2.2 Calibrated Classifier (Two-stage process using 'prefit' strategy)
# Stage 1: Train the base estimator only on X_train
base_estimator = GradientBoostingClassifier(n_estimators=25, random_state=42)
base_estimator.fit(X_train, y_train)

# Stage 2: Calibrate using the validation set (X_valid, y_valid)
# cross_val='prefit' tells CalibratedClassifierCV that base_estimator is already trained.
cal_clf = CalibratedClassifierCV(base_estimator, method="sigmoid", cv="prefit")
cal_clf.fit(X_valid, y_valid)

# Get predictions on the test set
uncal_probs = uncal_clf.predict_proba(X_test)
cal_probs = cal_clf.predict_proba(X_test)

# --- 3. Metric Comparison ---

print("--- Comparison of Log Loss ---")
logloss_uncal = log_loss(y_test, uncal_probs)
logloss_cal = log_loss(y_test, cal_probs)

print(f"Uncalibrated Classifier Log Loss: {logloss_uncal:.3f}")
print(f"Calibrated Classifier Log Loss: {logloss_cal:.3f}")

print("\n--- Comparison of Brier Score ---")
brier_uncal = brier_score_loss(y_test, uncal_probs)
brier_cal = brier_score_loss(y_test, cal_probs)

print(f"Uncalibrated Classifier Brier Score: {brier_uncal:.3f}")
print(f"Calibrated Classifier Brier Score: {brier_cal:.3f}")

# --- 4. Plotting Utilities (Simplex Configuration) ---
colors = ["r", "g", "b"]


def draw_simplex_annotations(ax):
    """Draws vertices, center point, and boundary annotations on the simplex."""
    # Plot perfect predictions, at each vertex
    ax.plot([1.0], [0.0], "ro", ms=20, label="Class 1")
    ax.plot([0.0], [1.0], "go", ms=20, label="Class 2")
    ax.plot([0.0], [0.0], "bo", ms=20, label="Class 3")

    # Plot boundaries of unit simplex
    ax.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

    # Annotate center and border points (simplified annotations for conciseness)
    ax.annotate(r"($\frac{1}{3}$, $\frac{1}{3}$)", xy=(1.0 / 3, 1.0 / 3), xytext=(1.0 / 3, 0.23),
                arrowprops=dict(facecolor="black", shrink=0.05), horizontalalignment="center")
    ax.plot([1.0 / 3], [1.0 / 3], "ko", ms=5)

    # Add grid lines
    ax.grid(False)
    for x in np.linspace(0.0, 1.0, 11):
        ax.plot([0, x], [x, 0], "k", alpha=0.2)
        ax.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
        ax.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

    ax.set_xlabel("Probability Class 1")
    ax.set_ylabel("Probability Class 2")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")


# --- 5. Plot 1: Probabilities Change on Test Samples ---
fig, ax = plt.subplots(figsize=(10, 10))

# Plot arrows showing change (uncalibrated -> calibrated)
for i in range(uncal_probs.shape[0]):
    ax.arrow(
        uncal_probs[i, 0],
        uncal_probs[i, 1],
        cal_probs[i, 0] - uncal_probs[i, 0],
        cal_probs[i, 1] - uncal_probs[i, 1],
        color=colors[y_test[i]],
        head_width=1e-2,
    )

draw_simplex_annotations(ax)
ax.set_title("Probability Shift on Test Samples after Sigmoid Calibration (Gradient Boosting)")
# plt.show() # Uncomment to display the first plot


# --- 6. Plot 2: Learned Calibration Map on a Grid ---
fig, ax = plt.subplots(figsize=(10, 10))

# Generate grid of probability values
p1d = np.linspace(0, 1, 20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

# Compute calibrated probabilities for the grid
calibrators_list = cal_clf.calibrated_classifiers_[0].calibrators

prediction = np.vstack(
    [
        calibrator.predict(p[:, i].reshape(-1, 1))  # Need to reshape for predict
        for i, calibrator in enumerate(calibrators_list)
    ]
).T

# Re-normalize the calibrated predictions
prediction /= prediction.sum(axis=1)[:, None]

# Plot changes in predicted probabilities induced by the calibrators
for i in range(prediction.shape[0]):
    ax.arrow(
        p[i, 0],
        p[i, 1],
        prediction[i, 0] - p[i, 0],
        prediction[i, 1] - p[i, 1],
        head_width=1e-2,
        color=colors[np.argmax(p[i])],
    )

draw_simplex_annotations(ax)
ax.set_title("Learned Sigmoid Calibration Map (Gradient Boosting)")

plt.show()
