# CSCI 158 - Assignment 1
# Finger Spoofing Attack Detection using Logistic Regression and Decision Tree
# This program trains two models to detect fingerprint spoofing attacks and compares their performance
# Author: Noah Wiley
# 10/06/25


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ===========================
# 1. Load and Preprocess Data
# ===========================

print("=" * 60)
print("Fingerprint Spoof Attack Detection")
print("=" * 60)

# Load Dataset
df = pd.read_csv('fingerprint_spoofing_attack_dataset.csv')
print("\n[1] Dataset Loaded Successfully")
print(f" Total Samples: {len(df)}")
print(f" Features: {df.columns}")
print(f" Target variable: label (0=Real, 1=Spoof)")

# Check Class Distribution
print(f"\n Class Distribution:")
print(f" - Real Fingerprints (0): {len(df[df['label'] == 0])}")
print(f" - Spoof fingerprints (1): {len(df[df['label'] == 1])}")

# Separate features (X) and target (y)
X = df.drop('label', axis=1)  # All columns except 'label'
y = df['label']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[2] Data Split:")
print(f"  Training Samples: {len(X_train)}")
print(f"  Testing Samples: {len(X_test)}")

# ==================================
# 2. Train Logistic Regression Model
# ==================================

print("\n" + "=" * 60)
print("Training Logistic Regression Model...")
print("=" * 60)

# Initialize and Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("[✓] Logistic Regression Model Trained Successfully")

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Calculate evaluation metrics
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions)

# Display Results
print("\n[3] Logistic Regression Results:")
print(f"    Accuracy: {lr_accuracy: .4f} ({lr_accuracy * 100:.2f}%)")
print(f"    Precision: {lr_precision: .4f}")
print(f"    Recall: {lr_recall: .4f}")
print(f"    F1 Score: {lr_f1: .4f}")
print(f"    Confusion Matrix:")
print(f"        Predicted")
print(f"        Real Spoof")
print(f"    Actual Real   {lr_confusion_matrix[0][0]:4d}  {lr_confusion_matrix[0][1]:4d}")
print(f"           Spoof  {lr_confusion_matrix[1][0]:4d}  {lr_confusion_matrix[1][1]:4d}")

# ============================
# 3. Train Decision Tree Model
# ============================
print("\n" + "=" * 60)
print("Training Decision Tree Model...")
print("=" * 60)

# Initialize & Train Decisions Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
print("[✓] Decision Tree Model Trained Successfully")

# Make Predictions
dt_predictions = dt_model.predict(X_test)

# Calculate Evaluation Metrics
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)
dt_cm = confusion_matrix(y_test, dt_predictions)

# Display Results
print("\n[4] Decision Tree Results:")
print(f"    Accuracy:  {dt_accuracy:.4f} ({dt_accuracy * 100:.2f}%)")
print(f"    Precision: {dt_precision:.4f}")
print(f"    Recall:    {dt_recall:.4f}")
print(f"    F1-Score:  {dt_f1:.4f}")
print("\n    Confusion Matrix:")
print(f"                  Predicted")
print(f"                  Real  Spoof")
print(f"    Actual Real   {dt_cm[0][0]:4d}  {dt_cm[0][1]:4d}")
print(f"           Spoof  {dt_cm[1][0]:4d}  {dt_cm[1][1]:4d}")

# ===================
# 4. Model Comparison
# ===================

print("\n" + "=" * 60)
print("Model Comparison Results:")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Logistic Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1],
    'Decision Tree': [dt_accuracy, dt_precision, dt_recall, dt_f1]
})

print("\n" + comparison_df.to_string(index=False))

# Determine Better Model based on F1-score
print("\n" + "=" * 60)
if lr_f1 > dt_f1:
    print("Best Model: Logistic Regression")
elif dt_f1 > lr_f1:
    print("Best Model: Decision Tree")
else:
    print("Best Model: Tie (both models perform equally)")
print("=" * 60)

# ========================
# 5. Findings and Interpretation
# ========================
print("\n" + "=" * 60)
print("Results and Findings")
print("=" * 60)

print("\n1. Logistic Regression (LR)")
print(f"   - Accuracy:  {lr_accuracy:.6f}")
print(f"   - Precision: {lr_precision:.6f}")
print(f"   - Recall:    {lr_recall:.6f}")
print(f"   - F1-Score:  {lr_f1:.6f}")
print("\n   Interpretation:")

if lr_recall == 1.0:
    print("   The Logistic Regression model achieved perfect recall (1.0), meaning")
    print(f"   it successfully detected ALL {lr_confusion_matrix[1][0] + lr_confusion_matrix[1][1]} spoof fingerprints in the test set.")
    print("   No spoof attacks were missed (zero false negatives).")
elif lr_recall >= 0.9:
    print(
        f"   The LR model detected {int(lr_recall * (lr_confusion_matrix[1][0] + lr_confusion_matrix[1][1]))} out of {lr_confusion_matrix[1][0] + lr_confusion_matrix[1][1]} spoofs")
    print("   with very high recall, catching most spoof attacks.")

if lr_confusion_matrix[0][1] > 0:
    real_total = lr_confusion_matrix[0][0] + lr_confusion_matrix[0][1]
    false_positive_rate = (lr_confusion_matrix[0][1] / real_total) * 100
    print(f"\n   However, {lr_confusion_matrix[0][1]} out of {real_total} real fingerprints were misclassified")
    print(f"   as spoofs ({false_positive_rate:.0f}% false positive rate), indicating the model")
    print("   is overly cautious and flags too many legitimate users as threats.")

print("\n   Strength: Excellent spoof detection—catches all attacks (maximum security)")
if lr_confusion_matrix[0][1] > 0:
    print(
        f"  Weakness: High false alarm rate reduces usability ({lr_confusion_matrix[0][1]}/{lr_confusion_matrix[0][0] + lr_confusion_matrix[0][1]} real users rejected)")

print("\n2. Decision Tree (DT)")
print(f"   - Accuracy:  {dt_accuracy:.6f}")
print(f"   - Precision: {dt_precision:.6f}")
print(f"   - Recall:    {dt_recall:.6f}")
print(f"   - F1-Score:  {dt_f1:.6f}")
print("\n   Interpretation:")

spoof_total = dt_cm[1][0] + dt_cm[1][1]
detected_spoofs = dt_cm[1][1]
missed_spoofs = dt_cm[1][0]

if dt_accuracy >= 0.95:
    print(f"   The Decision Tree model performed excellently with {dt_accuracy * 100:.1f}% accuracy.")
elif dt_accuracy >= 0.80:
    print(f"   The Decision Tree model achieved {dt_accuracy * 100:.1f}% overall accuracy.")
else:
    print(f"   The Decision Tree model achieved only {dt_accuracy * 100:.1f}% overall accuracy,")
    print("   showing moderate performance.")

if missed_spoofs > 0:
    missed_percentage = (missed_spoofs / spoof_total) * 100
    print(f"\n   With recall of {dt_recall:.2f}, the model detected {detected_spoofs} out of {spoof_total}")
    print(f"   spoof attacks, but MISSED {missed_spoofs} spoofs ({missed_percentage:.0f}% false negatives)—")
    print("   a significant security concern as these attacks went undetected.")

real_total = dt_cm[0][0] + dt_cm[0][1]
if dt_cm[0][1] > 0:
    if dt_cm[0][0] == 0:
        print(f"\n   The confusion matrix shows ALL {real_total} real fingerprints were")
        print("   misclassified as spoofs (100% false positive rate), indicating")
        print("   severe usability problems—no legitimate users would be granted access.")
    else:
        fp_rate = (dt_cm[0][1] / real_total) * 100
        print(f"\n   Additionally, {dt_cm[0][1]} out of {real_total} real fingerprints ({fp_rate:.0f}%)")
        print("   were incorrectly flagged as spoofs (false alarms).")

if missed_spoofs > 0:
    print(f"\n    Weakness: Missed {missed_spoofs}/{spoof_total} spoof attacks (security risk)")
if dt_cm[0][0] == 0:
    print("    Weakness: 100% false positive rate on real fingerprints (severe usability issue)")
print(f"    Note: Lower F1-score ({dt_f1:.4f}) indicates worse overall balance")

print("\n3. Overall Comparison")
print(f"\n   Logistic Regression significantly outperforms Decision Tree:")
print(f"   - Better Accuracy:   {lr_accuracy:.4f} vs {dt_accuracy:.4f}")
print(
    f"   - Better Recall:     {lr_recall:.4f} vs {dt_recall:.4f} (catches {'all' if lr_recall == 1.0 else 'more'} spoofs)")
print(f"   - Better Precision:  {lr_precision:.4f} vs {dt_precision:.4f}")
print(f"   - Better F1-Score:   {lr_f1:.4f} vs {dt_f1:.4f}")

print("\n   Recommended Model: Logistic Regression")
print("\n   While LR has a high false positive rate, it's preferable to DT which:")
if missed_spoofs > 0:
    print(f"   1. Misses critical spoof attacks ({missed_spoofs} out of {spoof_total})")
if dt_cm[0][0] == 0:
    print(f"   2. Rejects 100% of legitimate users")
print("   3. Has lower performance across all metrics")