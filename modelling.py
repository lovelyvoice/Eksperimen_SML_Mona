import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# LOAD DATASET
# ============================================================
train_df = pd.read_csv('banknote_preprocessing/banknote_train.csv')
test_df  = pd.read_csv('banknote_preprocessing/banknote_test.csv')

feature_cols = ['variance', 'skewness', 'curtosis', 'entropy']

X_train = train_df[feature_cols]
y_train = train_df['class']
X_test  = test_df[feature_cols]
y_test  = test_df['class']

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
param_grid = {
    'svc__C'     : [0.1, 1, 10],
    'svc__gamma' : ['scale', 'auto'],
    'svc__kernel': ['rbf', 'linear']
}

print("Memulai GridSearchCV...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42, probability=True))
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"Best Params: {best_params}")

# ============================================================
# EVALUASI
# ============================================================
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
roc_auc   = roc_auc_score(y_test, y_proba[:, 1])

# ============================================================
# ARTEFAK
# ============================================================
os.makedirs('outputs', exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Asli (0)', 'Palsu (1)'],
            yticklabels=['Asli (0)', 'Palsu (1)'])
plt.title('Confusion Matrix – SVM Banknote')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()

# Classification report
report = classification_report(
    y_test, y_pred,
    target_names=['Asli (0)', 'Palsu (1)'],
    output_dict=True
)
with open('outputs/classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ============================================================
# MLFLOW LOGGING
# ============================================================
run = mlflow.last_active_run()

mlflow.log_param("kernel",    best_params['svc__kernel'])
mlflow.log_param("C",         best_params['svc__C'])
mlflow.log_param("gamma",     best_params['svc__gamma'])
mlflow.log_param("cv_folds",  5)

mlflow.log_metric("accuracy",           accuracy)
mlflow.log_metric("precision_weighted", precision)
mlflow.log_metric("recall_weighted",    recall)
mlflow.log_metric("f1_weighted",        f1)
mlflow.log_metric("roc_auc_weighted",   roc_auc)
mlflow.log_metric("cv_mean_score",      cv_scores.mean())
mlflow.log_metric("cv_std_score",       cv_scores.std())

mlflow.sklearn.log_model(best_model, artifact_path="svm_model")
mlflow.log_artifact('outputs/confusion_matrix.png')
mlflow.log_artifact('outputs/classification_report.json')

mlflow.set_tag("model_type", "SVM")
mlflow.set_tag("dataset",    "Banknote_Authentication")

print(f"\nRun ID: {run.info.run_id if run else 'N/A'}")
print(f"Accuracy: {accuracy:.4f}")

print("\nCI Modelling selesai")