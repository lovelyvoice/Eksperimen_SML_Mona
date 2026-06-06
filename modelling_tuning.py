import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# KONFIGURASI DAGSHUB
# ============================================================
import dagshub
dagshub.init(repo_owner='lovelyvoice', repo_name='Eksperimen_SML_Mona', mlflow=True)

# ============================================================
# LOAD DATASET PREPROCESSING
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
# HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================
param_grid = {
    'svc__C'      : [0.1, 1, 10, 100],
    'svc__gamma'  : ['scale', 'auto', 0.01, 0.1],
    'svc__kernel' : ['rbf', 'linear', 'poly']
}

print("\nMemulai GridSearchCV...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42, probability=True))
])

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring='accuracy',
    n_jobs=1, verbose=1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"\nBest Params: {best_params}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# ============================================================
# EVALUASI MODEL TERBAIK
# ============================================================
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

# ROC-AUC (binary)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

print(f"\n=== Hasil Evaluasi Model Terbaik ===")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"ROC-AUC (OVR)  : {roc_auc:.4f}")
print(f"CV Mean Score  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Asli (0)', 'Palsu (1)']))

# ============================================================
# ARTEFAK 1: CONFUSION MATRIX PLOT
# ============================================================
def save_confusion_matrix(y_true, y_pred, path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Asli (0)', 'Palsu (1)'],
                yticklabels=['Asli (0)', 'Palsu (1)'])
    plt.title('Confusion Matrix – SVM Banknote')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix disimpan: {path}")

# ============================================================
# ARTEFAK 2: FEATURE IMPORTANCE (permutation-based)
# ============================================================
def save_feature_importance(model, X_test, y_test, feature_names, path='feature_importance.png'):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    importances = result.importances_mean

    plt.figure(figsize=(8, 5))
    colors = ['#3498db' if i == np.argmax(importances) else '#95a5a6' for i in range(len(importances))]
    bars = plt.barh(feature_names, importances, color=colors)
    plt.xlabel('Permutation Importance (Mean Accuracy Drop)')
    plt.title('Feature Importance – SVM (Permutation)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    # Get feature importances from the SVC inside the pipeline
    svc_model = model.named_steps['svc']
    if svc_model.kernel == 'linear':
        importances = np.abs(svc_model.coef_[0])
    else:
        # Fallback for non-linear kernels
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean

    importance_dict = dict(zip(feature_names, importances.tolist()))
    with open('feature_importance.json', 'w') as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Feature importance disimpan: {path}")
    return importance_dict

# ============================================================
# ARTEFAK 3: CLASSIFICATION REPORT (JSON)
# ============================================================
def save_classification_report(y_true, y_pred, path='classification_report.json'):
    report = classification_report(
        y_true, y_pred,
        target_names=['Asli (0)', 'Palsu (1)'],
        output_dict=True
    )
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Classification report disimpan: {path}")
    return report

# ============================================================
# MLFLOW MANUAL LOGGING
# ============================================================
mlflow.set_experiment("Banknote_SVM_Advanced")

with mlflow.start_run(run_name="SVM_Manual_Logging_Advanced"):

    # --- Log Parameters ---
    mlflow.log_param("kernel",     best_params['svc__kernel'])
    mlflow.log_param("C",          best_params['svc__C'])
    mlflow.log_param("gamma",      best_params['svc__gamma'])
    mlflow.log_param("cv_folds",   5)
    mlflow.log_param("test_size",  0.2)
    mlflow.log_param("random_state", 42)

    # --- Log Metrics (autolog equivalent) ---
    mlflow.log_metric("accuracy",        accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted",    recall)
    mlflow.log_metric("f1_weighted",        f1)

    # --- Log Metrics Tambahan (beyond autolog) ---
    mlflow.log_metric("roc_auc_weighted",   roc_auc)
    mlflow.log_metric("cv_mean_score",      cv_scores.mean())
    mlflow.log_metric("cv_std_score",       cv_scores.std())
    mlflow.log_metric("best_cv_score",      grid_search.best_score_)

    # --- Log Model ---
    mlflow.sklearn.log_model(best_model, artifact_path="svm_model")

    # --- Generate & Log Artefak ---
    save_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    importance_dict = save_feature_importance(
        best_model, X_test, y_test, feature_cols, 'feature_importance.png'
    )
    mlflow.log_artifact('feature_importance.png')
    mlflow.log_artifact('feature_importance.json')

    report = save_classification_report(y_test, y_pred, 'classification_report.json')
    mlflow.log_artifact('classification_report.json')

    # --- Log Tags ---
    mlflow.set_tag("model_type",  "SVM")
    mlflow.set_tag("dataset",     "Banknote")
    mlflow.set_tag("tuning",      "GridSearchCV")
    mlflow.set_tag("level",       "Advanced")

    print("\n=== MLflow Logging Selesai ===")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("Artefak yang disimpan:")
    print("  - svm_model (model)")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - feature_importance.json")
    print("  - classification_report.json")

print("\nModelling Advanced selesai! Cek DagsHub MLflow UI.")