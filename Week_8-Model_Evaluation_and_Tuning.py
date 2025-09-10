# Week 8: Model Evaluation and Tuning
# Theory: Study performance metrics for ML models, cross-validation, and hyperparameter tuning (GridSearchCV).
# Hands-On: Evaluate a model’s performance using accuracy,precision,recall and F1-score.Use GridSearchCV to tune hyperparameters.
# Client Project: Tune a model to improve its performance.

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# Load dataset
X, y = load_iris(return_X_y=True)

# Define model
svm = SVC()

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# GridSearchCV with 5-fold CV
grid = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit
grid.fit(X, y)
# Results
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
print("Best Model:", grid.best_estimator_)

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 2: Load dataset
X, y = load_iris(return_X_y=True)

# Split data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Define the model
log_reg = LogisticRegression(max_iter=500)

# Step 4: Define parameter grid for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],         # Regularization strength
    'solver': ['liblinear', 'lbfgs'] # Optimization solvers
}

# Step 5: Apply GridSearchCV
grid = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# Step 6: Evaluate on test data
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Step 7: Compute evaluation metrics
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Test Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("Test F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 1 — Imports all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from time import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import joblib

# Step 2 — Load data and quick EDA
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Features:", X.shape)
print("Classes distribution:\n", y.value_counts(normalize=True))
X.head()

# Step 3 — Create a robust evaluation function and baseline CV
def evaluate_cv(model, X, y, cv=5):
    scoring = ['accuracy','precision','recall','f1','roc_auc']
    start = time()
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    elapsed = time() - start
    summary = {k: np.mean(v) for k, v in scores.items() if k.startswith('test_')}
    summary['fit_time'] = np.mean(scores['fit_time'])
    summary['total_time_sec'] = round(elapsed, 2)
    return summary

# Baseline pipeline: scaler + logistic regression (simple, interpretable baseline)
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

print("Baseline CV results (Stratified 5-fold):")
print(evaluate_cv(baseline_pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)))

# Step 4 — Train/test split for later holdout evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print("Train/test shapes:", X_train.shape, X_test.shape)

# Step 5 — Quick baseline fit and test evaluation
baseline_pipe.fit(X_train, y_train)
y_pred = baseline_pipe.predict(X_test)
y_proba = baseline_pipe.predict_proba(X_test)[:,1]

print("Classification report (baseline on holdout):")
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC (holdout):", roc_auc_score(y_test, y_proba))

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'Baseline (AUC={roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Baseline')
plt.legend()
plt.grid(True)
plt.show()

# Step 6 — Use RandomForest + RandomizedSearchCV to quickly explore hyperparameters
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),    # not required for RF but keeps pipeline consistent
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
param_dist = {
    'clf__n_estimators': [50, 100, 200, 400],
    'clf__max_depth': [None, 5, 10, 20, 30],
    'clf__min_samples_split': [2, 4, 8, 12],
    'clf__min_samples_leaf': [1, 2, 4, 6],
    'clf__max_features': ['sqrt', 'log2', 0.2, 0.5, None]
}
rand_search = RandomizedSearchCV(
    rf_pipe,
    param_dist,
    n_iter=30,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rand_search.fit(X_train, y_train)
print("Best params (RandomizedSearch):")
pprint(rand_search.best_params_)
print("Best CV AUC:", rand_search.best_score_)

# Evaluate the best random-forest found on the holdout set
best_rf = rand_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:,1]

print("RandomForest holdout report:")
print(classification_report(y_test, y_pred_rf, digits=4))
print("ROC AUC (RF holdout):", roc_auc_score(y_test, y_proba_rf))

# Plot ROC of baseline vs RF
plt.figure(figsize=(6,4))
fpr_b, tpr_b, _ = roc_curve(y_test, baseline_pipe.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.plot(fpr_b, tpr_b, label=f'Baseline LR (AUC={roc_auc_score(y_test, baseline_pipe.predict_proba(X_test)[:,1]):.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Best RF (AUC={roc_auc_score(y_test, y_proba_rf):.3f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Baseline vs Tuned RF'); plt.legend(); plt.grid(True); plt.show()

# Step 7 — Grid search to finely tune Logistic Regression (smaller search space, deterministic)
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=42, max_iter=10000))
])
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2'],
    'clf__class_weight': [None, 'balanced']
}
grid_lr = GridSearchCV(
    lr_pipe,
    param_grid,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)
grid_lr.fit(X_train, y_train)
print("Best LR params:", grid_lr.best_params_)
print("Best LR CV AUC:", grid_lr.best_score_)

# Evaluate tuned LR on holdout
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)
y_proba_lr = best_lr.predict_proba(X_test)[:,1]
print("Logistic Regression holdout report:")
print(classification_report(y_test, y_pred_lr, digits=4))
print("ROC AUC (LR holdout):", roc_auc_score(y_test, y_proba_lr))

# Step 8 — Feature importance / selection: use the tuned RandomForest to get importances
rf_model = best_rf.named_steps['clf']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 features by importance:")
print(importances.head(10))

# Example: automatic selection via SelectFromModel
selector = SelectFromModel(rf_model, threshold='median', prefit=True)
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features (median threshold):", selected_features)

# Re-train a model using only selected features and check if performance holds or improves
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

clf_sel = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))])
clf_sel.fit(X_train_sel, y_train)
y_proba_sel = clf_sel.predict_proba(X_test_sel)[:,1]
print("Selected-features RF AUC (holdout):", roc_auc_score(y_test, y_proba_sel))

# Step 9 — Small ensemble: VotingClassifier combining tuned LR and tuned RF
voting = VotingClassifier(estimators=[
    ('lr', best_lr.named_steps['clf']),
    ('rf', best_rf.named_steps['clf'])
], voting='soft', n_jobs=-1)

# Need to wrap voting in a pipeline if scaling required for LR
voting_pipe = Pipeline([('scaler', StandardScaler()), ('voting', voting)])
voting_pipe.fit(X_train, y_train)
y_proba_v = voting_pipe.predict_proba(X_test)[:,1]
print("Voting ensemble ROC AUC (holdout):", roc_auc_score(y_test, y_proba_v))

# Step 10 — Save the best performing model (choose whichever has best holdout AUC)
results = {
    'baseline': roc_auc_score(y_test, baseline_pipe.predict_proba(X_test)[:,1]),
    'lr_tuned': roc_auc_score(y_test, y_proba_lr),
    'rf_tuned': roc_auc_score(y_test, y_proba_rf),
    'selected_rf': roc_auc_score(y_test, y_proba_sel),
    'voting': roc_auc_score(y_test, y_proba_v)
}
pprint(results)
best_key = max(results, key=results.get)
print("Best on holdout:", best_key, results[best_key])

# Map to estimator
est_map = {
    'baseline': baseline_pipe,
    'lr_tuned': best_lr,
    'rf_tuned': best_rf,
    'selected_rf': clf_sel,
    'voting': voting_pipe
}
best_model = est_map[best_key]

joblib.dump(best_model, f'best_model_{best_key}.joblib')
print(f"Saved best model to best_model_{best_key}.joblib")

# Step 11 — Quick diagnostic: plot learning curve (train vs validation score)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv, scoring='roc_auc', title='Learning curve'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_mean, 'o-', label='Train')
    plt.plot(train_sizes, test_mean, 'o-', label='CV')
    plt.xlabel('Train size'); plt.ylabel(scoring)
    plt.title(title); plt.legend(); plt.grid(True); plt.show()
plot_learning_curve(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='roc_auc', title='Learning Curve of Best Model')