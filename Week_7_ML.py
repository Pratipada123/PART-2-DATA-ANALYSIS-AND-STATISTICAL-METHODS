#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Week 7: Introduction to Machine Learning (ML)
# Theory: Study supervised vs. unsupervised learning, types of ML algorithms (regression, classification, clustering), and an introduction to scikit-learn.
# Hands-On: Implement a basic machine learning model using scikit-learn (e.g.Linear Regression).
# Client Project: Create a prediction model for client data (e.g., house prices).


# **Introduction to Machine Learning**
# - Machine Learning is a subset of Artificial Intelligence that enables systems to learn from data and improve performance over time without being explicitly programmed.

# **1. Theory: Study supervised vs. unsupervised learning, types of MLalgorithms(regression,classification,clustering), and an introduction to scikit-learn.**

# **Supervised Learning**

# - A type of machine learning where the model is trained on a labeled dataset.
# - The algorithm learns the mapping between inputs (features) and outputs (labels).
# - The goal is Predict outcomes (labels) for new, unseen data.
# - The example of Algorithms is:
#     - Linear Regression
#     - Logistic Regression
#     - Decision Trees
#     - Random Forest
#     - Support Vector Machines (SVM)
#     - Neural Networks

# **Unsupervised Learning**

# - A type of machine learning where the model is trained on unlabeled data.
# - The algorithm tries to find hidden patterns, structures, or groupings in the data.
# - The goal is Discover structure in data (clustering, grouping, dimensionality reduction).
# - The Examples of Algorithms is:
#     - K-Means Clustering
#     - Hierarchical Clustering
#     - DBSCAN
#     - Principal Component Analysis (PCA)
#     - Autoencoders

# **Types of ML algorithms**
# - Regression
# - Classification
# - Clustering

# **Regression Algorithm**
# - Regression algorithms are a type of supervised learning used to predict a continuous (numeric) value based on input features.
# - In regression we try to find relationships between independent variables (X) and a dependent variable (Y).
# - Example:
#     - Predicting house price (Y) based on area, number of rooms, location (X).
#     - Predicting temperature, stock price, salary, etc.
# - The regression algorithm is different types such as:
#     - Linear Regression – fits a straight line to predict continuous values.
#     - Multiple Linear Regression – uses multiple input features for prediction.
#     - Polynomial Regression – models non-linear relationships using polynomial terms.
#     - Ridge Regression – linear regression with L2 regularization to reduce overfitting.
#     - Lasso Regression – linear regression with L1 regularization for feature selection.
#     - Elastic Net Regression – combination of Ridge and Lasso regularization.
#     - Support Vector Regression (SVR) – regression using SVM principles.
#     - Decision Tree Regression – predicts values by splitting data into regions.
#     - Random Forest Regression – ensemble of decision trees for more accuracy.
#     - Gradient Boosting Regression (XGBoost, LightGBM, CatBoost) – sequential tree-based models improving over errors.

# **Classification Algorithm**
# - Classification algorithms are a type of supervised learning that predict a category (class label) instead of a continuous value.
# - Output is discrete (e.g., Yes/No, Spam/Not Spam, Disease A/B/C).
# - The example is:
#     - Email spam detection (spam or not spam).
#     - Medical diagnosis (disease type).
# - The classification algorithm is different types such as
#     - Logistic Regression – predicts class probabilities using a sigmoid function.
#     - K-Nearest Neighbors (KNN) – classifies based on the majority vote of nearest neighbors.
#     - Decision Tree – splits data into branches to classify outcomes.
#     - Random Forest – ensemble of decision trees for better accuracy.
#     - Naïve Bayes – uses Bayes’ theorem assuming feature independence.
#     - Support Vector Machine (SVM) – finds the best hyperplane separating classes.
#     - Gradient Boosting (XGBoost, LightGBM, CatBoost) – builds sequential trees to improve predictions.
#     - Neural Networks (Deep Learning) – multi-layered models for complex classification tasks.

# **Clustering Algorithm**
# - Clustering is an unsupervised learning technique where the goal is to group similar data points together into clusters without using labels.
# - Output = Groups (clusters).
# - The example is:
#     - Grouping customers based on shopping behavior.
# - The clustering algorithm is different types such as
#     - K-Means – partitions data into K clusters using centroids.
#     - Hierarchical Clustering – builds a tree of nested clusters.
#     - DBSCAN – finds clusters in dense regions and detects outliers.
#     - Gaussian Mixture Model (GMM) – clusters using probability distributions.
#     - Mean-Shift – groups data by shifting points toward dense regions.

# **Introduction to Scikit-Learn**
# - Scikit-Learn (sklearn) is a popular open-source Python library for Machine Learning.
# - It provides easy-to-use tools for data preprocessing, model training, evaluation, and selection.
# - Supports classification, regression, clustering, and dimensionality reduction.

# - Features of scikit-learn is:
#     - Built-in datasets (e.g., Iris, Digits).
#     - Preprocessing tools (scaling, encoding, splitting).
#     - Implementation of algorithms (regression, classification, clustering).
#     - Model evaluation (accuracy, confusion matrix, cross-validation).

# **Example**

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a simple dataset (house size vs. price)
# House size in square feet
X = np.array([500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000]).reshape(-1, 1)

# House price (in $1000s)
y = np.array([150, 200, 240, 270, 310, 360, 400, 480, 550])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price ($1000s)")
plt.legend()
plt.show()


# **2. Hands-On: Implement a basic machine learning model using scikit-learn (e.g.Linear Regression).**

# **Import Libraries**
# 
# - pandas → for data handling.
# - numpy → for math operations.
# - matplotlib / seaborn → for visualization.
# - scikit-learn → for ML model.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# **Create and Load Dataset**

# In[2]:


# Sample dataset
data = {
    "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 100000]
}
df = pd.DataFrame(data)
print(df)


# **Data Visualization**

# In[3]:


sns.scatterplot(x="YearsExperience", y="Salary", data=df)
plt.title("Years of Experience vs Salary")
plt.show()


# **Prepare Data (Features & Target)**
# 
# - X (features/input) - independent variable (YearsExperience)
# - y (target/output) - dependent variable (Salary)

# In[6]:


X = df[["YearsExperience"]] 
y = df["Salary"]    


# **Train-Test Split**

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# **Train the Linear Regression Model**

# In[8]:


model = LinearRegression()
model.fit(X_train, y_train)


# **Make Predictions**

# In[9]:


y_pred = model.predict(X_test)

print("Predicted values:", y_pred)
print("Actual values:", list(y_test))


# **Evaluate Model**

# In[10]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)


# **Visualize Regression Line**

# In[11]:


plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()


# **3. Client Project: Create a prediction model for client data (e.g., house prices).**

# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# In[38]:


# CONFIG
# -----------------------------
DATA_PATH = Path(r"C:\Users\prati\OneDrive\Datafiles\House Prices.csv")   # change here if needed
OUT_DIR = Path("/mnt/data/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


# In[40]:


df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))


# In[41]:


# 2) QUICK EDA (prints)
# -----------------------------
print("\nColumn dtypes:")
print(df.dtypes.value_counts())
print("\nMissing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

# show basic numeric summary
print("\nNumeric summary (first 8 columns):")
print(df.select_dtypes(include=[np.number]).describe().T.head(8).to_string())


# In[42]:


# 3) IDENTIFY TARGET & FEATURES
# -----------------------------
# common target names:

candidate_targets = ["SalePrice", "saleprice", "price", "Price", "MEDV", "target", "y"]
target = None
for t in candidate_targets:
    if t in df.columns:
        target = t
        break

if target is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns detected — please set the TARGET manually in the script.")
    target = numeric_cols[-1]
    print(f"\nNo common target name found. Falling back to last numeric column: '{target}'")

print("\nUsing target column ->", target)


# In[43]:


from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from scipy.stats import randint, uniform


# In[44]:


# features: numeric + categorical (basic auto selection)
numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target]
categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# In[45]:


# 4) TRAIN/TEST SPLIT
# -----------------------------
X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")


# In[46]:


# 5) PREPROCESSING
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])
print(numeric_transformer)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
print(categorical_transformer)

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)])
print(preprocessor)


# In[47]:


# helper: function to get feature names after ColumnTransformer
def get_feature_names_from_ct(ct: ColumnTransformer):
    """Return final feature names after ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in ct.transformers_:
        if transformer == 'drop' or transformer == 'passthrough':
            continue
        # get last step if pipeline
        trans = transformer
        if hasattr(transformer, "named_steps"):
            # last transform step
            last = list(transformer.named_steps.keys())[-1]
            trans = transformer.named_steps[last]
        try:
            names = trans.get_feature_names_out(cols)
        except Exception:
            # fallback
            names = cols
        # ensure list
        feature_names.extend(list(names))
    return feature_names


# In[48]:


# 6) MODELS: baseline & CV
try:
    import lightgbm as lgb
except Exception:
    lgb = None

models = {
    "ridge": Pipeline([("preproc", preprocessor), ("model", Ridge(random_state=42))]),
    "rf":    Pipeline([("preproc", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
}
if lgb is not None:
    models["lgbm"] = Pipeline([("preproc", preprocessor), ("model", lgb.LGBMRegressor(random_state=42))])

print("\nQuick cross-validated RMSE (3-fold) on training set:")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f" {name:6s} RMSE = {-scores.mean():.4f}")


# In[49]:


# 7) OPTIONAL: QUICK TUNING for RandomForest
import time
print("\nStarting a quick RandomizedSearchCV for RandomForest (n_iter=20) ...")
rf_param_dist = {
    "model__n_estimators": randint(50, 300),
    "model__max_depth": randint(3, 30),
    "model__min_samples_split": randint(2, 10),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["auto", "sqrt", "log2", 0.5]
}
rf_pipe = models["rf"]
rs = RandomizedSearchCV(rf_pipe, rf_param_dist, n_iter=20, scoring="neg_root_mean_squared_error",
                        cv=3, random_state=42, n_jobs=-1, verbose=0)
t0 = time.time()
rs.fit(X_train, y_train)
t1 = time.time()
print(f"RandomizedSearch done in {(t1-t0):.1f}s. Best CV RMSE = {-rs.best_score_:.4f}")
print("Best params:", rs.best_params_)


# In[50]:


# 8) FINAL EVALUATION ON TEST SET
# Fit baseline ridge and tuned RF (and lgbm if present), evaluate:
evaluators = {}
# fit ridge
models["ridge"].fit(X_train, y_train)
evaluators["ridge"] = models["ridge"]

# tuned RF
evaluators["rf_tuned"] = rs.best_estimator_
def evaluate(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "preds": preds
    }

results = {}
for name, pipe in evaluators.items():
    results[name] = evaluate(pipe, X_test, y_test)
    print(f"\n{name} -> RMSE: {results[name]['rmse']:.4f}, MAE: {results[name]['mae']:.4f}, R2: {results[name]['r2']:.4f}")
# choose best by RMSE
best_name = min(results.items(), key=lambda x: x[1]["rmse"])[0]
best_pipe = evaluators[best_name]
print("\nBest model on test set:", best_name)


# In[51]:


# 9) FEATURE IMPORTANCE (if tree model)
feat_names = get_feature_names_from_ct(best_pipe.named_steps["preproc"])
if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
    importances = best_pipe.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(fi.head(15).to_string())
    # save to CSV
    fi.head(50).to_csv(OUT_DIR / "feature_importances.csv")
else:
    print("\nModel does not expose feature_importances_ (e.g., Ridge).")


# In[55]:


# 10) SAVE ARTIFACTS: model + metadata + plot
import json
model_path = OUT_DIR / "model.joblib"
metadata_path = OUT_DIR / "model_metadata.json"
plot_path = OUT_DIR / "actual_vs_pred.png"

joblib.dump(best_pipe, model_path)
meta = {
    "model_name": best_name,
    "target": target,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "results_test": {k: {"rmse": v["rmse"], "mae": v["mae"], "r2": v["r2"]} for k, v in results.items()},
    "trained_at": time.ctime()
}
with open(metadata_path, "w") as f:
    json.dump(meta, f, indent=2)

# actual vs predicted plot
y_test_vals = y_test.values
preds = results[best_name]["preds"]
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test_vals, y=preds, alpha=0.7)
minv = min(y_test_vals.min(), preds.min())
maxv = max(y_test_vals.max(), preds.max())
plt.plot([minv, maxv], [minv, maxv], ls="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout()
plt.savefig(plot_path)
print(f"\nSaved model to {model_path}")
print(f"Saved metadata to {metadata_path}")
print(f"Saved plot to {plot_path}")

