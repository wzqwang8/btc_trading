import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load and process data
file_path = f'M:/24.Naphtha/Python scripts/bot_testing/btc_gbp_hourly.csv'
btc = pd.read_csv(file_path)
btc['time'] = pd.to_datetime(btc['time'])
btc.set_index('time', inplace=True)

# Calculate indicators
btc['ema12'] = btc['close'].ewm(span=12, adjust=False).mean()
btc['ema26'] = btc['close'].ewm(span=26, adjust=False).mean()
btc['macd'] = btc['ema12'] - btc['ema26']
btc['std'] = btc['close'].rolling(window=14).std()

delta = btc['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
btc['RSI'] = 100 - (100 / (1 + rs))

btc['obv'] = (np.sign(btc['close'].diff()) * btc['volume']).fillna(0).cumsum()

# Create target
btc['target'] = (btc['close'].shift(-8) >= btc['close'] * 1.01).astype(int)

# Drop rows with NaNs
btc = btc.dropna()
btc = btc.drop(columns=['high', 'low', 'open', 'ema26'], errors='ignore')
# Features and labels
x = btc.drop(columns=['target'])
y = btc['target']

corr_matrix = x.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.show()


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Time Series Split and SMOTE
tscv = TimeSeriesSplit(n_splits=5)

# Cross-validation with SMOTE
f1_scorer = make_scorer(f1_score)
model_scores = {}

for name, model in models.items():
    print(f"\n=== Cross-validating {name} ===")
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply SMOTE to training data
        smote = SMOTE(random_state=42)
        x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

        # Fit and predict
        model.fit(x_train_res, y_train_res)
        y_pred = model.predict(x_test)

        # Store metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    # Print average metrics
    avg_f1 = np.mean(f1_scores)
    model_scores[name] = avg_f1
    print("Avg Accuracy: ", round(np.mean(acc_scores), 3))
    print("Avg Precision:", round(np.mean(prec_scores), 3))
    print("Avg Recall:   ", round(np.mean(rec_scores), 3))
    print("Avg F1 Score: ", round(avg_f1, 3))

# === Select Best Model Based on F1 Score ===
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
print(f"\nBest model selected for SHAP: {best_model_name}")

# === SHAP Analysis ===
best_model.fit(x, y)  # Fit on entire dataset
explainer = shap.Explainer(best_model, x)
shap_values = explainer(x)

shap.summary_plot(shap_values, x)
