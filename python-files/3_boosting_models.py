import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
print("__XGBosst__ ")
# Load the data
data = pd.read_csv("data.csv\data_2.csv")
X = data.drop("Bankrupt?", axis=1)
y = data["Bankrupt?"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- Model 1: XGBoost ---------
xgb_model = XGBClassifier(scale_pos_weight=10)  # try 10, or use dynamic formula
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("\nXGBoost Report:\n", classification_report(y_test, xgb_preds))

# --------- Model 2: LightGBM ---------
lgbm_model = LGBMClassifier(class_weight='balanced')
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
print("__Light Gradient Boosting__")
print("LightGBM Accuracy:", accuracy_score(y_test, lgbm_preds))
print("\nLightGBM Report:\n", classification_report(y_test, lgbm_preds))
