import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import sklearn

# Load dataset
data = pd.read_csv("data.csv/data.csv")
data.columns = data.columns.str.strip()  # Clean column names
print("pandas :",pd.__version__)
print("sklearn :",sklearn.__version__)

# Selected features
selected_features = [
    'ROA(C) before interest and depreciation before interest',
    'Debt ratio %',
    'Net Income Flag',
    'Current Ratio',
    'Operating Gross Margin',
    'Interest Coverage Ratio (Interest expense to EBIT)',
    'Equity to Liability',
    'Net Income to Total Assets',
    'Cash Flow Per Share'
]

# Split data with stratification
df_train, df_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Bankrupt?'])
X_train = df_train[selected_features]
y_train = df_train['Bankrupt?']
X_test = df_test[selected_features]
y_test = df_test['Bankrupt?']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Define and train individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
lgbm_model = LGBMClassifier()

rf_model.fit(X_resampled, y_resampled)
xgb_model.fit(X_resampled, y_resampled)
lgbm_model.fit(X_resampled, y_resampled)

# Build hybrid ensemble model with weights
hybrid = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)],
    voting='soft',
    weights=[1, 2, 2]
)

hybrid.fit(X_resampled, y_resampled)

# Custom threshold
threshold = 0.55
y_prob = hybrid.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
print("__Final Hybrid Model__")
print("Combination of random forest,XG boost,Light gradinet boost ")
# Evaluate
print("Hybrid Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(hybrid, 'model.pkl')

# Manual Test
manual_test = [[-0.25, 0.94, 0, 0.45, 0.06, -1.0, 0.18, -0.11, -3.2]]
manual_prob = hybrid.predict_proba(manual_test)[0][1]
manual_pred = int(manual_prob >= threshold)
print("\n🔬 Manual Test Prediction:", manual_pred, "| Prob:", round(manual_prob, 3))
