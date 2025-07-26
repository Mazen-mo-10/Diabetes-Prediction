import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import pickle
from constants import TARGET, TOP_FEATURES


csv_path = "cleaned_data.csv"
model_path = "xgb_model.pkl"

df = pd.read_csv(csv_path)
X = df[TOP_FEATURES]
y = df[TARGET].map({1:1, 2:0, 3:2, 4:3}).dropna() 
X = X.loc[y.index]

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

print("Training the model...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

with open(model_path, 'wb') as file:
    pickle.dump(xgb, file)
print(f"Model saved as '{model_path}'")