# fraud_detection.py
# Usage: prepare CSV data with features and 'isFraud' binary label
# Run: python fraud_detection.py data/transactions.csv

import sys, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def main():
    fn = sys.argv[1] if len(sys.argv)>1 else "data/transactions.csv"
    df = pd.read_csv(fn)
    # naive preprocessing: drop non-numeric, fillna
    y = df['isFraud']
    X = df.drop(columns=['isFraud','transactionID'], errors='ignore')
    X = pd.get_dummies(X).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    joblib.dump(clf, "models/fraud_rf.joblib")
    print("Saved model to models/fraud_rf.joblib")

if __name__=="__main__":
    main()
