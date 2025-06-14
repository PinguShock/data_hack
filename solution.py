





import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

# ----------------------------
# 1. Load and Preprocess Data
# ----------------------------
def load_data():
    # Load your dataset (replace with actual file paths)
    raw = pd.read_csv("278_labelled_uri_train.csv")
    valid_df = pd.read_csv("278_labelled_uri_test_no_label.csv", sep=";")

    # Drop redundant/unnamed columns and 'uri' (identifier, not a feature)
    raw = raw.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'uri'])
    valid_df = valid_df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'uri'])

    # Separate features and labels
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(raw, test_size=0.2, random_state=42)
    print(len(df_train))
    print(len(df_test))


    X_train = df_train.drop(columns=['labels'])
    y_train = df_train['labels']
    X_test = df_test.drop(columns=['labels'])
    y_test = df_test['labels']
    X_valid = valid_df  # Test data has no 'labels'

    # Handle missing values (if any)
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]  # Align indices after dropping
    X_test = X_test.dropna()

    return X_train, y_train, X_test, y_test, X_valid

# ----------------------------
# 2. Define Model and Parameters
# ----------------------------
def train_model(X, y):
    params = {
        'n_estimators': 2000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 1.0,
        'reg_lambda': 0.1,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

    params2 = {
        'n_estimators': 2000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 1.0,
        'reg_lambda': 0.1,
        'objective': 'binary:logistic',
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params2)

    # Time-based cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    print(f"Cross-Validation RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

    return model

# ----------------------------
# 3. Train and Predict
# ----------------------------
def main():
    # Load and preprocess data
    X_train, y_train, X_test, y_test, X_valid = load_data()

    # Train model
    model = train_model(X_train, y_train)

    # Fit final model on full training data
    model.fit(X_train, y_train)

    # Save model (optional)
    joblib.dump(model, 'spotify_xgb_model.pkl')


    # Predict on test set
    y_pred = model.predict(X_test)

    # results
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    matches = (y_pred == y_test)
    print(f'Matches: {matches.sum()} of {len(y_pred)}')
    print(f'Accuracy: {matches.sum() / len(y_test)}')

    # Predict on valid set
    y_pred = model.predict(X_valid)

    # Generate submission (assuming 'uri' is in test_df as ID)
    valid_df = pd.read_csv("278_labelled_uri_test_no_label.csv", sep=";")  # Reload test data to get 'uri' as IDs
    submission = pd.DataFrame({
        'ID': valid_df['Unnamed: 0'],  # Use 'uri' as the submission ID
        'Label': y_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to 'submission.csv'")

# ----------------------------
# 4. Run the Pipeline
# ----------------------------
if __name__ == '__main__':
    main()