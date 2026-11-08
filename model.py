import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

def load_and_train_models():
    # Load the real dataset CSV from the root directory
    df = pd.read_csv("ai4i2020.csv")
    
    # Clean and rename columns to standard clean identifiers
    df = df.rename(columns={
        "Air temperature [K]": "air_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "machine_failure"
    })
    
    feature_cols = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    X = df[feature_cols]
    y = df["machine_failure"]
    
    # Stratified split to preserve class balance across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest Ensemble Pipeline
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced'))
    ])
    rf_pipeline.fit(X_train, y_train)
    
    # Logistic Regression Baseline Pipeline
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    
    return rf_pipeline, lr_pipeline, df, X_test, y_test