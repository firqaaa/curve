#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import wandb
import pandas as pd
import numpy as np
from models.linreg import LinearRegression
from utils.model_selection import train_test_split
from utils.preproc import LabelEncoder, StandardScaler


def preproc(df):
    # Create a copy of dataframe
    data = df.copy()

    # Handle missing values
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].median())
    
    # Convert categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = pd.factorize(data[column])[0]
    
    # Select features for the model
    features = ["area", "bedrooms", "bathrooms", "stories", 
                "mainroad", "guestroom", "basement", "hotwaterheating", 
                "airconditioning", "parking", "prefarea", "furnishingstatus"]

    # Encode categorical variables
    le = LabelEncoder()
    for feature in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']:
        data[feature] = le.fit_transform(data[feature])
    
    return data[features]

def main():
    # Load dataset
    df = pd.read_csv('../data/house_pricing/housing.csv')
    print(df['price'])

    # Separate features and target
    X = df.drop('price', axis=1)  # Need to specify axis=1 for columns
    y = df['price']

    # Preprocess data
    X = preproc(X)

    # Log dataset info to wandb
    wandb.init(
        project="house-price-prediction",
        config={
            "dataset": "house_prices",
            "total_samples": len(X),
            "n_features": X.shape[1],
            "learning_rate": 0.01,
            "num_epochs": 1000
        }
    )

    # Scale features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42
    )

    # Create and train model
    model = LinearRegression(
        learning_rate=wandb.config.learning_rate,
        num_epochs=wandb.config.num_epochs
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Final Evaluation
    final_metrics = model.history

    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(model.weights.squeeze().tolist())
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Log final metrics and feature importance to wandb
    wandb.log({
        "final_train_r2": final_metrics['r2'][-1],
        "final_val_r2": final_metrics['val_r2'][-1],
        "final_train_rmse": final_metrics['rmse'][-1],
        "final_val_rmse": final_metrics['val_rmse'][-1],
        "feature_importance": wandb.Table(
            dataframe=feature_importance
        )
    })

    # Print top 10 most importance features
    print("\nTop 10 Most Importance Features:")
    print(feature_importance.head(10))

    wandb.finish()


if __name__ == "__main__":
    main()