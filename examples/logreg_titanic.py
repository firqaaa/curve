#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import wandb
import pandas as pd
from models.logreg import LogisticRegression
from utils.preproc import StandardScaler, LabelEncoder
from utils.model_selection import train_test_split


def preproc(df):
    # Create a copy of the dataframe
    data = df.copy()

    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # Select features for the model
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']

    # Encode categorical variables
    le = LabelEncoder()
    for feature in ['Sex', 'Embarked']:
        data[feature] = le.fit_transform(data[feature])
    
    return data[features]


def main():
    # Initialize wandb
    wandb.init(
        project='mlx-logreg-titanic',
        config={
            "learning_rate": 0.1,
            "epochs": 1000,
            "batch_size": "full",
            "architecture": "logistic_regression",
            "dataset": "titanic"
        }
    )

    # Load titanic dataset
    train_data = pd.read_csv('../data/titanic/train.csv')
    test_data = pd.read_csv('../data/titanic/test.csv')
    labels_data = pd.read_csv('../data/titanic/labels.csv')
    test_data.merge(labels_data, on='PassengerId', how='left')

    # Preprocess data
    X = preproc(train_data)
    y = train_data['Survived'].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)

    # Convert to MLX arrays
    X_train_mlx = mx.array(X_train_scaled)
    y_train_mlx = mx.array(y_train)
    X_val_mlx = mx.array(X_val_scaled)
    y_val_mlx = mx.array(y_val)

    # Create and train model
    model = LogisticRegression(
        input_dim=X_train_scaled.shape[1],
        learning_rate=wandb.config.learning_rate,
        num_epochs=wandb.config.epochs
    )

    # Train model with wandb logging
    model.fit(X_train_mlx, y_train_mlx, X_val_mlx, y_val_mlx, wandb.run, logging=True)

    # Final evaluation
    final_val_pred = model.predict(X_val_mlx)
    final_accuracy = mx.mean((final_val_pred >= 0.5).astype(mx.float32) == y_val_mlx)

    # Log final metrics
    wandb.log({
        "final_val_accuracy": final_accuracy.item()
    })

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()