import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Load dataset Iris dan konversi ke DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    os.makedirs('iris_raw', exist_ok=True)
    df.to_csv('iris_raw/iris_raw.csv', index=False)
    print(f"[1/4] Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def clean_data(df):
    """Hapus duplikat dan tangani missing values."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[2/4] Cleaning: {before - after} duplikat dihapus")

    feature_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(f"       Missing values: {df.isnull().sum().sum()}")
    return df

def split_data(df):
    """Split data menjadi train dan test set."""
    feature_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']
    X = df[feature_cols]
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[3/4] Split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def scale_and_save(X_train, X_test, y_train, y_test):
    """Standarisasi fitur dan simpan hasil preprocessing."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    feature_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']

    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df['species'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df['species'] = y_test.values

    os.makedirs('preprocessing/iris_preprocessing', exist_ok=True)
    train_df.to_csv('preprocessing/iris_preprocessing/iris_train.csv', index=False)
    test_df.to_csv('preprocessing/iris_preprocessing/iris_test.csv',   index=False)

    print(f"[4/4] Saved:")
    print(f"       preprocessing/iris_preprocessing/iris_train.csv")
    print(f"       preprocessing/iris_preprocessing/iris_test.csv")

def main():
    print("=== Automate Preprocessing Iris Dataset ===")
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    scale_and_save(X_train, X_test, y_train, y_test)
    print("\nPreprocessing selesai!")

if __name__ == "__main__":
    main()
