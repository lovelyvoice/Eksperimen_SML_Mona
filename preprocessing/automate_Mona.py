import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Load dataset Banknote Authentication dan konversi ke DataFrame."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(url, names=column_names)
    
    os.makedirs('banknote_raw', exist_ok=True)
    df.to_csv('banknote_raw/banknote_raw.csv', index=False)
    print(f"[1/4] Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def clean_data(df):
    """Hapus duplikat dan tangani missing values."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[2/4] Cleaning: {before - after} duplikat dihapus")

    feature_cols = ['variance', 'skewness', 'curtosis', 'entropy']
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(f"       Missing values: {df.isnull().sum().sum()}")
    return df

def split_data(df):
    """Split data menjadi train dan test set."""
    feature_cols = ['variance', 'skewness', 'curtosis', 'entropy']
    X = df[feature_cols]
    y = df['class']

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

    feature_cols = ['variance', 'skewness', 'curtosis', 'entropy']

    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df['class'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df['class'] = y_test.values

    os.makedirs('preprocessing/banknote_preprocessing', exist_ok=True)
    train_df.to_csv('preprocessing/banknote_preprocessing/banknote_train.csv', index=False)
    test_df.to_csv('preprocessing/banknote_preprocessing/banknote_test.csv',   index=False)

    print(f"[4/4] Saved:")
    print(f"       preprocessing/banknote_preprocessing/banknote_train.csv")
    print(f"       preprocessing/banknote_preprocessing/banknote_test.csv")

def main():
    print("=== Automate Preprocessing Banknote Dataset ===")
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    scale_and_save(X_train, X_test, y_train, y_test)
    print("\nPreprocessing selesai!")

if __name__ == "__main__":
    main()
