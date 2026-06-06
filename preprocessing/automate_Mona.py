import pandas as pd
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df):
    print("Cleaning data...")
    # Menghapus duplikasi
    df_cleaned = df.drop_duplicates().copy()
    
    # Menghapus kolom class_name karena tidak diperlukan untuk modelling
    if 'class_name' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['class_name'])
        
    print(f"Data shape after cleaning: {df_cleaned.shape}")
    return df_cleaned

def save_data(df, output_dir):
    print(f"Saving preprocessed data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'banknote_preprocessed.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved successfully: {output_path}")

def main():
    # Path konfigurasi
    input_path = 'banknote_raw/banknote_raw.csv'
    output_dir = 'preprocessing/banknote_preprocessing'
    
    # Eksekusi pipeline preprocessing
    df = load_data(input_path)
    df_cleaned = clean_data(df)
    save_data(df_cleaned, output_dir)
    print("Automate preprocessing selesai!")

if __name__ == "__main__":
    main()
