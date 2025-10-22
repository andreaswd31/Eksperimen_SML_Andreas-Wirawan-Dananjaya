import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Fungsi-fungsi ---

def load_data(file_path):
    """Memuat data mentah dari file CSV."""
    # Kita asumsikan file mentahnya bernama 'healthcare-dataset-stroke-data.csv'
    # Pastikan file ini ada di repositorimu!
    raw_data_path = os.path.join(os.path.dirname(file_path), 'healthcare-dataset-stroke-data.csv')
    
    # Jika file mentah tidak ada, coba muat dari file yang sudah di-download
    # Ini hanya untuk fallback, idealnya script ini jalan pakai data mentah
    if not os.path.exists(raw_data_path):
        print("Peringatan: File data mentah 'healthcare-dataset-stroke-data.csv' tidak ditemukan.")
        # Jika kamu ingin skrip ini tetap jalan,
        # kamu harus meletakkan 'healthcare-dataset-stroke-data.csv' di folder yang sama
        # Untuk sekarang, kita hentikan jika file mentah tidak ada
        raise FileNotFoundError("Pastikan 'healthcare-dataset-stroke-data.csv' ada di folder proyek.")
        
    df = pd.read_csv(raw_data_path)
    print("Data mentah berhasil dimuat.")
    return df

def preprocess_data(df):
    """Melakukan semua langkah preprocessing data."""
    print("Memulai preprocessing...")
    
    # 1. Drop kolom 'id'
    df_clean = df.drop('id', axis=1)
    
    # 2. Handle 'Other' di gender (hanya ada 1 data, jadi kita drop barisnya)
    df_clean = df_clean[df_clean['gender'] != 'Other']
    print(f"Ukuran data setelah drop 'Other' di gender: {df_clean.shape}")

    # 3. Pemisahan Data
    X = df_clean.drop('stroke', axis=1)
    y = df_clean['stroke']
    
    # Split data, gunakan 'stratify=y' karena data imbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Ukuran X_train: {X_train.shape}')
    print(f'Ukuran X_test: {X_test.shape}')

    # 4. Membuat Pipeline Preprocessing
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # Pipeline untuk fitur numerik
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategorikal
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Gabungkan kedua pipeline menggunakan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough'
    )

    # 5. Terapkan pipeline
    # Fit dan transform di data latih
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    # Hanya transform di data uji
    X_test_preprocessed = preprocessor.transform(X_test)

    # Ambil nama fitur baru setelah OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    # Ubah hasil array numpy kembali ke DataFrame
    df_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names, index=X_train.index)
    df_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names, index=X_test.index)

    # Gabungkan kembali fitur (X) dengan target (y)
    df_train_final = pd.concat([df_train_preprocessed, y_train], axis=1)
    df_test_final = pd.concat([df_test_preprocessed, y_test], axis=1)

    print("Preprocessing selesai.")
    return df_train_final, df_test_final

def save_data(train_df, test_df, output_path):
    """Menyimpan data yang sudah diproses ke file CSV."""
    train_path = os.path.join(output_path, 'train_preprocessed.csv')
    test_path = os.path.join(output_path, 'test_preprocessed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data bersih berhasil disimpan di:")
    print(f"1. {train_path}")
    print(f"2. {test_path}")

# --- Fungsi Utama untuk Menjalankan Skrip ---

if __name__ == "__main__":
    # Dapatkan path direktori di mana skrip ini berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Memuat data mentah
    # Kita asumsikan file mentahnya ada di direktori yang sama
    df_raw = load_data(current_dir)
    
    # Melakukan preprocessing
    df_train, df_test = preprocess_data(df_raw)
    
    # Menyimpan data yang sudah diproses
    # Kita simpan di direktori yang sama
    save_data(df_train, df_test, current_dir)